#!/usr/bin/env python3
"""Opt-in hard session token cap for the pinned Uni-Agent Gateway/Framework.

The upstream trajectory capacity (prompt_length + response_length) ordinarily
materializes a chain and allows the Agent to continue. This patch makes that
same capacity TERMINATE the *session*, preserving a single unfinished sample
with reward 0 rather than continuing Claude Code in a new training trajectory.

Usage:
    python patches/apply_uniagent_session_hardcap.py --check /workspace/uni-agent-cc
    python patches/apply_uniagent_session_hardcap.py /workspace/uni-agent-cc

No modifications to VERL, TPR, Claude Code or the reward function are needed.
The behavior is disabled by default and enabled only by Framework config:
    actor_rollout_ref.rollout.custom.agent_framework.hard_stop_on_trajectory_capacity=true
"""
from __future__ import annotations

import argparse
import ast
import difflib
import os
import tempfile
from pathlib import Path


class PatchError(RuntimeError):
    pass


def replace_exact(src: str, before: str, after: str, label: str) -> str:
    """Idempotent; reject drift and partial edits instead of patching nearby code."""
    if after in src:
        return src
    count = src.count(before)
    if count != 1:
        raise PatchError(f"{label}: expected one unmodified anchor, got {count}")
    return src.replace(before, after, 1)


def patch_session(src: str) -> str:
    # These anchors are shared across upstream Gateway variants, including
    # revisions using owner_request, owner_future or no request coalescing.
    # We deliberately never touch their in-flight bookkeeping.
    src = replace_exact(
        src,
        '        self._metadata = dict(metadata or {})\n'
        '        self._trace_identity = dict(self._metadata.get("_trace_identity") or {})\n',
        '        self._metadata = dict(metadata or {})\n'
        '        self._hard_stop_on_trajectory_capacity = (\n'
        '            self._metadata.get("hard_stop_on_trajectory_capacity") is True\n'
        '        )\n'
        '        self._hard_cap_reached = False\n'
        '        self._trace_identity = dict(self._metadata.get("_trace_identity") or {})\n',
        "session.__init__",
    )
    src = replace_exact(
        src,
        '                # Prepare can touch codec and multimodal extractor state, so only\n',
        '                if self._hard_stop_on_trajectory_capacity and self._hard_cap_reached:\n'
        '                    raise HTTPException(status_code=410, detail="Session trajectory capacity exhausted")\n'
        '                # Prepare can touch codec and multimodal extractor state, so only\n',
        "session.run_generation guard",
    )
    # Put flag at a shared, low-risk location *after* input preparation.
    # Both current owner_future and older/newer request bookkeeping versions
    # have the same EncodedData.capacity_exhausted predicate.
    # Coalescing revisions nest this condition at different indentation
    # depths. Match the whole line, never a suffix of a longer indentation
    # (which would yield syntactically invalid Python).
    cap_lines = [line for line in src.splitlines(keepends=True)
                 if line.strip() == "if encoded.capacity_exhausted:"]
    if len(cap_lines) != 1:
        raise PatchError(
            f"session.capacity_without_generation: expected one condition, found {len(cap_lines)}"
        )
    cap_line = cap_lines[0]
    cap_indent = cap_line[:len(cap_line) - len(cap_line.lstrip())]
    src = replace_exact(
        src,
        cap_line,
        cap_line
        + cap_indent + '    if self._hard_stop_on_trajectory_capacity:\n'
        + cap_indent + '        self._hard_cap_reached = True\n',
        "session.capacity_without_generation",
    )
    # Do not delete the committed chain here: owner_future / coalesced waiters
    # may still read it later in the same method. At finalize the active chain
    # is materialized with max_trajectory_length; the watcher stops the Runner.
    src = replace_exact(
        src,
        '                chain_id = self._commit_generation_to_chain(encoded, assistant_msg)\n',
        '                chain_id = self._commit_generation_to_chain(encoded, assistant_msg)\n'
        '                if (\n'
        '                    self._hard_stop_on_trajectory_capacity\n'
        '                    and self._trajectory_capacity is not None\n'
        '                    and len(encoded.buffer.prompt_ids) + len(encoded.buffer.response_ids)\n'
        '                    >= self._trajectory_capacity\n'
        '                ):\n'
        '                    self._hard_cap_reached = True\n',
        "session.exact_capacity_commit",
    )
    # A concurrent request can finish backend generation after another
    # request filled the budget: do not commit this response as a new chain.
    src = replace_exact(
        src,
        '                assistant_msg, finish_reason = await self._codec.decode_response(\n',
        '                if self._hard_stop_on_trajectory_capacity and self._hard_cap_reached:\n'
        '                    raise HTTPException(status_code=410, detail="Session trajectory capacity exhausted")\n'
        '                assistant_msg, finish_reason = await self._codec.decode_response(\n',
        "session.concurrent_commit_guard",
    )
    src = replace_exact(
        src,
        '    def _close_length_exhausted_chain(self, encoded: EncodedData) -> None:\n'
        '        if encoded.chain_id is None:\n',
        '    def _close_length_exhausted_chain(self, encoded: EncodedData) -> None:\n'
        '        if self._hard_stop_on_trajectory_capacity:\n'
        '            self._hard_cap_reached = True\n'
        '        if encoded.chain_id is None:\n',
        "session.close_exhausted",
    )
    src = replace_exact(
        src,
        '                    trajectory=self._build_materialized_trajectory(chain=chain),\n',
        '                    trajectory=self._build_materialized_trajectory(\n'
        '                        chain=chain,\n'
        '                        extra_fields=(\n'
        '                            {"materialization_reason": "max_trajectory_length"}\n'
        '                            if self._hard_stop_on_trajectory_capacity\n'
        '                            and self._hard_cap_reached\n'
        '                            and self._trajectory_capacity is not None\n'
        '                            and len(chain.buffer.prompt_ids) + len(chain.buffer.response_ids)\n'
        '                            >= self._trajectory_capacity\n'
        '                            else None\n'
        '                        ),\n'
        '                    ),\n',
        "session.materialize_at_exact_capacity",
    )
    src = replace_exact(
        src,
        '            "num_trajectories": len(self.materialized_chains),\n',
        '            "num_trajectories": len(self.materialized_chains),\n'
        '            "hard_cap_reached": self._hard_cap_reached,\n',
        "session.snapshot",
    )
    return src

def patch_manager(src: str) -> str:
    return replace_exact(
        src,
        '    async def shutdown(self) -> None:\n'
        '        """Stop owned gateway actors and clear routing state."""\n',
        '    async def get_session_state(self, session_id: str) -> dict:\n'
        '        """Forward the Gateway actor session snapshot without changing its lifecycle."""\n'
        '        gateway, _ = self._get_gateway(session_id)\n'
        '        return await gateway.get_session_state.remote(session_id=session_id)\n'
        '\n'
        '    async def shutdown(self) -> None:\n'
        '        """Stop owned gateway actors and clear routing state."""\n',
        "manager.get_session_state",
    )


CAP_HELPERS = '''
    async def _wait_for_session_hard_cap(self, session_id: str) -> None:
        """Observe Gateway token truth; never infer token usage from wall time."""
        while True:
            state = await self.gateway_manager.get_session_state(session_id)
            if state.get("hard_cap_reached", False):
                return
            await asyncio.sleep(0.25)

    async def _await_runner_with_hard_cap(
        self, object_ref, session_id: str, timeout: float | None
    ) -> tuple[TaskResult | None, bool]:
        """Stop Ray Task, then finalize a failed/unfinished first trajectory."""
        waiter = asyncio.ensure_future(asyncio.wait_for(object_ref, timeout=timeout))
        watcher = asyncio.create_task(self._wait_for_session_hard_cap(session_id))
        capped = TaskResult(
            reward=0.0,
            accuracy=0.0,
            finished=False,
            extra_info={"session_length_truncated": True, "materialization_reason": "max_trajectory_length"},
        )
        try:
            done, _ = await asyncio.wait({waiter, watcher}, return_when=asyncio.FIRST_COMPLETED)
            if watcher in done:
                await watcher  # Surface an actual Gateway failure, never call it a token cap.
                try:
                    await self._cancel_runner_task(object_ref, session_id)
                except asyncio.CancelledError:
                    # A cancelled Ray ObjectRef may raise CancelledError inside
                    # the cancellation helper; preserve external parent cancellation.
                    if asyncio.current_task().cancelling():
                        raise
                logger.info("session %s: hard token cap reached; Ray runner stopped", session_id)
                return capped, True
            try:
                runner_result = await waiter
            except Exception:
                # The provider may have returned HTTP 410 before the poll woke.
                if (await self.gateway_manager.get_session_state(session_id)).get("hard_cap_reached", False):
                    return capped, True
                raise
            if (await self.gateway_manager.get_session_state(session_id)).get("hard_cap_reached", False):
                return capped, True
            return runner_result, False
        finally:
            for pending in (watcher, waiter):
                if not pending.done():
                    pending.cancel()
            await asyncio.gather(watcher, waiter, return_exceptions=True)

'''
def patch_framework(src: str) -> str:
    src = replace_exact(
        src,
        '        trajectory_postprocessor_kwargs: dict[str, object] | None = None,\n'
        '    ):\n',
        '        trajectory_postprocessor_kwargs: dict[str, object] | None = None,\n'
        '        hard_stop_on_trajectory_capacity: bool = False,\n'
        '    ):\n',
        "framework.init_signature",
    )
    src = replace_exact(
        src,
        '        self._trajectory_postprocessor_kwargs = trajectory_postprocessor_kwargs or {}\n',
        '        self._trajectory_postprocessor_kwargs = trajectory_postprocessor_kwargs or {}\n'
        '        self._hard_stop_on_trajectory_capacity = hard_stop_on_trajectory_capacity\n',
        "framework.init_field",
    )
    src = replace_exact(
        src,
        '            trajectory_postprocessor_kwargs=trajectory_postprocessor_kwargs,\n'
        '        )\n',
        '            trajectory_postprocessor_kwargs=trajectory_postprocessor_kwargs,\n'
        '            hard_stop_on_trajectory_capacity=af_cfg.get("hard_stop_on_trajectory_capacity", False),\n'
        '        )\n',
        "framework.from_config",
    )
    src = replace_exact(
        src,
        '        for runner_name, runner_cfg in agent_runners_cfg.items():\n'
        '            runner_registry[str(runner_name)] = _RunnerConfig.from_config(runner_name, runner_cfg)\n',
        '        for runner_name, runner_cfg in agent_runners_cfg.items():\n'
        '            runner_registry[str(runner_name)] = _RunnerConfig.from_config(runner_name, runner_cfg)\n'
        '        hard_stop = af_cfg.get("hard_stop_on_trajectory_capacity", False)\n'
        '        if type(hard_stop) is not bool:\n'
        '            raise TypeError("hard_stop_on_trajectory_capacity must be a bool")\n'
        '        if hard_stop and any(cfg.dispatch_mode != "ray_task" for cfg in runner_registry.values()):\n'
        '            raise ValueError("hard_stop_on_trajectory_capacity requires dispatch_mode=ray_task")\n',
        "framework.validate_hardcap",
    )
    src = replace_exact(
        src,
        '                metadata={"_trace_identity": trace_identity},\n',
        '                metadata={\n'
        '                    "_trace_identity": trace_identity,\n'
        '                    "hard_stop_on_trajectory_capacity": self._hard_stop_on_trajectory_capacity,\n'
        '                },\n',
        "framework.create_session_metadata",
    )
    src = replace_exact(
        src,
        '    async def _run_agent_episode(\n',
        CAP_HELPERS + '    async def _run_agent_episode(\n',
        "framework.runner_hardcap_helpers",
    )
    src = replace_exact(
        src,
        '                    try:\n'
        '                        task_result = await asyncio.wait_for(\n'
        '                            object_ref,\n'
        '                            timeout=runner_config.session_timeout_seconds,\n'
        '                        )\n'
        '                    except (asyncio.TimeoutError, asyncio.CancelledError):\n',
        '                    try:\n'
        '                        if self._hard_stop_on_trajectory_capacity:\n'
        '                            task_result, hit_capacity = await self._await_runner_with_hard_cap(\n'
        '                                object_ref, session_id, runner_config.session_timeout_seconds\n'
        '                            )\n'
        '                        else:\n'
        '                            task_result = await asyncio.wait_for(\n'
        '                                object_ref,\n'
        '                                timeout=runner_config.session_timeout_seconds,\n'
        '                            )\n'
        '                            hit_capacity = False\n'
        '                    except (asyncio.TimeoutError, asyncio.CancelledError):\n',
        "framework.ray_runner_wait",
    )
    # Upgrade already-patched Docker sources from the older strict
    # "exactly one chain" guard. The user may have run the old patcher last
    # night; a patcher upgrade must be idempotent, not require restoring files.
    old_guard = (
        '                    if (\n'
        '                        len(session_trajectories) != 1\n'
        '                        or session_trajectories[0].extra_fields.get("materialization_reason")\n'
        '                        != "max_trajectory_length"\n'
        '                    ):\n'
    )
    new_guard = (
        '                    if (\n'
        '                        not session_trajectories\n'
        '                        or not any(\n'
        '                            t.extra_fields.get("materialization_reason") == "max_trajectory_length"\n'
        '                            for t in session_trajectories\n'
        '                        )\n'
        '                    ):\n'
    )
    if old_guard in src:
        if src.count(old_guard) != 1:
            raise PatchError("framework old hardcap guard appears multiple times")
        src = src.replace(old_guard, new_guard, 1)
    src = replace_exact(
        src,
        '                session_trajectories = await self.gateway_manager.finalize_session(session_id)\n'
        '                session_trajectories = _select_session_trajectories(\n',
        '                session_trajectories = await self.gateway_manager.finalize_session(session_id)\n'
        '                if self._hard_stop_on_trajectory_capacity and hit_capacity:\n'
        '                    if (\n'
        '                        not session_trajectories\n'
        '                        or not any(\n'
        '                            t.extra_fields.get("materialization_reason") == "max_trajectory_length"\n'
        '                            for t in session_trajectories\n'
        '                        )\n'
        '                    ):\n'
        '                        raise RuntimeError(\n'
        '                            f"Session hard cap expected one truncated trajectory; got {len(session_trajectories)}"\n'
        '                        )\n'
        '                session_trajectories = _select_session_trajectories(\n',
        "framework.avoid_multiplying_rollouts",
    )
    # Even with a 48K hard cap, a Gateway Session may have already
    # materialized an earlier chain before the stop takes effect. Preserve
    # exactly the FIRST chain, discard later materializations, and force a
    # failure reward so a late successful chain never rewards the truncated
    # episode. This policy is OPT-IN with hard_stop_on_trajectory_capacity.
    src = replace_exact(
        src,
        '                session_trajectories = _select_session_trajectories(\\n',
        '                hardcap_multi_chain_count = (\\n'
        '                    len(session_trajectories) if self._hard_stop_on_trajectory_capacity\\n'
        '                    else 0\\n'
        '                )\\n'
        '                if hardcap_multi_chain_count > 1:\\n'
        '                    logger.warning(\\n'
        '                        "session %s: %d finalized chains despite hard cap; keeping first and forcing reward 0",\\n'
        '                        session_id, hardcap_multi_chain_count,\\n'
        '                    )\\n'
        '                    session_trajectories = session_trajectories[:1]\\n'
        '                session_trajectories = _select_session_trajectories(\\n',
        "framework.select_first_chain_after_hardcap",
    )
    src = replace_exact(
        src,
        '            annotations = self._score_from_reward_info(session_trajectories)\\n',
        '            annotations = (\\n'
        '                [(0.0, {"multi_chain_session": True,\\n'
        '                        "original_session_chain_count": hardcap_multi_chain_count})]\\n'
        '                if hardcap_multi_chain_count > 1\\n'
        '                else self._score_from_reward_info(session_trajectories)\\n'
        '            )\\n',
        "framework.zero_reward_without_running_reward_worker",
    )
    src = replace_exact(
        src,
        '            self._log_trajectory_summary(session_id, result_trajectories)\\n',
        '            if hardcap_multi_chain_count > 1:\\n'
        '                if len(result_trajectories) != 1:\\n'
        '                    raise RuntimeError("multi-chain hard-cap must retain exactly one first trajectory")\\n'
        '                result_trajectories = [\\n'
        '                    replace(\\n'
        '                        result_trajectories[0],\\n'
        '                        reward_score=0.0,\\n'
        '                        reward_info={**dict(result_trajectories[0].reward_info or {}), "reward": 0.0},\\n'
        '                        extra_fields={\\n'
        '                            **dict(result_trajectories[0].extra_fields or {}),\\n'
        '                            "multi_chain_session": True,\\n'
        '                            "original_session_chain_count": hardcap_multi_chain_count,\\n'
        '                        },\\n'
        '                    )\\n'
        '                ]\\n'
        '            self._log_trajectory_summary(session_id, result_trajectories)\\n',
        "framework.first_chain_zero_reward_metadata",
    )
    return src


PATCHES = {
    "uni_agent/gateway/session/session.py": patch_session,
    "uni_agent/gateway/manager.py": patch_manager,
    "uni_agent/framework/framework.py": patch_framework,
}


def apply(root: Path, *, check: bool) -> dict[str, str]:
    before: dict[Path, str] = {}
    after: dict[Path, str] = {}
    for rel, transform in PATCHES.items():
        target = root / rel
        if not target.is_file():
            raise PatchError(f"missing Uni-Agent file: {target}")
        source = target.read_text(encoding="utf-8")
        modified = transform(source)
        ast.parse(modified, filename=str(target))
        before[target] = source
        after[target] = modified
    for target, original in before.items():
        modified = after[target]
        print(f"{target}: {'already-patched' if original == modified else 'patch-required'}")
        if original != modified:
            print("".join(difflib.unified_diff(
                original.splitlines(keepends=True),
                modified.splitlines(keepends=True),
                fromfile=str(target), tofile=str(target) + " (hard-cap)",
            )), end="")
    if check:
        print("CHECK ONLY: no files changed")
        return {str(p): "check" for p in before}
    updated: list[Path] = []
    try:
        for target, original in before.items():
            modified = after[target]
            if original == modified:
                continue
            if target.read_text(encoding="utf-8") != original:
                raise PatchError(f"target changed after preflight: {target}")
            fd, tmp = tempfile.mkstemp(prefix=".uniagent-hardcap-", dir=target.parent)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as stream:
                    stream.write(modified)
                os.chmod(tmp, target.stat().st_mode)
                os.replace(tmp, target)
                updated.append(target)
            finally:
                if os.path.exists(tmp):
                    os.unlink(tmp)
    except BaseException:
        for target in reversed(updated):
            target.write_text(before[target], encoding="utf-8")
        raise
    print("APPLIED: strict 48K session cap available (opt-in config)")
    return {str(p): "patched" for p in before}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("uniagent_root", type=Path)
    parser.add_argument("--check", action="store_true", help="Validate anchors and print diff, no writes")
    args = parser.parse_args()
    root = args.uniagent_root.expanduser().resolve()
    try:
        apply(root, check=args.check)
    except (OSError, PatchError, SyntaxError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
