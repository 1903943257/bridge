"""CPU-only regressions for locally modified Uni-Agent-CC VERL installation."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
INSTALLER = ROOT / "patches" / "apply_uniagent_cc_tpr.py"

CFG = """from typing import Any, Optional
class EngineConfig: pass
class McoreEngineConfig(EngineConfig):
    use_megatron_fsdp: bool = False
    strategy: str = "megatron"
"""
ENGINE = """class MegatronEngine:
    def _build_megatron_module(self):
        wrap_config = 1
        override_ddp_config = self._resolve_override_ddp_config()
        module, updated_tf_config = make_megatron_module(
            wrap_config=wrap_config,
            tf_config=self.tf_config,
        )
        return module
    def forward_backward_batch(self, data, loss_function, forward_only=False):
        tu.assign_non_tensor(data, batch_num_tokens=1)
        tu.assign_non_tensor(data, dp_size=self.get_data_parallel_size())
        # BSHD path only: preserved local engine comment
        micro_batches, indices = prepare_micro_batches(data=data)
        return micro_batches
"""
WORKER = """class TrainingWorker:
    def train_mini_batch(self, data):
        for batch_idx, mini_batch_td in enumerate(data):
            with record_function(f"mini_batch{batch_idx}"):
                # local profiling marker (not present upstream)
                actor_output = self.train_batch(mini_batch_td)
"""


def _checkout(tmp_path, worker=WORKER):
    files = {
        "verl/workers/config/engine.py": CFG,
        "verl/workers/engine/megatron/transformer_impl.py": ENGINE,
        "verl/workers/engine_workers.py": worker,
        "verl/models/mcore/tpr/megatron_adapter.py": "# copied from Bridge\n",
    }
    for name, body in files.items():
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(body)
    return {name: tmp_path / name for name in files}


def _run(root, *args):
    return subprocess.run(
        [sys.executable, str(INSTALLER), *args, str(root)],
        capture_output=True, text=True,
    )


def test_check_apply_repeat_preserves_local_worker_changes(tmp_path):
    files = _checkout(tmp_path)
    before = {name: path.read_text() for name, path in files.items()}
    checked = _run(tmp_path, "--check")
    assert checked.returncode == 0, checked.stderr
    assert {name: path.read_text() for name, path in files.items()} == before

    result = _run(tmp_path)
    assert result.returncode == 0, result.stderr
    assert "APPLIED" in result.stdout
    assert "tpr_enabled: bool = False" in files["verl/workers/config/engine.py"].read_text()
    assert "install_tpr_module_spec(self.provider, bridge=self.bridge)" in files["verl/workers/engine/megatron/transformer_impl.py"].read_text()
    assert "run_tpr_forward_backward_batch" in files["verl/workers/engine/megatron/transformer_impl.py"].read_text()
    worker = files["verl/workers/engine_workers.py"].read_text()
    assert "local profiling marker" in worker
    assert "capture_real_actor_update_minibatch" in worker

    patched = {name: path.read_text() for name, path in files.items()}
    repeated = _run(tmp_path)
    assert repeated.returncode == 0, repeated.stderr
    assert {name: path.read_text() for name, path in files.items()} == patched


def test_missing_worker_call_fails_without_mutating_other_files(tmp_path):
    files = _checkout(tmp_path, WORKER.replace(
        "actor_output = self.train_batch(mini_batch_td)",
        "actor_output = self.other_api(mini_batch_td)",
    ))
    before = {name: path.read_text() for name, path in files.items()}
    failed = _run(tmp_path)
    assert failed.returncode != 0
    assert "Expected exactly one" in failed.stderr
    assert {name: path.read_text() for name, path in files.items()} == before


def test_partial_tpr_config_fails_closed(tmp_path):
    files = _checkout(tmp_path)
    config = files["verl/workers/config/engine.py"]
    config.write_text(CFG.replace("    strategy:", "    tpr_enabled: bool = False\n    strategy:"))
    before = {name: path.read_text() for name, path in files.items()}
    failed = _run(tmp_path)
    assert failed.returncode != 0
    assert "Partial TPR config" in failed.stderr
    assert {name: path.read_text() for name, path in files.items()} == before

def test_upgrade_existing_uniagent_cc_patch_without_rewriting_other_hooks(tmp_path):
    files = _checkout(tmp_path)
    engine = files["verl/workers/engine/megatron/transformer_impl.py"]
    original = engine.read_text()
    with_legacy = original.replace(
        "        module, updated_tf_config = make_megatron_module(",
        "        if self.engine_config.tpr_enabled:\\n"
        "            from verl.models.mcore.tpr.megatron_adapter import install_tpr_module_spec\\n"
        "            install_tpr_module_spec(self.provider)\\n"
        "        module, updated_tf_config = make_megatron_module(",
    ).replace("\\n", "\n")
    engine.write_text(with_legacy)
    first = _run(tmp_path)
    assert first.returncode == 0, first.stderr
    patched = engine.read_text()
    assert "install_tpr_module_spec(self.provider, bridge=self.bridge)" in patched
    assert "install_tpr_module_spec(self.provider)\n" not in patched
    assert patched.count("run_tpr_forward_backward_batch") == 2
    second = _run(tmp_path)
    assert second.returncode == 0, second.stderr
    assert engine.read_text() == patched
