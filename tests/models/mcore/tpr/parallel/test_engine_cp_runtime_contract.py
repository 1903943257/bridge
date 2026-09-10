# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from types import SimpleNamespace

import pytest

import verl.models.mcore.tpr.parallel.engine_runtime as runtime_module


class _Backend:
    def __init__(self, name, size, rank):
        self.backend_name = name
        self.parallel_size = size
        self.parallel_rank = rank


def _engine(*, cp_size, backend=None, transformer_algo=None):
    return SimpleNamespace(
        engine_config=SimpleNamespace(
            context_parallel_size=cp_size,
            dynamic_context_parallel=False,
            tpr_cp_backend=backend,
            override_transformer_config={},
            mcore_kwargs={},
        ),
        tf_config=SimpleNamespace(context_parallel_algo=transformer_algo),
    )


def test_cp1_runtime_does_not_require_distributed_initialization():
    runtime = runtime_module.resolve_engine_cp_runtime(
        _engine(cp_size=1),
        SimpleNamespace(),
    )

    assert not runtime.enabled
    assert runtime.group is None
    assert runtime.backend is None
    assert runtime.backend_name == "none"


@pytest.mark.parametrize(
    ("algorithm", "expected_backend"),
    (
        ("kvallgather_cp_algo", "allgather"),
        ("ulysses_cp_algo", "ulysses"),
        ("megatron_cp_algo", "ring"),
        ("hybrid_cp_algo", "hybrid"),
    ),
)
def test_cp_runtime_selects_backend_from_transformer_config(
    monkeypatch,
    algorithm,
    expected_backend,
):
    cp_group = object()
    captured = {}
    monkeypatch.setattr(runtime_module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(runtime_module.parallel_state, "model_parallel_is_initialized", lambda: True)
    monkeypatch.setattr(runtime_module.parallel_state, "get_context_parallel_group", lambda: cp_group)
    monkeypatch.setattr(runtime_module, "resolve_cp_group", lambda group: (2, 1))

    def resolve_backend(name, **kwargs):
        captured.update(name=name, **kwargs)
        return _Backend(expected_backend, kwargs["parallel_size"], kwargs["parallel_rank"])

    monkeypatch.setattr(runtime_module, "resolve_tpr_cp_backend", resolve_backend)
    runtime = runtime_module.resolve_engine_cp_runtime(
        _engine(cp_size=2, transformer_algo=algorithm),
        SimpleNamespace(),
    )

    assert runtime.enabled
    assert runtime.backend_name == expected_backend
    assert (runtime.size, runtime.rank) == (2, 1)
    assert captured == {
        "name": algorithm,
        "cp_group": cp_group,
        "parallel_size": 2,
        "parallel_rank": 1,
    }


def test_explicit_tpr_backend_overrides_mindspeed_algorithm(monkeypatch):
    cp_group = object()
    monkeypatch.setattr(runtime_module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(runtime_module.parallel_state, "model_parallel_is_initialized", lambda: True)
    monkeypatch.setattr(runtime_module.parallel_state, "get_context_parallel_group", lambda: cp_group)
    monkeypatch.setattr(runtime_module, "resolve_cp_group", lambda group: (2, 0))
    monkeypatch.setattr(
        runtime_module,
        "resolve_tpr_cp_backend",
        lambda name, **kwargs: _Backend(name, kwargs["parallel_size"], kwargs["parallel_rank"]),
    )

    runtime = runtime_module.resolve_engine_cp_runtime(
        _engine(cp_size=2, backend="allgather", transformer_algo="megatron_cp_algo"),
        SimpleNamespace(),
    )

    assert runtime.backend_name == "allgather"


def test_cp_runtime_rejects_configured_and_initialized_size_mismatch(monkeypatch):
    monkeypatch.setattr(runtime_module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(runtime_module.parallel_state, "model_parallel_is_initialized", lambda: True)
    monkeypatch.setattr(runtime_module.parallel_state, "get_context_parallel_group", lambda: object())
    monkeypatch.setattr(runtime_module, "resolve_cp_group", lambda group: (4, 0))

    with pytest.raises(RuntimeError, match="topology mismatch"):
        runtime_module.resolve_engine_cp_runtime(
            _engine(cp_size=2, backend="allgather"),
            SimpleNamespace(),
        )
