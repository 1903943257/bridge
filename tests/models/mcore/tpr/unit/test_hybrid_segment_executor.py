"""CPU algebraic tests for mixed FA/GDN state dispatch and multi-level relay."""

import pytest
import torch
from torch import nn

from verl.models.mcore.tpr.context import get_tpr_attention_context
from verl.models.mcore.tpr.fixed_topology_scheduler import FixedTopologyScheduler
from verl.models.mcore.tpr.prefix_state import GDNLayerState
from verl.models.mcore.tpr.segment_executor import SegmentExecutor
from verl.models.mcore.tpr.segment_plan import SegmentLossTerm, SegmentPlan, SegmentSpec


class _GDNMarker(nn.Module):
    tpr_state_kind = "gdn"
    layer_number = 1


class _Hybrid(nn.Module):
    def __init__(self):
        super().__init__()
        self.gdn = _GDNMarker()
        self.scale = nn.Parameter(torch.tensor(0.07, dtype=torch.float64))

    def rotary_pos_emb(self, max_seq_len, **kwargs):
        return torch.zeros(max_seq_len, 1, 1, 2)

    def forward(self, input_ids, position_ids, attention_mask):
        context = get_tpr_attention_context()
        initial = context.get_initial_gdn_state(1)
        conv = self.scale.new_zeros((1, 1, 1)) if initial is None else initial.conv_state
        recurrent = self.scale.new_zeros((1, 1, 1, 1)) if initial is None else initial.recurrent_state
        outputs = []
        for token in input_ids[0]:
            conv = conv * 0.3 + token * self.scale
            recurrent = recurrent * 0.4 + conv.unsqueeze(-1) * self.scale
            outputs.append(recurrent.reshape(1))
        signal = torch.cat(outputs).view(-1, 1, 1, 1)
        key, value = signal * 2, signal * 3
        context.set_new_gdn_state(1, GDNLayerState(conv, recurrent))
        context.set_new_kv(2, key, value)
        past = context.get_past_kv(2)
        past_sum = 0 if past is None else past[0].sum() + past[1].sum()
        attended = (key + value).cumsum(0).view(1, -1, 1) + past_sum
        return attended * torch.arange(8, dtype=self.scale.dtype).view(1, 1, 8)


def _plan():
    # R -> P -> (S1, S2): catches accidentally restoring the root instead of P,
    # and verifies Pop(P) relays its own loss plus sibling gradients to R.
    segments = []
    for sid, parent, start in ((0, None, 0), (1, 0, 2), (2, 1, 4), (3, 1, 4)):
        segments.append(SegmentSpec(sid, parent, torch.tensor([sid + 1, sid + 2]),
                                    start, start, (SegmentLossTerm(0, sid + 2),)))
    return SegmentPlan(segments, root_id=0)


@pytest.mark.parametrize("direct_leaf", [True, False])
def test_hybrid_multilevel_relay_matches_connected_graph(direct_leaf):
    plan = _plan()
    reference = _Hybrid()
    connected = SegmentExecutor(reference, plan, expected_layer_numbers=(2,))
    contexts, accumulated_kv = {}, {}
    loss = 0
    for sid in (0, 1, 2, 3):
        segment = plan.get(sid)
        parent = segment.parent_id
        kv = {} if parent is None else accumulated_kv[parent]
        gdn = {} if parent is None else contexts[parent].new_gdn_states
        context, logits = connected._forward(segment, past_key_values=kv,
                                              initial_gdn_states=gdn, no_grad=False)
        contexts[sid] = context
        accumulated_kv[sid] = {
            layer: tuple(torch.cat((kv[layer][i], pair[i])) if layer in kv else pair[i]
                         for i in (0, 1))
            for layer, pair in context.new_key_values.items()
        }
        loss = loss + connected._compute_loss(segment, logits)[1]
    loss.backward()

    actual = _Hybrid()
    executor = SegmentExecutor(actual, plan, expected_layer_numbers=(2,))
    if direct_leaf:
        result_loss = FixedTopologyScheduler(plan, executor).run().normalized_loss
    else:
        result_loss = 0
        saved = []
        for event in plan.dfs_events():
            if event.kind.value == "push":
                executor.push(event.segment_id)
                state = executor.gdn_states[event.segment_id]
                saved.append(state)
                assert not state.restore(1).conv_state.requires_grad
                assert not state.restore(1).recurrent_state.requires_grad
            else:
                result_loss = result_loss + executor.pop(event.segment_id).normalized_loss
        assert all(state.released for state in saved)
    torch.testing.assert_close(result_loss, loss.detach(), atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(actual.scale.grad, reference.scale.grad, atol=1e-12, rtol=1e-12)
    assert not executor.gdn_states
    executor.kv_stack.assert_empty()


def test_hybrid_rejects_cp_before_execution():
    with pytest.raises(NotImplementedError, match="CP=1"):
        SegmentExecutor(_Hybrid(), _plan(), expected_layer_numbers=(2,), cp_group=object())
