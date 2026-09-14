"""Focused pure-GDN CP tree execution; no FA/KV, Engine, or packed dispatch."""

from contextlib import contextmanager
from dataclasses import dataclass

import torch
from torch import nn

from ..prefix_state import GDNPrefixState
from .gdn_state import forward_gdn_cp_with_state


class GDNCPStack(nn.Module):
    """Residual stack of real GDN modules sharing a CP group (TP=SP=1).

    No MLP or additional norm is inserted. Each layer's own GDN projections
    and gated norm remain unchanged; the block computes hidden + GDN(hidden).
    """

    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        if not self.layers:
            raise ValueError("GDN stack must not be empty")
        self.layer_numbers = tuple(layer.layer_number for layer in self.layers)
        if len(set(self.layer_numbers)) != len(self.layers):
            raise ValueError("GDN layer numbers must be unique")
        self.cp_group = self.layers[0].pg_collection.cp
        self.cp_size = self.cp_group.size()
        for layer in self.layers:
            if layer.pg_collection.cp is not self.cp_group:
                raise ValueError("all GDN layers must share the same CP group")
            if layer.cp_size != self.cp_size or layer.tp_size != 1 or layer.sp_size != 1:
                raise ValueError("inconsistent GDN stack topology")

    def forward(self, hidden, initial_states=None):
        initial_states = {} if initial_states is None else initial_states
        if initial_states and set(initial_states) != set(self.layer_numbers):
            raise ValueError("initial states must cover every GDN layer")
        states = {}
        for layer in self.layers:
            (output, bias), state = forward_gdn_cp_with_state(
                layer, hidden, initial_states.get(layer.layer_number),
            )
            hidden = hidden + output if bias is None else hidden + output + bias
            states[layer.layer_number] = state
        return hidden, states


@dataclass
class _Entry:
    segment_id: int
    hidden: torch.Tensor
    state: GDNPrefixState
    own_loss: object


@dataclass
class GDNBranchResult:
    output: torch.Tensor
    loss: torch.Tensor
    input_gradient: torch.Tensor


class GDNCPBranchExecutor:
    """Graph-free Push, independent leaf anchors, and reverse Pop state VJPs.

    All ranks MUST invoke the same event order. Loss callbacks return local
    scalar contributions already normalized by the GLOBAL tree denominator.
    No CP scaling, parameter all-reduce or optimizer step is performed here.
    States stay private to this executor/group; cross-rank restore is not an API.
    """

    def __init__(self, model):
        self.model = model
        self.stack = []
        self.seen = set()
        self.failed = False
        self._versions = None

    @contextmanager
    def _operation(self):
        if self.failed:
            raise RuntimeError("GDN branch executor is failed")
        try:
            if not self.model.training:
                raise RuntimeError("GDN branch execution requires training mode")
            versions = tuple(p._version for p in self.model.parameters())
            if self._versions is not None and versions != self._versions:
                raise RuntimeError("model parameters changed while prefix states were active")
            yield
        except Exception:
            self.failed = True
            raise

    def _new_segment(self, segment_id, parent_id):
        if not isinstance(segment_id, int) or isinstance(segment_id, bool) or segment_id < 0:
            raise ValueError("segment ID must be a nonnegative integer")
        if segment_id in self.seen:
            raise ValueError("segment ID already consumed")
        expected = self.stack[-1].segment_id if self.stack else None
        if parent_id != expected:
            raise ValueError(f"parent must be stack top {expected}, got {parent_id}")
        self.seen.add(segment_id)

    def _anchors(self):
        parent = self.stack[-1].state if self.stack else None
        return parent, None if parent is None else parent.make_anchors()

    def push(self, segment_id, parent_id, hidden, own_loss=None):
        with self._operation():
            self._new_segment(segment_id, parent_id)
            if own_loss is not None and not callable(own_loss):
                raise TypeError("own_loss must be callable or None")
            initial = self.stack[-1].state.layer_states if self.stack else {}
            saved_input = hidden.detach().clone()
            with torch.no_grad():
                output, states = self.model(saved_input, initial)
            saved = GDNPrefixState.save(segment_id, hidden.shape[0] * self.model.cp_size, states)
            self.stack.append(_Entry(segment_id, saved_input, saved, own_loss))
            self._versions = tuple(p._version for p in self.model.parameters())
            return output.detach()

    def _backward(self, hidden, loss_fn, relay):
        parent, anchors = self._anchors()
        x = hidden.detach().clone().requires_grad_(True)
        output, states = self.model(x, {} if anchors is None else anchors.layer_states)
        loss = output.float().sum() * 0 if loss_fn is None else loss_fn(output)
        if not isinstance(loss, torch.Tensor) or loss.numel() != 1 or not loss.requires_grad:
            raise ValueError("loss callback must return a differentiable scalar")
        roots, gradients = [loss], [None]
        for number, gradient in relay.items():
            roots.extend((states[number].conv_state, states[number].recurrent_state))
            gradients.extend((gradient.conv_state, gradient.recurrent_state))
        torch.autograd.backward(roots, grad_tensors=gradients)
        if anchors is not None:
            parent.accumulate_anchor_gradients(anchors)
        if x.grad is None:
            raise RuntimeError("missing segment input gradient")
        return GDNBranchResult(output.detach(), loss.detach(), x.grad.detach())

    def visit(self, segment_id, parent_id, hidden, own_loss):
        with self._operation():
            self._new_segment(segment_id, parent_id)
            return self._backward(hidden, own_loss, {})

    def pop(self, segment_id):
        with self._operation():
            if not self.stack or self.stack[-1].segment_id != segment_id:
                raise ValueError("Pop must consume the stack top")
            entry = self.stack.pop()
            result = self._backward(entry.hidden, entry.own_loss, entry.state.consume_gradients())
            entry.state.release()
            if not self.stack:
                self._versions = None
            return result

    def assert_empty(self):
        if self.failed or self.stack:
            raise RuntimeError("GDN branch executor did not finish cleanly")
