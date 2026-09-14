"""Forward-only L5-L8 closure: align once, then propagate whole blocks."""

from contextlib import contextmanager

import torch
import torch.distributed as dist


def shard_conv_state(value, *, q, v, rank):
    """CP shard is Q_rank + K_rank + V_rank, not half the concatenation."""
    assert rank in (0, 1) and q % 2 == v % 2 == 0
    return torch.cat([t.chunk(2, dim=1)[rank] for t in value.split((q, q, v), dim=1)], dim=1).contiguous()


class ClosureCapture:
    def __init__(self):
        self.data = None

    @contextmanager
    def capture(self, model, sid):
        if sid != 2:
            yield
            return
        from verl.models.mcore.tpr.context import get_tpr_attention_context

        def before(module, args, kwargs):
            assert self.data is None
            ctx = get_tpr_attention_context()
            assert ctx.attention_backend is None and ctx.prefix_length == ctx.suffix_length == 128
            hidden = args[0] if args else kwargs["hidden_states"]
            data = {"hidden": hidden}
            for number in (5, 6, 7):
                state = ctx.get_initial_gdn_state(number)
                data[f"L{number}.conv"] = state.conv_state
                data[f"L{number}.recurrent"] = state.recurrent_state
            data["pk"], data["pv"] = ctx.get_past_kv(8)
            rope = ctx.suffix_rotary_pos_emb
            self.rope_tuple = isinstance(rope, tuple)
            for i, value in enumerate(rope if self.rope_tuple else (rope,)):
                data[f"rope{i}"] = value
            self.data = {n: t.detach().cpu().clone() for n, t in data.items()}

        handle = model.decoder.layers[4].register_forward_pre_hook(before, with_kwargs=True)
        try:
            yield
        finally:
            handle.remove()


def propagation_closure(model, capture, reference, runtime, monkeypatch, metrics, *, cp):
    from verl.models.mcore.tpr.context import TPRAttentionContext, use_tpr_attention_context
    from verl.models.mcore.tpr.prefix_state import GDNLayerState
    from verl.models.mcore.tpr import attention
    from verl.models.mcore.tpr.parallel import ring_attention as ring
    from ._first_gdn_zigzag_control import zigzag_control_forward

    rank = runtime.cp_group.rank()
    shard = ring.make_ring_sequence_shard(128, cp_rank=rank, cp_size=2)
    indices = shard.global_indices(device="cpu")
    layers = tuple(model.decoder.layers[4:8])
    assert tuple(l.layer_number for l in layers) == (5, 6, 7, 8)
    assert all(l.config.hidden_dropout == l.config.attention_dropout == 0 for l in layers)
    if cp == 1:
        assert capture.data is not None, "missing S2/L5 closure capture"
        data = {}
        for name, tensor in capture.data.items():
            value = tensor.to(runtime.device)
            dist.broadcast(value, src=0, group=runtime.cp_group)
            data[name] = value.cpu()
        rope_tuple = capture.rope_tuple
        weights = {f"{l.layer_number}.{n}": p.detach().cpu().clone()
                   for l in layers for n, p in l.named_parameters()}
    else:
        data, rope_tuple, weights = reference["data"], reference["rope_tuple"], reference["weights"]
        for l in layers:
            for n, p in l.named_parameters():
                assert torch.equal(weights[f"{l.layer_number}.{n}"], p.detach().cpu()), (l.layer_number, n)

    def sliced(name, value):
        if name.endswith(".conv"):
            number = int(name.split('.')[0][1:])
            layer = model.decoder.layers[number-1].self_attention
            q, v = layer.qk_dim_local_tp, layer.v_dim_local_tp
            return shard_conv_state(value, q=q, v=v, rank=rank)
        if name.endswith(".recurrent"):
            return value.chunk(2, dim=1)[rank].contiguous()
        return value.index_select(0, indices).contiguous()

    def run():
        values = {n: (sliced(n, t) if cp == 2 else t).to(runtime.device).clone() for n, t in data.items()}
        states = {i: GDNLayerState(values[f"L{i}.conv"], values[f"L{i}.recurrent"]) for i in (5, 6, 7)}
        rope = tuple(values[f"rope{i}"] for i in range(2 if rope_tuple else 1))
        backend = None if cp == 1 else ring.RingCPAttentionBackend(
            global_prefix_length=128, current_shard=shard, cp_group=runtime.cp_group,
            prefix_blocks_by_layer={8: (ring.RingLocalKVBlock(0, shard, values["pk"], values["pv"]),)})
        ctx = TPRAttentionContext(prefix_length=128, suffix_length=128,
            past_key_values={8: (values["pk"], values["pv"])} if cp == 1 else {},
            initial_gdn_states=states, suffix_rotary_pos_emb=rope if rope_tuple else rope[0],
            attention_backend=backend)
        records, counts = {}, []
        def save(name, tensor):
            assert name not in records
            records[name] = tensor.detach().cpu().clone()
        rect, ring_fn = attention.rectangular_causal_attention, ring.ring_cp_attention
        def core(q, k, v, **kwargs):
            save("L8.q", q)
            save("L8.k", k if cp == 2 else k[128:])
            save("L8.v", v if cp == 2 else v[128:])
            save("L8.pk", values["pk"])
            save("L8.pv", values["pv"])
            output = (rect if cp == 1 else ring_fn)(q, k, v, **kwargs)
            save("L8.core", output)
            return output
        with monkeypatch.context() as patch, torch.no_grad(), use_tpr_attention_context(ctx):
            patch.setattr(attention if cp == 1 else ring,
                          "rectangular_causal_attention" if cp == 1 else "ring_cp_attention", core)
            if cp == 1:
                for l in layers:
                    attn = l.self_attention
                    modules = (attn.in_proj, attn.out_proj) if l.layer_number != 8 else (attn.linear_qkv, attn.linear_proj)
                    for module in (*modules, l.mlp.linear_fc1, l.mlp.linear_fc2):
                        counter = {"full128_to_zigzag_2x64": 0}
                        counts.append(counter)
                        patch.setattr(module, "forward", zigzag_control_forward(module.forward, counter))
            hidden = values["hidden"]
            for l in layers:
                save(f"L{l.layer_number}.input", hidden)
                output = l(hidden_states=hidden, attention_mask=None)
                hidden = output[0] if isinstance(output, tuple) else output
                assert isinstance(hidden, torch.Tensor)
                save(f"L{l.layer_number}.output", hidden)
                if l.layer_number != 8:
                    state = ctx.new_gdn_states[l.layer_number]
                    save(f"L{l.layer_number}.conv", state.conv_state)
                    save(f"L{l.layer_number}.recurrent", state.recurrent_state)
        assert cp == 2 or (len(counts) == 16 and all(c["full128_to_zigzag_2x64"] == 1 for c in counts))
        return records

    result, repeat = run(), run()
    repeat_exact = all(torch.equal(result[n], repeat[n]) for n in result)
    print(f"STAGE-4.4 CLOSURE r={rank} CP={cp} repeat_all_exact={repeat_exact}; "
          "full blocks L5-L8; align L5 hidden + per-layer L5/6/7 initial states + L8 prefix KV/RoPE; "
          "CP1 16 projections zigzag64; no intermediate hidden resets; forward only", flush=True)
    if cp == 2:
        assert result.keys() == reference["result"].keys()
        first = None
        for name, right in result.items():
            left = sliced(name, reference["result"][name])
            assert left.shape == right.shape and torch.isfinite(left).all() and torch.isfinite(right).all()
            exact = torch.equal(left, right)
            if not exact and first is None:
                first = name
            pair = metrics({name: left.float()}, {name: right.float()}).aggregate
            print(f"STAGE-4.4 CLOSURE r={rank} {name}: exact={exact} rel={pair.relative_l2:.6e} "
                  f"max={(right.float()-left.float()).abs().max().item():.6e}", flush=True)
        print(f"STAGE-4.4 CLOSURE r={rank} first_nonexact={first}; diagnostic, not full-model PASS", flush=True)
    return dict(data=data, rope_tuple=rope_tuple, weights=weights, result=result)
