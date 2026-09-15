"""Test-only native schedule with TND kernel calls; no production patch.

All native step ordering, merge/casts and dKV communication are retained.
Only kernel-boundary layout is converted; B=1, ordinary unpadded GQA only.
"""


def install_native_tnd_layout(patch, torch_npu):
    forward = torch_npu.npu_fusion_attention
    backward = torch_npu.npu_fusion_attention_grad

    def to_tnd(x, head_dim):
        if x.ndim != 3 or x.shape[1] != 1 or x.shape[2] % head_dim:
            raise AssertionError(f"expected SBH B=1, got {tuple(x.shape)}")
        return x.reshape(x.shape[0], x.shape[2] // head_dim, head_dim).contiguous()

    def to_sbh(x):
        return x.reshape(x.shape[0], 1, -1).contiguous()

    def fwd(q, k, v, n, layout, **kwargs):
        assert layout == "SBH", layout
        head_dim = q.shape[-1] // n
        result = forward(to_tnd(q, head_dim), to_tnd(k, head_dim), to_tnd(v, head_dim),
                         n, "TND", actual_seq_qlen=[q.shape[0]],
                         actual_seq_kvlen=[k.shape[0]], **kwargs)
        # B=1 TND stats are head-major, same semantic layout as SBH.
        return (to_sbh(result[0]), result[1].reshape(1, n, q.shape[0], 8).contiguous(),
                result[2].reshape(1, n, q.shape[0], 8).contiguous(), *result[3:])

    def bwd(q, k, v, dy, n, layout, **kwargs):
        assert layout == "SBH", layout
        head_dim = q.shape[-1] // n
        kwargs = dict(kwargs)
        kwargs["attention_in"] = to_tnd(kwargs["attention_in"], head_dim)
        for key in ("softmax_max", "softmax_sum"):
            kwargs[key] = kwargs[key].reshape(1, n, q.shape[0], 8).contiguous()
        result = backward(to_tnd(q, head_dim), to_tnd(k, head_dim), to_tnd(v, head_dim),
                          to_tnd(dy, head_dim), n, "TND",
                          actual_seq_qlen=[q.shape[0]], actual_seq_kvlen=[k.shape[0]], **kwargs)
        return (*(to_sbh(x) for x in result[:3]), *result[3:])

    patch.setattr(torch_npu, "npu_fusion_attention", fwd)
    patch.setattr(torch_npu, "npu_fusion_attention_grad", bwd)
