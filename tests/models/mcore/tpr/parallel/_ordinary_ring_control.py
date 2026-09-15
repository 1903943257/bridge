"""Explicit whole-only transport selection; never changes segmented roots."""


def install_ordinary_ring(patch):
    from verl.models.mcore.tpr.parallel import ring_attention as ring

    def whole(query, current_key, current_value, *, prefix_blocks=(), **kwargs):
        if prefix_blocks:
            raise AssertionError("ordinary Ring control cannot execute a Prefix path")
        return ring.ordinary_ring_cp_attention(query, current_key, current_value, **kwargs)

    patch.setattr(ring, "ring_cp_attention", whole)
