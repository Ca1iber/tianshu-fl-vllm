# SPDX-License-Identifier: Apache-2.0
"""Keep Decode producer and segment reducer on the same KV tile size."""


class _ReductionRouter:
    def __init__(self, original):
        self.original = original
        self.tile = None

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            if self.tile is not None:
                kwargs["TILE_SIZE"] = self.tile
            return self.original[grid](*args, **kwargs)
        return launch


class LaunchRouter:
    def __init__(self, original, configuration):
        import vllm.v1.attention.ops.triton_unified_attention as native
        self.original = original
        self.configuration = dict(configuration)
        self.metadata = {}
        if not isinstance(native.reduce_segments, _ReductionRouter):
            native.reduce_segments = _ReductionRouter(native.reduce_segments)
        self.reduction = native.reduce_segments

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            assert not args and kwargs["IS_3D"] is True
            assert kwargs["BLOCK_M"] == 16 and kwargs["BLOCK_Q"] == 2
            assert kwargs["num_query_heads"] == 16 and kwargs["num_queries_per_kv"] == 8
            assert kwargs["HEAD_SIZE"] == 128 and kwargs["KV_QUANT_MODE"] == 0
            kwargs.update(self.configuration)
            self.reduction.tile = kwargs["TILE_SIZE"]
            compiled = self.original[grid](**kwargs)
            self.metadata = {k: getattr(compiled.metadata, k, None)
                             for k in ("num_warps", "num_stages", "shared")}
            self.metadata["producer_and_reducer_tile"] = self.reduction.tile
            self.metadata["launch_grid"] = tuple(grid)
            return compiled
        return launch
