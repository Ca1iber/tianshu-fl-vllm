# SPDX-License-Identifier: Apache-2.0
"""Change only the native segment reducer's launch warp count."""


class ReducerLaunchRouter:
    def __init__(self, original, on_compile=None):
        self.original = original
        self.num_warps = None
        self.metadata = {}
        self.on_compile = on_compile

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            assert not args
            requested = self.num_warps
            assert requested in (None, 1, 2, 4, 8)
            assert "num_warps" not in kwargs
            if requested is not None:
                kwargs["num_warps"] = requested
            compiled = self.original[grid](**kwargs)
            self.metadata = dict(
                requested_num_warps=requested, actual_num_warps=compiled.metadata.num_warps,
                actual_num_stages=compiled.metadata.num_stages,
                shared=compiled.metadata.shared, n_regs=getattr(compiled, "n_regs", None),
                n_spills=getattr(compiled, "n_spills", None), grid=list(grid),
                TILE_SIZE=kwargs["TILE_SIZE"], NUM_SEGMENTS_PER_SEQ=kwargs["NUM_SEGMENTS_PER_SEQ"],
                HEAD_SIZE=kwargs["HEAD_SIZE"], num_query_heads=kwargs["num_query_heads"],
                native_reducer_source_unchanged=True)
            if requested is not None:
                assert self.metadata["actual_num_warps"] == requested
            if self.on_compile is not None:
                self.on_compile(compiled, dict(self.metadata))
            return compiled
        return launch
