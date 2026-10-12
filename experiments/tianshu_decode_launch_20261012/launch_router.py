# SPDX-License-Identifier: Apache-2.0
"""Process-local 3D Decode launch tuning; native arithmetic and grid unchanged."""


class LaunchRouter:
    def __init__(self, original, configuration):
        self.original=original
        self.configuration=dict(configuration)
        self.metadata={}

    def __getitem__(self, grid):
        def invoke(*args, **kwargs):
            assert not args and kwargs['IS_3D'] is True
            assert kwargs['BLOCK_M']==16 and kwargs['BLOCK_Q']==2
            assert kwargs['num_query_heads']==16 and kwargs['num_queries_per_kv']==8
            assert kwargs['HEAD_SIZE']==128 and kwargs['KV_QUANT_MODE']==0
            kwargs.update(self.configuration)
            kernel=self.original[grid](**kwargs)
            if hasattr(kernel,'metadata'):
                self.metadata={k:getattr(kernel.metadata,k,None)
                               for k in ('num_warps','num_stages','shared')}
            return kernel
        return invoke
