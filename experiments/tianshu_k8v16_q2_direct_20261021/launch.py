# SPDX-License-Identifier: Apache-2.0
"""Component-only bounded Q1/Q2 launch; not installed into the engine."""
class Producer:
    def __init__(self, kernel, tile, warps):
        self.kernel, self.tile, self.warps = kernel, tile, warps
        self.records = {}

    def __getitem__(self, grid):
        def call(*args, **kw):
            assert not args and kw['IS_3D'] and kw['BLOCK_M'] == 16 and kw['BLOCK_Q'] == 2
            assert kw['BLOCK_SIZE'] == 16 and kw['HEAD_SIZE'] == 128
            assert kw['num_query_heads'] == 16 and kw['num_queries_per_kv'] == 8
            assert kw['KV_QUANT_MODE'] == 0 and not kw['USE_TD'] and not kw['USE_TD_QO']
            assert 1 <= kw['num_seqs'] <= 64
            tokens = kw['query_ptr'].shape[0]
            assert kw['num_seqs'] <= tokens <= kw['num_seqs'] * 2
            for name in ('segm_output_ptr', 'segm_max_ptr', 'segm_expsum_ptr'):
                assert kw[name].shape[0] >= tokens, (name, kw[name].shape, tokens)
            kw.update(TILE_SIZE=self.tile, num_warps=self.warps, num_stages=1, KV_CACHE_MODIFIER='.cv')
            compiled = self.kernel[grid](**kw)
            key = str((tokens, kw['num_seqs'], kw['NUM_SEGMENTS_PER_SEQ']))
            self.records[key] = dict(tokens=tokens, sequences=kw['num_seqs'], grid=list(grid),
                tile=self.tile, warps=compiled.metadata.num_warps, stages=compiled.metadata.num_stages,
                n_regs=getattr(compiled, 'n_regs', None), n_spills=getattr(compiled, 'n_spills', None),
                shared=getattr(compiled.metadata, 'shared', None),
                compiler_hash=compiled.hash, binary_sha256=__import__('hashlib').sha256(compiled.kernel).hexdigest())
            return compiled
        return call


class Reducer:
    def __init__(self, kernel, producer):
        self.kernel, self.producer = kernel, producer

    def __getitem__(self, grid):
        def call(*args, **kw):
            kw['TILE_SIZE'] = self.producer.tile
            return self.kernel[grid](*args, **kw)
        return call


def attach(reader, tile, warps):
    producer = Producer(reader.kernel_unified_attention, tile, warps)
    reader.kernel_unified_attention = producer
    reader.reduce_segments = Reducer(reader.reduce_segments, producer)
    return producer
