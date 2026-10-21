# SPDX-License-Identifier: Apache-2.0
"""Use unchanged kernels; select query tile and correct total query-block grid."""
class Producer:
    def __init__(self,kernel,mode,config):self.kernel,self.mode,self.config=kernel,mode,config;self.last=None
    def __getitem__(self,grid):
        def run(*args,**kw):
            assert not args and kw['HEAD_SIZE']==kw['HEAD_SIZE_PADDED']==128 and kw['BLOCK_SIZE']==16
            assert kw['num_query_heads']==16 and kw['num_queries_per_kv']==8 and kw['KV_QUANT_MODE']==0
            assert not kw['USE_TD'] and not kw['USE_TD_QO'] and kw['SLIDING_WINDOW']==0
            assert kw['key_cache_ptr'].dtype.__str__()=='torch.uint8' and kw['value_cache_ptr'].dtype.__str__()=='torch.bfloat16'
            assert kw['IS_3D']==(self.mode=='candidate')
            c=self.config if self.mode=='candidate' else dict(block_m=256,tile=128,warps=16,stages=1)
            bq=c['block_m']//8;kw.update(BLOCK_M=c['block_m'],BLOCK_Q=bq,TILE_SIZE=c['tile'],num_warps=c['warps'],num_stages=c['stages'],KV_CACHE_MODIFIER='.cv')
            actual=(kw['query_ptr'].shape[0]//bq+kw['num_seqs'],2)
            if kw['IS_3D']:actual=(*actual,kw['NUM_SEGMENTS_PER_SEQ'])
            binary=self.kernel[actual](**kw)
            self.last=dict(mode=self.mode,config=c,IS_3D=kw['IS_3D'],grid=list(actual),sequences=kw['num_seqs'],
                query_tokens=kw['query_ptr'].shape[0],segments=kw['NUM_SEGMENTS_PER_SEQ'],block_q=bq,
                kernel_hash=binary.hash,n_regs=getattr(binary,'n_regs',None),n_spills=getattr(binary,'n_spills',None),
                warps=binary.metadata.num_warps,shared=getattr(binary.metadata,'shared',None))
            return binary
        return run
class Reducer:
    def __init__(self,kernel,producer):self.kernel,self.producer=kernel,producer;self.last=None
    def __getitem__(self,grid):
        def run(*args,**kw):
            assert not args and self.producer.mode=='candidate'
            kw.update(TILE_SIZE=self.producer.config['tile'],BLOCK_Q=self.producer.config['block_m']//8)
            binary=self.kernel[grid](**kw);self.last=dict(kernel_hash=binary.hash,grid=list(grid),n_regs=getattr(binary,'n_regs',None),n_spills=getattr(binary,'n_spills',None))
            return binary
        return run
def attach(reader,mode,config):
    producer=Producer(reader.kernel_unified_attention,mode,config)
    reader.kernel_unified_attention=producer;reducer=Reducer(reader.reduce_segments,producer);reader.reduce_segments=reducer
    return producer,reducer
