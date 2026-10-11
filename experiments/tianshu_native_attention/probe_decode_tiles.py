# SPDX-License-Identifier: Apache-2.0
"""Joint segment/tile sweep of unchanged native 3D attention kernels."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import signal
import statistics
import time

import torch
from probe_decode_segments import native, reference, check, save


def case(batch, length, strided, seed):
    torch.manual_seed(seed)
    lengths = [length-(i*37%257) if strided else length for i in range(batch)]
    width = (length+15)//16
    pages = batch*width
    table = torch.randperm(pages,device='cuda').int().reshape(batch,width)
    q = torch.randn(batch,20 if strided else 16,128,device='cuda',dtype=torch.bfloat16)[:,:16]
    k = torch.randn(pages*(2 if strided else 1),16,2,128,device='cuda',dtype=q.dtype)
    v = torch.randn_like(k)
    if strided:
        k,v = k[::2],v[::2]
    cu = torch.arange(batch+1,device='cuda',dtype=torch.int32)
    used = torch.tensor(lengths,device='cuda',dtype=torch.int32)
    expected = reference(q,k,v,table,lengths)
    original_tile = native._get_tile_size
    graphs, outputs, keep, results = {}, {}, [], {}
    configs = [(16,16)] + list(itertools.product((4,8,16,32,64),(32,64,128)))
    try:
        for segments,tile in configs:
            output = torch.empty_like(q)
            partial = torch.empty((batch,16,segments,128),device='cuda',dtype=torch.float32)
            maximum = torch.empty((batch,16,segments),device='cuda',dtype=torch.float32)
            expsum = torch.empty_like(maximum)
            native._get_tile_size = lambda hs,sw,es,is_prefill: original_tile(hs,sw,es,is_prefill) if is_prefill else tile
            def invoke():
                native.unified_attention(q=q,k=k,v=v,out=output,cu_seqlens_q=cu,max_seqlen_q=1,
                    seqused_k=used,max_seqlen_k=length,softmax_scale=128**-.5,
                    causal=True,window_size=(-1,-1),block_table=table,softcap=0.,
                    q_descale=None,k_descale=None,v_descale=None,seq_threshold_3D=64,
                    num_par_softmax_segments=segments,softmax_segm_output=partial,
                    softmax_segm_max=maximum,softmax_segm_expsum=expsum)
            for _ in range(3): invoke()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph): invoke()
            graph.replay(); torch.cuda.synchronize()
            key = (segments,tile)
            graphs[key],outputs[key] = graph,output
            keep.extend([partial,maximum,expsum])
            results[key] = {'segments':segments,'tile':tile,'vs_fp32':check(output,expected),'samples_ms':[]}
        order=list(graphs)
        for trial in range(7):
            for key in order[trial%len(order):]+order[:trial%len(order)]:
                start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(30): graphs[key].replay()
                end.record();end.synchronize()
                results[key]['samples_ms'].append(start.elapsed_time(end)/30)
        baseline=statistics.median(results[(16,16)]['samples_ms'])
        for key,result in results.items():
            result['median_ms']=statistics.median(result['samples_ms'])
            result['speedup_over_native']=baseline/result['median_ms']
            result['vs_native']=check(outputs[key],outputs[(16,16)])
    finally:
        native._get_tile_size=original_tile
    return {'batch':batch,'kv_lengths':lengths,'strided':strided,'variants':list(results.values())}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    signal.alarm(1800)
    torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False
    data={'official_benchmark':False,'native_source':native.__file__,
          'native_sha256':hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest(),
          'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'cases':[]}
    with torch.inference_mode():
        for i,(batch,length,strided) in enumerate(((1,4096,False),(8,4096,False),
                (16,4096,False),(32,4096,False),(64,4096,False),(64,5120,True))):
            result=case(batch,length,strided,2026101130+i)
            data['cases'].append(result);save(args.output/'result.json',data)
            best=min((v for v in result['variants'] if v['vs_fp32']['passed']),key=lambda v:v['median_ms'])
            print(json.dumps({'batch':batch,'length':length,'best':best,
                'all_passed':all(v['vs_fp32']['passed'] for v in result['variants'])}),flush=True)
            torch.cuda.empty_cache()
    data['finished_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime());save(args.output/'result.json',data)


if __name__=='__main__': main()
