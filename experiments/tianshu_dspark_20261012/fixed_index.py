# SPDX-License-Identifier: Apache-2.0
"""Fixed, untuned row indexing for valid1D integer indices in this Spec lane."""
import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=['source_rows','rows','row_stride','column_stride','index_stride'])
def gather(source,indices,output,source_rows,rows,row_stride,column_stride,index_stride,
           WIDTH:tl.constexpr,BLOCK:tl.constexpr):
    row=tl.program_id(0)
    column=tl.program_id(1)*BLOCK+tl.arange(0,BLOCK)
    index=tl.load(indices+row*index_stride)
    index=tl.where(index<0,index+source_rows,index)
    valid=(column<WIDTH)&(index>=0)&(index<source_rows)
    value=tl.load(source+index*row_stride+column*column_stride,mask=valid,other=0)
    tl.store(output+row*WIDTH+column,value,mask=column<WIDTH)


def fixed_into(source,indices,output):
    assert source.ndim in (1,2) and indices.ndim==1
    width=source.shape[1] if source.ndim==2 else 1
    assert output.numel()==indices.numel()*width and output.is_contiguous()
    if indices.numel():
        gather[(indices.numel(),triton.cdiv(width,256))](source,indices,output,
            source.shape[0],indices.numel(),source.stride(0),source.stride(1) if source.ndim==2 else 1,
            indices.stride(0),WIDTH=width,BLOCK=256,num_warps=4,num_stages=1)


def install(module,record=None):
    original=module._index_func
    adjacent_original=module._linearized_adjacent_index_func
    def supported(source,indices,output):
        return (source.ndim in (1,2) and len(indices)==1 and indices[0].ndim==1
            and source.dtype in (torch.bfloat16,torch.float32,torch.int32,torch.int64)
            and indices[0].dtype in (torch.int32,torch.int64) and output.is_contiguous())
    def selected(source,indices,output):
        if supported(source,indices,output):
            if record is not None:record(source,indices[0],output)
            return fixed_into(source,indices[0],output)
        return original(source,indices,output)
    def adjacent_selected(source,start_dim,indices,output):
        if start_dim==0 and supported(source,indices,output):
            if record is not None:record(source,indices[0],output)
            return fixed_into(source,indices[0],output)
        return adjacent_original(source,start_dim,indices,output)
    module._index_func=selected
    module._linearized_adjacent_index_func=adjacent_selected


def run_dispatch_validation(output_directory):
    """Exercise the actual public API, both internal routes and graph replay."""
    import importlib,json
    from pathlib import Path
    root=Path(output_directory);assert not root.exists();root.mkdir(parents=True)
    module=importlib.import_module('flag_gems.ops.index')
    calls=[];install(module,lambda s,i,o:calls.append((tuple(s.shape),i.numel())))
    torch.manual_seed(2026101240);cases=[]
    for shape in [(168,),(193,2048),(129,130560)]:
        for dtype in (torch.bfloat16,torch.float32,torch.int32,torch.int64):
            for strided in (False,True):
                fullshape=tuple(x*2 for x in shape) if strided else shape
                storage=torch.randint(-200,200,fullshape,device='cuda').to(dtype)
                source=storage[::2,::2] if strided and len(shape)==2 else storage[::2] if strided else storage
                for index_dtype in (torch.int32,torch.int64):
                    indices=torch.randint(0,shape[0],(147,),device='cuda',dtype=index_dtype)
                    indices[0]=-1;indices[1]=-shape[0]
                    before=len(calls);actual=module.index(source,[indices])
                    expected=source.cpu()[indices.cpu().long()]
                    assert len(calls)==before+1 and torch.equal(actual.cpu(),expected)
                    cases.append({'shape':shape,'dtype':str(dtype),'index_dtype':str(index_dtype),
                                  'strided':strided,'actual_api_fixed_route':True,'bitwise_cpu_equal':True})
    source=torch.arange(193*257,device='cuda',dtype=torch.float32).reshape(193,257)
    indices=torch.arange(147,device='cuda',dtype=torch.int64)
    fixed_into(source,indices,torch.empty((147,257),device='cuda'))
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):actual=module.index(source,[indices])
    replays=[]
    for shift in (1,17,46):
        source.add_(shift);indices.copy_((torch.arange(147,device='cuda')+shift)%193)
        graph.replay();assert torch.equal(actual.cpu(),source.cpu()[indices.cpu()])
        replays.append({'shift':shift,'bitwise_cpu_equal':True})
    result={'complete':True,'passed':True,'cases':cases,'graph_replays':replays,
            'official_benchmark':False,'contract':'Valid axis0 single1D integer indices only'}
    (root/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


def run_validation(output_directory):
    import json
    from pathlib import Path
    root=Path(output_directory);assert not root.exists();root.mkdir(parents=True)
    torch.manual_seed(2026101216);rows=[]
    for shape,count in [((37,),17),((193,),129),((565,2048),65),((1537,2048),147),
                        ((129,130560),98),((512,130560),448),((32,256),21)]:
        for dtype in (torch.bfloat16,torch.float32,torch.int32,torch.int64):
            for strided in (False,True):
                fullshape=tuple(x*2 for x in shape) if strided else shape
                storage=torch.randint(-200,200,fullshape,device='cuda').to(dtype)
                source=storage[::2,::2] if strided and len(shape)==2 else storage[::2] if strided else storage
                indices=torch.randint(0,shape[0],(count,),device='cuda',dtype=torch.int64)
                indices[0]=-1;indices[1]=-shape[0]
                cpu=source.cpu();expected=cpu[indices.cpu()]
                actual=torch.empty_like(expected,device='cuda')
                fixed_into(source,indices,actual)
                passed=torch.equal(actual.cpu(),expected)
                entry={'shape':shape,'rows':count,'dtype':str(dtype),'strided':strided,
                       'bitwise_cpu_equal':passed,'negative_valid_indices_checked':True}
                rows.append(entry);assert passed,entry
    result={'complete':True,'passed':True,'cases':rows,'official_benchmark':False,
            'contract':'Valid indices in[-source_rows,source_rows); other indexing shapes fall back'}
    (root/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':
    import argparse,subprocess
    p=argparse.ArgumentParser();p.add_argument('--output',required=True)
    p.add_argument('--dispatch-validation',action='store_true');args=p.parse_args()
    assert 'No running processes found' in subprocess.check_output(['ixsmi'],text=True)
    (run_dispatch_validation if args.dispatch_validation else run_validation)(args.output)
