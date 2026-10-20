# SPDX-License-Identifier: Apache-2.0
"""Native page_size_padded counts K, V and both FP32 scale planes exactly."""
MODES=('Soft8OwnWriter','K8V16','K8VI8Heads')

def geometry(mode):
    assert mode in MODES
    return dict(page_bytes=8448 if mode=='K8VI8Heads' else 12288 if mode=='K8V16' else 8192,
        key_bytes=1,value_bytes=2 if mode=='K8V16' else 1,value_offset=4096,
        key_scale_offset=8192 if mode=='K8VI8Heads' else None,value_scale_offset=8320 if mode=='K8VI8Heads' else None)

def allocate(torch,FullAttentionSpec,reshape,mode,physical_pages,strided,device='cuda'):
    g=geometry(mode)
    spec=FullAttentionSpec(block_size=16,num_kv_heads=2,head_size=128,dtype=torch.uint8,
        page_size_padded=g['page_bytes'] if mode!='Soft8OwnWriter' else None)
    assert spec.page_size_bytes==g['page_bytes']
    raw=torch.full((physical_pages*g['page_bytes'],),17,device=device,dtype=torch.uint8)
    native=reshape(raw,spec,(physical_pages,2,16,2,128),tuple(range(5)),physical_pages,None)
    if strided:native=native[::2]
    count=native.shape[0];step=native.stride(0)*native.element_size()
    def view(dtype,offset,shape,strides):
        size=torch.empty(0,dtype=dtype).element_size()
        result=torch.empty(0,device=device,dtype=dtype).set_(raw.untyped_storage(),offset//size,shape,(step//size,*strides))
        assert result.untyped_storage().data_ptr()==raw.untyped_storage().data_ptr() and result.data_ptr()==raw.data_ptr()+offset
        last=offset+sum((s-1)*v for s,v in zip(result.shape,result.stride()))*size+size
        assert last<=raw.numel()
        return result
    k=view(torch.uint8,0,(count,16,2,128),(256,128,1))
    v=view(torch.int8 if mode=='K8VI8Heads' else torch.bfloat16 if mode=='K8V16' else torch.uint8,4096,(count,16,2,128),(256,128,1))
    ks=vs=None
    if mode=='K8VI8Heads':
        ks=view(torch.float32,8192,(count,16,2),(2,1));vs=view(torch.float32,8320,(count,16,2),(2,1))
    proof=dict(mode=mode,**g,physical_pages=physical_pages,num_blocks=count,actual_spec_page_bytes=spec.page_size_bytes,
        actual_raw_bytes=raw.numel(),actual_native_view_stride_bytes=step,all_planes_alias_actual_raw=True,
        key_dtype=str(k.dtype),value_dtype=str(v.dtype),key_strides=list(k.stride()),value_strides=list(v.stride()),
        key_scale_strides=None if ks is None else list(ks.stride()),value_scale_strides=None if vs is None else list(vs.stride()),
        full_engine_allocator_not_installed=True,all_scale_bytes_in_native_page_accounting=True)
    return raw,native,k,v,ks,vs,proof
