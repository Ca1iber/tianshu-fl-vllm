# SPDX-License-Identifier: Apache-2.0
"""Bounded integer-only variants of the current qualified K8V16 S16 reader."""
import ast,copy,hashlib,importlib.util,re,sys
from pathlib import Path
PARENT_SHA='7b94b63e0b33a2775b9c0cd043b51120fbbb6b35a81f5134897e00fbea4cf106'
CACHE_STRIDES=tuple(f'stride_{plane}_cache_{axis}' for plane in ('k','v') for axis in range(3))

def offset_bound(shape,strides):
    assert len(shape)==len(strides) and shape and all(n>0 for n in shape) and all(s>=0 for s in strides)
    # Includes conservative speculative/masked tile address formation, in tensor element units.
    return sum((n-1)*s for n,s in zip(shape,strides))+2048*sum(strides)

def safe_tensor(tensor):
    try:return 0<=offset_bound(tuple(tensor.shape),tuple(tensor.stride()))<2**31
    except (AssertionError,OverflowError,TypeError):return False

class Normalize64(ast.NodeTransformer):
    def visit_Attribute(self,node):
        self.generic_visit(node)
        if isinstance(node.value,ast.Name) and node.value.id=='tl' and node.attr=='int64':node.attr='int32'
        return node

def generate(parent,mode):
    raw=Path(parent).read_bytes();assert hashlib.sha256(raw).hexdigest()==PARENT_SHA
    text=raw.decode();assert mode in ('kv32','all32')
    if mode=='all32':
        result,annotations=re.subn(r': tl\.int64\b',': tl.int32',text)
        result,casts=re.subn(r'\.to\(\s*tl\.int64\s*\)','.to(tl.int32)',result)
    else:
        result=text;annotations=0
        for name in CACHE_STRIDES:
            result,count=re.subn(r'\b'+name+r': tl\.int64\b',name+': tl.int32',result);assert count==1;annotations+=count
        pattern=r'(physical_block_(?:idx|scalar) = tl\.load\([\s\S]*?\)\.to\()tl\.int64(\))'
        result,casts=re.subn(pattern,r'\1tl.int32\2',result);assert casts==2
    assert annotations==(6 if mode=='kv32' else 28) and casts==(2 if mode=='kv32' else 6),(mode,annotations,casts)
    old,new=ast.parse(text),ast.parse(result)
    assert ast.dump(Normalize64().visit(copy.deepcopy(old)),include_attributes=False)==ast.dump(Normalize64().visit(copy.deepcopy(new)),include_attributes=False)
    if mode=='all32':assert 'tl.int64' not in result
    return result,dict(parent_sha256=PARENT_SHA,mode=mode,annotations_changed=annotations,integer_casts_changed=casts,
                       generated_sha256=hashlib.sha256(result.encode()).hexdigest(),only_integer_dtype_AST_changes=True,
                       floating_point_body_and_physical_pointer_types_unchanged=True,requires_runtime_signed_element_offset_guard=True)

def load(parent,root,mode):
    text,proof=generate(parent,mode);root=Path(root);root.mkdir(parents=True,exist_ok=False)
    path=root/('generated_'+mode+'.py');path.write_text(text)
    name='_K8V16_bounded_'+mode+'_'+proof['generated_sha256'][:12]
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    proof['generated_source']=str(path)
    return module,proof
