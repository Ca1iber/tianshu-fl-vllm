# SPDX-License-Identifier: Apache-2.0
"""Permute 3D launch axes; invert their program IDs inside the same reader."""
import ast,copy,hashlib,importlib.util,sys
from pathlib import Path
PARENT_SHA='7b94b63e0b33a2775b9c0cd043b51120fbbb6b35a81f5134897e00fbea4cf106'
ORDERS={'current':(0,1,2),'heads_first':(1,0,2),'segments_first':(2,0,1),'heads_segments_first':(1,2,0)}
ORIGINAL=('    q_block_global_idx = tl.program_id(0)\n'
          '    kv_head_idx = tl.program_id(1)\n'
          '    segm_idx = tl.program_id(2) if IS_3D else 0')
NAMES=('q_block_global_idx','kv_head_idx','segm_idx')


def remap_grid(grid,mode,is_3d=True):
    assert mode in ORDERS and len(grid)==(3 if is_3d else 2)
    assert all(isinstance(n,int) and n>0 for n in grid)
    return tuple(grid[i] for i in ORDERS[mode]) if is_3d else tuple(grid)

def render(parent,mode):
    raw=Path(parent).read_bytes();assert hashlib.sha256(raw).hexdigest()==PARENT_SHA
    assert mode in ORDERS
    text=raw.decode();assert text.count(ORIGINAL)==1
    order=ORDERS[mode];inverse=tuple(order.index(i) for i in range(3))
    if mode=='current':replacement=ORIGINAL;generated=text
    else:
        replacement=(f'    q_block_global_idx = tl.program_id({inverse[0]}) if IS_3D else tl.program_id(0)\n'
          f'    kv_head_idx = tl.program_id({inverse[1]}) if IS_3D else tl.program_id(1)\n'
          f'    segm_idx = tl.program_id({inverse[2]}) if IS_3D else 0')
        generated=text.replace(ORIGINAL,replacement)
    assert generated.replace(replacement,ORIGINAL)==text
    old,new=ast.parse(text),ast.parse(generated)
    before=next(n for n in old.body if isinstance(n,ast.FunctionDef) and n.name=='kernel_unified_attention')
    after=next(n for n in new.body if isinstance(n,ast.FunctionDef) and n.name=='kernel_unified_attention')
    assert ast.dump(before.args)==ast.dump(after.args)
    assignments={n.targets[0].id:n for n in before.body if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id in NAMES}
    hits=[]
    for index,node in enumerate(after.body):
        if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id in NAMES:
            hits.append(node.targets[0].id);after.body[index]=copy.deepcopy(assignments[node.targets[0].id])
    assert set(hits)==set(NAMES) and len(hits)==3
    assert ast.dump(old,include_attributes=False)==ast.dump(new,include_attributes=False)
    return generated,dict(mode=mode,parent_sha256=PARENT_SHA,grid_axis_order=list(order),inverse_program_id_axes=list(inverse),
        generated_sha256=hashlib.sha256(generated.encode()).hexdigest(),exact_source_reversal=True,
        AST_only_three_program_ID_assignments_changed=True,ABI_and_floating_point_body_unchanged=True,
        prefill_2D_IDs_preserved=True,CTA_count_preserved=True)

def load(parent,directory,mode):
    text,proof=render(parent,mode);directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
    path=directory/f'generated_{mode}.py';path.write_text(text)
    name='_K8V16_CTA_'+mode+'_'+hashlib.sha256(str(path).encode()).hexdigest()[:12]
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module);proof['generated_source']=str(path)
    return module,proof
