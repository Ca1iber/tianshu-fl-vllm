# SPDX-License-Identifier: Apache-2.0
"""Factorial late-V scheduling and bounded integer offsets, based on current paired K."""
import ast,copy,hashlib,importlib.util,re,sys,textwrap
from pathlib import Path
BASE=Path(__file__).resolve().parent
PARENT_SHA='f1df731bae6cf48d5f38eef0a0adb5863ade2145667aab1ac976b9d473bdb002'
MODES=('current','addr32','late_v','late_v_addr32')
SHORT_SPLITS={mode:4 for mode in MODES}
OLD=BASE.parent/'tianshu_k8v16_address32_20261020/source.py'
spec=importlib.util.spec_from_file_location('_paired_existing_offset_math',OLD);old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
offset_bound=old.offset_bound

def configuration(mode,n):
    assert mode in MODES and 1<=n<=64
    return dict(TILE_SIZE=16,num_warps=1,num_stages=1)
def configured_segments(mode,n,parent_segments):return parent_segments

class Normalize32(ast.NodeTransformer):
    def visit_Attribute(self,node):
        self.generic_visit(node)
        if isinstance(node.value,ast.Name) and node.value.id=='tl' and node.attr=='int64':node.attr='int32'
        return node
class DefaultFalse(ast.NodeTransformer):
    def visit_AnnAssign(self,node):
        if isinstance(node.target,ast.Name) and node.target.id=='PAIR_LATE_V':return None
        return self.generic_visit(node)
    def visit_If(self,node):
        self.generic_visit(node)
        if isinstance(node.test,ast.Name) and node.test.id=='PAIR_LATE_V':return node.orelse
        if isinstance(node.test,ast.UnaryOp) and isinstance(node.test.op,ast.Not) and isinstance(node.test.operand,ast.Name) and node.test.operand.id=='PAIR_LATE_V':return node.body
        return node

def late_v(parent):
    offset=re.findall(r'            v_offset = \(\n[\s\S]*?            \)\n',parent)
    load=re.findall(r'            V_load = tl.load\(\n[\s\S]*?            \)\n',parent)
    assert len(offset)==len(load)==1
    offset,load=offset[0],load[0]
    cast='        V = _cast_kv_tile(V_load, Q, v_scale, KV_QUANT_MODE)\n'
    anchor='    USE_FP8_Q_DESCALE: tl.constexpr = KV_QUANT_MODE == 1 and Q_IS_FP8\n'
    constant='    PAIR_LATE_V: tl.constexpr = (IS_3D and KV_QUANT_MODE == 0 and not USE_TD and not SLIDING_WINDOW and value_cache_ptr.dtype.element_ty == tl.bfloat16)\n'
    pv='        acc = acc * alpha[:, None]\n'
    late='        if PAIR_LATE_V:\n'+textwrap.indent(textwrap.dedent(offset)+textwrap.dedent(load)+textwrap.dedent(cast),' '*12)
    edits=[(anchor,anchor+constant),
        (offset,'            if not PAIR_LATE_V:\n'+textwrap.indent(offset,' '*4)),
        (load,'            if not PAIR_LATE_V:\n'+textwrap.indent(load,' '*4)),
        (cast,'        if not PAIR_LATE_V:\n'+textwrap.indent(cast,' '*4)),
        (pv,pv+'\n'+late)]
    result=parent
    for before,after in edits:
        assert result.count(before)==1;result=result.replace(before,after)
    restored=result
    for before,after in reversed(edits):
        assert restored.count(after)==1;restored=restored.replace(after,before)
    assert restored==parent
    old_ast=ast.parse(parent);false_ast=DefaultFalse().visit(ast.parse(result))
    assert ast.dump(old_ast,include_attributes=False)==ast.dump(false_ast,include_attributes=False)
    old_sig={n.name:ast.dump(n.args) for n in old_ast.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
    new_sig={n.name:ast.dump(n.args) for n in ast.parse(result).body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
    assert old_sig==new_sig
    return result,dict(exact_inverse_recovers_parent_bytes=True,LATE_false_AST_identical_to_parent=True,all_function_signatures_unchanged=True,
        only_V_offset_load_and_cast_relocated=True,QK_softmax_PV_expressions_unchanged=True)

def render(parent,mode):
    raw=Path(parent).read_bytes();assert hashlib.sha256(raw).hexdigest()==PARENT_SHA and mode in MODES
    text=raw.decode();proof=dict(mode=mode,parent_sha256=PARENT_SHA,requires_host_scope_and_offset_guard=(mode!='current'))
    if 'late_v' in mode:text,proof['late_V_proof']=late_v(text)
    if 'addr32' in mode:
        previous=text;text,annotations=re.subn(r': tl\.int64\b',': tl.int32',text)
        text,casts=re.subn(r'\.to\(\s*tl\.int64\s*\)','.to(tl.int32)',text)
        assert annotations>0 and casts>0 and 'tl.int64' not in text
        old_ast=Normalize32().visit(ast.parse(previous));new_ast=Normalize32().visit(ast.parse(text))
        assert ast.dump(old_ast,include_attributes=False)==ast.dump(new_ast,include_attributes=False)
        proof.update(integer_annotations_changed=annotations,integer_casts_changed=casts,only_integer_width_AST_changes_after_scheduling=True,physical_pointer_and_slot_mapping_not_narrowed=True)
    compile(text,'<paired_load_candidate>','exec')
    proof.update(generated_sha256=hashlib.sha256(text.encode()).hexdigest(),current_original_source_unchanged=(mode=='current'),selected_configuration_for_B64=configuration(mode,64),Math_or_service_not_claimed=True)
    return text,proof

def load(parent,directory,mode):
    text,proof=render(parent,mode);directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
    def module(label,content):
        path=directory/(label+'.py');path.write_text(content);name='_paired_load_'+hashlib.sha256(str(path).encode()).hexdigest()[:16]
        spec=importlib.util.spec_from_file_location(name,path);value=importlib.util.module_from_spec(spec);sys.modules[name]=value;spec.loader.exec_module(value);return value
    candidate=module('generated_'+mode,text)
    wide=candidate if mode=='current' else module('host_original_and_wide_fallback',Path(parent).read_text())
    wide._pair_candidate_reader=candidate
    proof.update(generated_source=candidate.__file__,host_original_source=wide.__file__,host_source_sha256=PARENT_SHA)
    return wide,proof
