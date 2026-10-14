# SPDX-License-Identifier: Apache-2.0
"""Prove the renderer changes only the new constexpr and two load keywords."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
from experimental_cache import render_source


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--l0-source',type=Path,required=True)
    args=parser.parse_args();raw=args.l0_source.read_bytes();text=render_source(raw)
    original,modified=ast.parse(raw),ast.parse(text)
    function=next(n for n in modified.body if isinstance(n,ast.FunctionDef) and n.name=='kernel_unified_attention')
    assert function.args.args[-1].arg=='KV_CACHE_MODIFIER' and function.args.defaults[-1].value==''
    function.args.args.pop();function.args.defaults.pop()
    changed=[]
    for node in ast.walk(function):
        if not isinstance(node,ast.Call):continue
        keywords=[k for k in node.keywords if k.arg=='cache_modifier']
        if not keywords:continue
        assert isinstance(node.func,ast.Attribute) and isinstance(node.func.value,ast.Name) and node.func.value.id=='tl' and node.func.attr=='load'
        assert len(keywords)==1
        value=keywords[0].value
        assert isinstance(value,ast.IfExp) and value.test.id=='IS_3D' and value.body.id=='KV_CACHE_MODIFIER' and value.orelse.value==''
        assert isinstance(node.args[0],ast.BinOp) and node.args[0].left.id in ('key_cache_ptr','value_cache_ptr')
        changed.append(node.args[0].left.id);node.keywords.remove(keywords[0])
    assert sorted(changed)==['key_cache_ptr','value_cache_ptr']
    assert ast.dump(original)==ast.dump(modified)
    base=Path(__file__).resolve().parent
    proof=dict(complete=True,cpu_only=True,original_L0_sha256=hashlib.sha256(raw).hexdigest(),
               generated_sha256=hashlib.sha256(text.encode()).hexdigest(),only_two_KV_load_keywords_and_one_constexpr_changed=True,
               other_AST_identical=True,cache_policy_supported_on_GPU_not_tested=True,
               source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (base/'experimental_cache.py',Path(__file__))})
    out=base/'results/20261014/prepared_32g';out.mkdir(parents=True,exist_ok=True)
    (out/'CPU_SOURCE_AUDIT_32g.json').write_text(json.dumps(proof,indent=2)+'\n')
    print(json.dumps(proof))


if __name__=='__main__':main()
