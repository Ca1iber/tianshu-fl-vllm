# SPDX-License-Identifier: Apache-2.0
"""Reversible segment-count variants of the frozen Q2 dispatch adapter."""
import hashlib
import importlib.util
from pathlib import Path
import sys

SOURCE_SHA='f66437760ad2d9d8b9f6210fde9c321dc51cde1c2bede58ec470bfc5e8038a4a'


def render(raw,segments):
    assert segments in (4,8,16,32) and hashlib.sha256(raw).hexdigest()==SOURCE_SHA
    before=raw.decode();after=before
    edits=[('segments=8 if n>=8 else 16','segments=FIXED_SEGMENTS'),
        ('(128,16,16,128)','(128,16,32,128)'),('(128,16,16)','(128,16,32)'),
        ('scratch_allocated_segments=16','scratch_allocated_segments=32')]
    for a,b in edits:assert after.count(a)==1;after=after.replace(a,b)
    restored=after
    for a,b in reversed(edits):assert restored.count(b)==1;restored=restored.replace(b,a)
    assert restored==before
    after+='\nFIXED_SEGMENTS='+str(segments)+'\n'
    compile(after,'<segment_variant>','exec')
    return after,dict(original_source_sha256=SOURCE_SHA,generated_sha256=hashlib.sha256(after.encode()).hexdigest(),
        segments=segments,scratch_allocated_segments=32,reversible=True,
        arithmetic_kernels_and_pointer_math_unchanged=True)


def load(path,directory,segments):
    text,derivation=render(Path(path).read_bytes(),segments)
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True);output=directory/f'generated_s{segments}.py'
    if output.exists():assert output.read_text()==text
    else:output.write_text(text)
    name=f'_tianshu_q2_segment_s{segments}'
    spec=importlib.util.spec_from_file_location(name,output);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module)
    return module,dict(derivation,generated_source=str(output))
