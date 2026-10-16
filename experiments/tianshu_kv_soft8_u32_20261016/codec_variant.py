# SPDX-License-Identifier: Apache-2.0
"""Keep the frozen software8 format; try native32-bit intermediate operations."""
import hashlib
import importlib.util
from pathlib import Path
import sys
LEGACY_PATH=Path(__file__).resolve().parents[1]/'tianshu_kv_soft8_20261016/codec.py'
assert hashlib.sha256(LEGACY_PATH.read_bytes()).hexdigest()=='0ccb887961dd0187abed4d77b9dfd2b3fa083e9e7e79c1650d0444e50838c14b'
spec=importlib.util.spec_from_file_location('_tianshu_legacy_soft8_codec',LEGACY_PATH)
legacy=importlib.util.module_from_spec(spec);sys.modules[spec.name]=legacy;spec.loader.exec_module(legacy)
decode_scalar=legacy.decode_scalar
check_cpu=legacy.check_cpu


def render_source(raw):
    text,record=legacy.render_source(raw)
    before='packed = data.to(tl.uint16)';after='packed = data.to(tl.uint32)'
    assert text.count(before)==1
    generated=text.replace(before,after);assert generated.replace(after,before)==text
    return generated,dict(record,base_soft8_generated_sha256=record['generated_sha256'],generated_sha256=hashlib.sha256(generated.encode()).hexdigest(),
        integer_intermediates_bits=32,same_format_and_quantizer=True,one_more_reversible_edit=True)


def load_codec(path,output):
    text,record=render_source(Path(path).read_bytes());output=Path(output);output.mkdir(parents=True,exist_ok=False)
    p=output/'generated_soft8_u32.py';p.write_text(text);name='tianshu_software_fp8_u32'
    spec=importlib.util.spec_from_file_location(name,p);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module,dict(record,generated_source=str(p))
