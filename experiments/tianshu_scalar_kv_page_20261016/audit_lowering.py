# SPDX-License-Identifier: Apache-2.0
"""Compare immutable executable ELF sections, not debug-dependent file hashes."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import struct


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    root=parser.parse_args().root.resolve();artifacts=root/'micro/kernel_artifacts';rows={}
    for mode in ('baseline_cv','False','True'):
        folder=artifacts/mode;manifest=json.loads((folder/'manifest.json').read_text())
        for name,digest in manifest['files'].items():assert hashlib.sha256((folder/name).read_bytes()).hexdigest()==digest
        data=(folder/'cubin').read_bytes();assert data[:6]==b'\x7fELF\x02\x01'
        offset=struct.unpack_from('<Q',data,40)[0];size,count,index=struct.unpack_from('<HHH',data,58)
        assert size==64 and offset+size*count<=len(data)
        headers=[struct.unpack_from('<IIQQQQIIQQ',data,offset+i*size) for i in range(count)]
        header=headers[index];strings=data[header[4]:header[4]+header[5]];sections={}
        for header in headers:
            name=strings[header[0]:].split(b'\0',1)[0].decode()
            if name=='.text':
                payload=data[header[4]:header[4]+header[5]];assert len(payload)==header[5]
                sections[name]=dict(bytes=len(payload),sha256=hashlib.sha256(payload).hexdigest())
        assert set(sections)=={'.text'}
        calls=Counter(re.findall(r'call[^\n]*?@([\w.]+)\(',(folder/'llir').read_text()))
        rows[mode]=dict(text=sections['.text'],llvm_page_and_other_i32_load_calls=calls['llvm.bi.load.kop.v1i32'],
                        llvm_bf16_load_calls=calls['llvm.bi.load.kop.v2bf16'],register_info=manifest['register_info'])
    assert rows['baseline_cv']['text']==rows['False']['text']==rows['True']['text']
    assert [rows[m]['llvm_page_and_other_i32_load_calls'] for m in rows]==[36,36,5]
    result=dict(complete=True,executable_text_byte_identical=True,rows=rows,
                scope='Actual B64/KV4096 compilation only. LLVM call counts are static; executable identity includes full .text.')
    (root/'local_lowering_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
