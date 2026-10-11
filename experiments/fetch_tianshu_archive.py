#!/usr/bin/env python3
"""Fetch immutable Tianshu archive parts with three connections and SHA checks."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import tarfile
import uuid


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--parts-dir',type=Path,required=True)
    parser.add_argument('--extract-to',type=Path,required=True)
    args=parser.parse_args()
    manifest=json.loads(args.manifest.read_text())
    parts=args.parts_dir.resolve();parts.mkdir(parents=True,exist_ok=True)
    destination=args.extract_to.resolve();destination.mkdir(parents=True,exist_ok=True)
    archive_name=Path(manifest['archive']).name
    assert re.fullmatch(r'tianshu_[a-zA-Z0-9_]+\.tgz',archive_name)
    assert len({p['name'] for p in manifest['parts']})==len(manifest['parts'])

    def valid(path,info):
        return path.exists() and path.stat().st_size==info['bytes'] and digest(path)==info['sha256']

    def fetch(info):
        assert re.fullmatch(re.escape(archive_name)+r'\.part\d{3}',info['name'])
        target=parts/info['name']
        if valid(target,info):return info['name']
        last_error=''
        for attempt in range(3):
            temporary=parts/(info['name']+'.attempt_'+uuid.uuid4().hex)
            command=['scp','-o','BatchMode=yes','-o','ConnectTimeout=15',
                     'tianshu:/tmp/'+info['name'],str(temporary)]
            process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=True)
            try:
                _,stderr=process.communicate(timeout=35)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid,signal.SIGKILL);process.wait()
                last_error='transfer timeout'
            else:
                last_error=stderr.decode(errors='replace')[-400:]
                if process.returncode==0 and valid(temporary,info):
                    temporary.replace(target)
                    print(json.dumps({'part':info['name'],'verified':True,'attempt':attempt+1}),flush=True)
                    return info['name']
                if process.returncode==0:last_error='SHA or byte count mismatch'
        raise RuntimeError(info['name']+': '+last_error)

    with ThreadPoolExecutor(max_workers=3) as pool:
        futures=[pool.submit(fetch,info) for info in manifest['parts']]
        for future in as_completed(futures):future.result()
    data=b''.join((parts/p['name']).read_bytes() for p in manifest['parts'])
    assert len(data)==manifest['bytes'] and hashlib.sha256(data).hexdigest()==manifest['sha256']
    archive=parts/archive_name;archive.write_bytes(data)
    with tarfile.open(archive) as tar:
        assert all(not p.name.startswith('/') and '..' not in Path(p.name).parts for p in tar.getmembers())
        tar.extractall(destination,filter='data')
    (destination/'backup_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'complete':True,'bytes':len(data),'sha256':manifest['sha256'],
                      'extracted_to':str(destination)}),flush=True)


if __name__=='__main__':
    main()
