# SPDX-License-Identifier: Apache-2.0
"""Archive a terminal success or failure without altering its result status."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--label',required=True);args=parser.parse_args()
    root=args.source.resolve();assert root.is_relative_to('/workspace/benchmark-results')
    result=json.loads((root/'result.json').read_text());assert result['phase'] in ('complete','failed')
    pidfile=Path('/proc')/str(result['runner_pid'])/'stat'
    assert not pidfile.exists() or pidfile.read_text().split()[2]=='Z'
    assert args.label.replace('_','').isalnum()
    repo=Path('/workspace/vllm-plugin-FL')
    for path,sha in result['source_snapshots'].items():
        assert hashlib.sha256((root/'source_snapshot'/Path(path).relative_to('/')).read_bytes()).hexdigest()==sha
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==sha
    subprocess.run(['git','bundle','create',str(root/'source_repository.bundle'),'HEAD'],cwd=repo,check=True,capture_output=True)
    (root/'archive_provenance.json').write_text(json.dumps(dict(phase=result['phase'],
        complete=result['complete'],created_utc=datetime.now(timezone.utc).isoformat(),
        repository_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
        failed_status_not_rewritten=True,all_raw_files_included=True),indent=2)+'\n')
    archive=Path('/tmp')/('tianshu_'+args.label+'.tgz');assert not archive.exists()
    with tarfile.open(archive,'w:gz') as tar:tar.add(root,arcname=root.name)
    data=archive.read_bytes();parts=[]
    for index,start in enumerate(range(0,len(data),262144)):
        p=archive.with_name(archive.name+f'.part{index:03d}');chunk=data[start:start+262144];p.write_bytes(chunk)
        parts.append(dict(name=p.name,bytes=len(chunk),sha256=hashlib.sha256(chunk).hexdigest()))
    manifest=dict(archive=str(archive),bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),parts=parts)
    p=Path('/tmp')/('tianshu_'+args.label+'_manifest.json');p.write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(manifest=str(p),phase=result['phase'],bytes=len(data),sha256=manifest['sha256'])))


if __name__=='__main__':main()
