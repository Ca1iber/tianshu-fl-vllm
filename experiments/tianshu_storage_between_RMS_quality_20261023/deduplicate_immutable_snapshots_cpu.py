# SPDX-License-Identifier: Apache-2.0
"""Share equal immutable snapshot copies; never link to or edit live source paths."""
import gzip,hashlib,json,os,stat,subprocess,time
from pathlib import Path
ROOT=Path('/workspace/benchmark-results');OUT=ROOT/'20261023_RMS_quality_snapshot_dedup_CPU_tianshu'
assert not OUT.exists();OUT.mkdir();assert 'No running processes found' in subprocess.check_output(['ixsmi'],text=True)
names={'source_snapshot','controller_source_snapshot','recovery_controller_source_snapshot'};directories=[]
for root,dirs,_ in os.walk(ROOT):
 for d in dirs[:]:
  if d in names:directories.append(Path(root)/d);dirs.remove(d)
assert directories;before=os.statvfs(ROOT);canonical={};inode_hash={};files=changed=existing=0;manifest=OUT/'FILES_EXACT_SHA256.jsonl.gz';started=time.time()
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
with gzip.open(manifest,'wt',compresslevel=1) as log:
 for directory in directories:
  for root,dirs,entries in os.walk(directory,followlinks=False):
   for name in entries:
    p=Path(root)/name;s=p.lstat()
    if not stat.S_ISREG(s.st_mode):continue
    assert p.is_relative_to(directory) and not p.is_symlink();inode=(s.st_dev,s.st_ino);h=inode_hash.get(inode)
    if h is None:h=sha(p);inode_hash[inode]=h
    key=(s.st_dev,s.st_size,stat.S_IMODE(s.st_mode),s.st_uid,s.st_gid,h);previous=canonical.get(key);linked=False
    if previous is None:canonical[key]=p
    elif previous.stat().st_ino==s.st_ino:existing+=1
    else:
     temporary=p.with_name(p.name+f'.snapshot_link_{os.getpid()}');assert not temporary.exists();os.link(previous,temporary);assert sha(temporary)==h;os.replace(temporary,p);changed+=1;linked=True
    log.write(json.dumps(dict(path=str(p),sha256=h,size=s.st_size,changed=linked))+'\n');files+=1
# Re-read the full manifest and recompute each distinct final inode's SHA.
verified={};checked=0
with gzip.open(manifest,'rt') as log:
 for line in log:
  row=json.loads(line);p=Path(row['path']);s=p.stat();inode=(s.st_dev,s.st_ino)
  if inode not in verified:verified[inode]=sha(p)
  assert verified[inode]==row['sha256'] and s.st_size==row['size'];checked+=1
assert checked==files and 'No running processes found' in subprocess.check_output(['ixsmi'],text=True);after=os.statvfs(ROOT)
gate=dict(complete=True,CPU_only=True,GPU_idle_before_after=True,snapshot_directories=len(directories),files_checked=files,relinked_identical_copies=changed,already_same_inode=existing,unique_final_inodes=len(verified),all_original_paths_and_bytes_preserved=True,no_live_source_or_checkpoint_used_as_link_target=True,original_and_final_SHA_for_all_files_verified=True,available_bytes_before=before.f_bavail*before.f_frsize,available_bytes_after=after.f_bavail*after.f_frsize,net_available_bytes_gain=after.f_bavail*after.f_frsize-before.f_bavail*before.f_frsize,seconds=time.time()-started,manifest=str(manifest),manifest_sha256=sha(manifest),source_sha256=sha(Path(__file__)))
(OUT/'GATE_32g.json').write_text(json.dumps(gate,indent=2)+'\n');print(json.dumps(gate),flush=True)
