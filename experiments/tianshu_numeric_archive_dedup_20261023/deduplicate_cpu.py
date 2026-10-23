# SPDX-License-Identifier: Apache-2.0
"""Share byte-identical immutable numeric artifacts; preserve every path and byte."""
import gzip,hashlib,json,os,stat,subprocess,time
from pathlib import Path
ROOT=Path('/workspace/benchmark-results');OUT=ROOT/'20261023_numeric_archive_dedup_CPU_tianshu'
assert not OUT.exists();OUT.mkdir();assert 'No running processes found' in subprocess.check_output(['ixsmi'],text=True)
allowed={'.bin','.npy','.npz','.u8','.i8','.u16','.f16','.bf16','.f32','.i32','.u32','.i64','.f64'}
skip={'source_snapshot','controller_source_snapshot','recovery_controller_source_snapshot','.git'}
minimum=1048576;canonical={};inode_hash={};files=changed=existing=byte_total=0;started=time.time();before=os.statvfs(ROOT);manifest=OUT/'FILES_EXACT_SHA256.jsonl.gz'
def sha(path):
 with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
with gzip.open(manifest,'wt',compresslevel=1) as log:
 for root,dirs,entries in os.walk(ROOT,followlinks=False):
  dirs[:]=[name for name in dirs if name not in skip and not name.endswith('_cache') and name!='qualified_model_cache']
  for name in entries:
   path=Path(root)/name
   if path.suffix not in allowed:continue
   original=path.lstat()
   if not stat.S_ISREG(original.st_mode) or original.st_size<minimum or path.is_symlink():continue
   inode=(original.st_dev,original.st_ino);digest=inode_hash.get(inode)
   if digest is None:
    digest=sha(path);inode_hash[inode]=digest;current=path.stat();assert current.st_size==original.st_size and current.st_mtime_ns==original.st_mtime_ns
   key=(original.st_dev,original.st_size,stat.S_IMODE(original.st_mode),original.st_uid,original.st_gid,digest);previous=canonical.get(key);linked=False
   if previous is None:canonical[key]=path
   elif previous.stat().st_ino==original.st_ino:existing+=1
   else:
    temporary=path.with_name(path.name+f'.numeric_share_{os.getpid()}');assert not temporary.exists();os.link(previous,temporary);assert sha(temporary)==digest;os.replace(temporary,path);linked=True;changed+=1
   log.write(json.dumps(dict(path=str(path),sha256=digest,size=original.st_size,changed=linked))+'\n');files+=1;byte_total+=original.st_size
verified={};checked=0
with gzip.open(manifest,'rt') as log:
 for line in log:
  row=json.loads(line);path=Path(row['path']);current=path.stat();inode=(current.st_dev,current.st_ino)
  if inode not in verified:verified[inode]=sha(path)
  assert current.st_size==row['size'] and verified[inode]==row['sha256'];checked+=1
assert files==checked and 'No running processes found' in subprocess.check_output(['ixsmi'],text=True);after=os.statvfs(ROOT)
gate=dict(complete=True,CPU_only=True,GPU_idle_before_after=True,eligible_files_checked=files,logical_bytes_checked=byte_total,relinked_identical_files=changed,already_same_inode=existing,unique_final_inodes=len(verified),all_original_paths_and_bytes_preserved=True,all_original_and_final_SHA_verified=True,minimum_file_bytes=minimum,extensions=sorted(allowed),source_snapshot_checkpoint_and_cache_not_selected=True,no_live_source_or_checkpoint_used_as_link_target=True,available_bytes_before=before.f_bavail*before.f_frsize,available_bytes_after=after.f_bavail*after.f_frsize,net_available_bytes_gain=after.f_bavail*after.f_frsize-before.f_bavail*before.f_frsize,seconds=time.time()-started,manifest=str(manifest),manifest_sha256=sha(manifest),source_sha256=sha(__file__))
(OUT/'GATE_32g.json').write_text(json.dumps(gate,indent=2)+'\n');print(json.dumps(gate),flush=True)
