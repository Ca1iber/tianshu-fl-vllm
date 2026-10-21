# SPDX-License-Identifier: Apache-2.0
"""Exact ELF section signature; only nonloaded .debug_info is excluded."""
import hashlib,json,struct

def signature(data):
 assert data[:6]==b'\x7fELF\x02\x01'
 assert len(data)>=64
 header=struct.unpack_from('<16sHHIQQQIHHHHHH',data)
 offset,entsize,count,string_index=header[6],header[11],header[12],header[13]
 assert entsize==64 and count>1 and offset+count*entsize<=len(data)
 rows=[struct.unpack_from('<IIQQQQIIQQ',data,offset+i*entsize) for i in range(count)]
 strings=rows[string_index];names=data[strings[4]:strings[4]+strings[5]];sections={};ignored={}
 for row in rows:
  name=names[row[0]:].split(b'\0',1)[0].decode();assert name not in sections and name not in ignored
  if row[1]==8:content=b''
  else:
   assert row[4]+row[5]<=len(data);content=data[row[4]:row[4]+row[5]]
  record=dict(type=row[1],flags=row[2],address=row[3],size=row[5],link=row[6],info=row[7],alignment=row[8],entry_size=row[9],sha256=hashlib.sha256(content).hexdigest())
  if name=='.debug_info':
   assert row[1]==1 and row[2]==0;ignored[name]=record
  else:sections[name]=record
 assert set(ignored)=={'.debug_info'} and '.text' in sections and '.rodata' in sections and '.symtab' in sections
 core_header=dict(ident=header[0].hex(),type=header[1],machine=header[2],version=header[3],entry=header[4],program_offset=header[5],flags=header[7],header_size=header[8],program_entry_size=header[9],program_count=header[10],section_entry_size=header[11],section_count=header[12],section_name_index=header[13])
 canonical=dict(header=core_header,sections=sections)
 return dict(non_debug_signature_sha256=hashlib.sha256(json.dumps(canonical,sort_keys=True,separators=(',',':')).encode()).hexdigest(),canonical=canonical,ignored_debug_info=ignored['.debug_info'],full_binary_sha256=hashlib.sha256(data).hexdigest())
