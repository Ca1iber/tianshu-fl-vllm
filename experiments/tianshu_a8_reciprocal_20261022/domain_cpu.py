# SPDX-License-Identifier: Apache-2.0
"""Deterministic finite-BF16 peak/value domain; CPU only."""
import numpy as np
EPS=np.float32(1.e-8)
def f32(words):return (np.asarray(words,dtype=np.uint16).astype(np.uint32)<<16).view(np.float32)
def peaks():
 words=np.arange(0x7f80,dtype=np.uint16);return words[f32(words)>=EPS]
def values_for_peaks(words,k=4096):
 words=np.asarray(words,np.uint16);e=(words>>7).astype(np.int32);d=np.arange(10,dtype=np.int32);mant=np.arange(128,dtype=np.int32)
 candidate=(((e[:,None,None]-d[None,:,None])<<7)|mant[None,None,:]).reshape(len(words),-1).astype(np.uint16)
 candidate=np.where(candidate<=words[:,None],candidate,0).astype(np.uint16);out=np.zeros((len(words),k),np.uint16);out[:,:1280]=candidate;out[:,1280:2560]=candidate|np.uint16(0x8000);return out
def clamp_words():
 words=np.arange(65536,dtype=np.uint16);values=f32(words);return words[np.isfinite(values)&(np.abs(values)<EPS)]
def old(a):
 peak=np.max(np.abs(a),axis=1);scale=np.maximum(peak,EPS)/np.float32(127);value=np.clip(np.rint(a/scale[:,None]),-127,127).astype(np.int8);return value,scale
