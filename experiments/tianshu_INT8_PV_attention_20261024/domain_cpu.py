# SPDX-License-Identifier: Apache-2.0
"""Independent real scalar FP32 rounding, signed INT64 dot, and writer reference."""
import struct
import numpy as np

def f32(x):return struct.unpack('<f',struct.pack('<f',x))[0]
def expected_quant(values,positive=False):
    values=[f32(float(x)) for x in values]
    scale=f32(max(max(map(abs,values)),f32(1.e-30 if positive else 1.e-8))/127.)
    q=[min(127,max(0 if positive else -127,round(f32(x/scale)))) for x in values]
    return q,scale

def verify_primitive(arrays):
    p,v,s=arrays['P'],arrays['V'],arrays['S']
    q=arrays['Q'];scale=arrays['Scale'];integer=arrays['Integer'];output=arrays['Output']
    expected_q=[];expected_scale=[]
    for row in p:
        weighted=[f32(float(a)*float(b)) for a,b in zip(row,s)]
        qi,si=expected_quant(weighted,True)
        expected_q.append(qi);expected_scale.append(si)
    eq=np.array(expected_q,dtype=np.int8);es=np.array(expected_scale,dtype=np.float32)
    assert np.array_equal(eq,q)
    assert np.array_equal(es.view(np.uint32),scale.view(np.uint32))
    vv=v.astype(np.int64)
    expected_integer=np.array([[sum(int(a)*int(vv[t,c]) for t,a in enumerate(row))
        for c in range(v.shape[1])] for row in expected_q],dtype=np.int64)
    assert np.array_equal(integer.astype(np.int64),expected_integer)
    expected_output=(expected_integer.astype(np.float32)*es[:,None]).astype(np.float32)
    assert np.array_equal(output.view(np.uint32),expected_output.view(np.uint32))
    return dict(quantized_elements=int(q.size),integer_outputs=int(integer.size),FP32_output_words=int(output.size))

def verify_writer(arrays):
    words=arrays['value_words'];values=(words.astype(np.uint32)<<16).view(np.float32)
    q=arrays['actual_V'];s=arrays['actual_scale'];ks=arrays['actual_K_scale']
    assert q.dtype==np.int8 and s.dtype==np.float32 and np.all(ks==1.)
    for i in range(values.shape[0]):
        for head in range(2):
            eq,es=expected_quant(values[i,head])
            assert np.array_equal(q[i,head],np.array(eq,dtype=np.int8))
            assert s[i,head].view(np.uint32)==np.float32(es).view(np.uint32)
    return dict(V_integer_elements=int(q.size),V_scale_words=int(s.size),K_scale_words=int(ks.size))
