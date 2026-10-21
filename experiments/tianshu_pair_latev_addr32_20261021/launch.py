# SPDX-License-Identifier: Apache-2.0
"""Wide host path and common producer/reducer selector, same qualified launch."""
import importlib.util,sys
from pathlib import Path
from source import MODES,SHORT_SPLITS,configuration
from pair_selector import Selector,Reducer
BASE=Path(__file__).resolve().parent
NATIVE_PATH=BASE.parent/'tianshu_k8v16_address32_20261020/component_launch.py'
sys.path.insert(0,str(NATIVE_PATH.parent))
spec=importlib.util.spec_from_file_location('_paired_lateV_qualified_launch',NATIVE_PATH)
native=importlib.util.module_from_spec(spec);sys.modules[spec.name]=native;spec.loader.exec_module(native)
def choose_config(mode,n,length):
    answer=native.choose_config(n,length);assert answer is not None
    return dict(answer,**configuration(mode,n))
def attach(reader,mode):
    assert mode in MODES
    import hashlib
    candidate=reader._pair_candidate_reader
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    selector=Selector(reader.kernel_unified_attention,candidate.kernel_unified_attention,mode,sha(reader.__file__),sha(candidate.__file__))
    producer=native.Producer(selector)
    reducer=Reducer(reader.reduce_segments,candidate.reduce_segments,producer,selector)
    reader.kernel_unified_attention=producer;reader.reduce_segments=reducer
    def invoke(kwargs):
        kw=dict(kwargs);assert kw['max_seqlen_q']==1;n=kw['seqused_k'].numel();assert 1<=n<=64
        producer.pure_decode=True;producer.decode_config=choose_config(mode,n,kw['max_seqlen_k'])
        kw['num_par_softmax_segments']=producer.decode_config['segments']
        result=reader.unified_attention(**kw)
        producer.last_record['split_policy']=dict(selector.last)
        producer.last_record['address_and_schedule_selection']=dict(selector.last_selection)
        producer.last_record['reducer_selection']=dict(reducer.last)
        return result
    return invoke,producer
