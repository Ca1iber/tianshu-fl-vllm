# SPDX-License-Identifier: Apache-2.0
"""Original-command runtime checks plus explicit TopP and M2048 operator evidence."""
from pathlib import Path
import sys
import re
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tianshu_original_command_20261016'))
from validation_profile import digest,read,records,frozen_check,official_command,CV_SHA

def audit_runtime(root, state, official=False, online=True):
    if online:
        frozen_check(state["frozen_sources"])
    is_k8v16 = True
    budget=4096 if state["profile_environment"]["TIANSHU_DENSE_BUDGET_MODE"]=="control" else 6144
    warm_points=83 if budget==4096 else 91
    assert state["actual_budget"]==budget
    expected_seqs = state.get("seq_limit",256)
    arena_pages = 8192 if state.get("arena_mode", state["mode"]) == "candidate" else 32768
    expected_page = 12288 if is_k8v16 else 6144
    original_root = Path(state["output_root"])

    def artifact(path):
        return root / Path(path).relative_to(original_root)

    for path, sha in state["frozen_sources"].items():
        assert digest(root / "source_snapshot" / Path(path).relative_to("/")) == sha
    workers = records(root, "max_seqs_records", "worker_*.json")
    assert workers and all(r["mode"] == "control" and r["max_num_seqs"] == expected_seqs and
        r["async_scheduling"] is True and r["max_num_batched_tokens"] == budget and
        r["model_max_len"] == 131072 for r in workers)
    config = records(root, "bucket_trace_records", "configured_*.json")
    warm = records(root, "bucket_trace_records", "warmup_completed_*.json")
    assert any(r["enabled"] and len(r["points"]) == warm_points and max(r["points"]) == budget for r in config)
    assert any(r["complete"] and len(r["calls"]) == warm_points and max(c["size"] for c in r["calls"]) == budget for r in warm)
    extended = records(root, "extended_bucket_records", "extended_*.json")
    assert extended and all(r["changed_constants"] == [[2048, 8192]] and len(r["extended_points"]) == 99 for r in extended)
    producers = records(root, "decode_coherent_records", "producer_*.json")
    reducers = records(root, "decode_coherent_records", "reducer_*.json")
    if 'candidate'=='control':assert producers and reducers
    for record in producers:
        selected = record["batch"] in (8, 16, 32, 40, 48, 56, 64)
        tile = 16 if selected else 32
        warps = 1 if selected else 2 if 8 <= record["batch"] <= 64 else 4
        assert record["configuration"] == dict(TILE_SIZE=tile, num_warps=warps, num_stages=1)
        assert record["compiled_metadata"]["num_warps"] == warps
        expected_segments = 8 if record["batch"] == 64 else 64 if record["batch"] == 1 else 4 if 16 <= record["batch"] <= 64 else 16
        assert record["segments"] == expected_segments
        assert record["generated_source_sha256"] == CV_SHA
        assert record["cache_service_mode"] == "candidate" and record["actual_cache_modifier"] == ".cv"
        assert digest(artifact(record["zero_initialization"]["generated_source"])) == CV_SHA
    assert {(r["batch"], r["segments"]) for r in producers} == {(r["batch"], r["segments"]) for r in reducers}
    assert all(r["actual_tile"] == (16 if r["batch"] in (8, 16, 32, 40, 48, 56, 64) else 32) for r in reducers)
    if official and 'candidate'=='control':
        assert any(r['batch']==64 and r['segments']==8 for r in producers)
    policy = records(root, "segment_policy_records", "choose_*.json")
    assert all(r["selected_segments"] == (8 if r["batch"] == 64 else r["baseline_segments"]) for r in policy)
    gather = records(root, "bucket_trace_records", "gather_*.json")
    assert gather and any(r["index_dtype"] == "torch.int32" for r in gather)
    sql = records(root, "sql_reflection_records", "route_*.json")
    assert sql and all(r["reflection_scope"] == "one_requested_table" and not r["configuration_values_modified"] for r in sql)
    top = records(root, "topp_launch_records", "route_*.json")
    pure = [r for r in top if r["topp"] and not r["topk"]]
    assert pure
    for r in pure:
        assert r["source_sha256"] == "64c66699cd398c2eafe9eb91f0e189162f83a6f83a209651048e163d7d6fcee8"
        assert r["configuration"] == dict(num_warps=16, num_stages=1)
        assert r["compiled_metadata"]["num_warps"] == 16
        active=True
        assert r['precision_service_mode']=='candidate' and r['corrected_source_active']==active
        assert r['actual_BLOCK_SIZE']==(32768 if active else 6144)
        assert r['actual_BLOCK_SIZE_TRUNC']==(16384 if active else 4096)
        assert r['sampling_parameters_and_RNG_unchanged'] and r['grid_unchanged']
        if active:
            generated=artifact(r['generation']['generated_source'])
            assert digest(generated)==r['actual_source_sha256']=='09156e66b6aa2f4460423ea5b8d70bee043022bd75f0a784417455d0e979479d'
        else:assert r['actual_source_sha256']==r['source_sha256']
    assert not records(root, "sampler_records", "sampler_*.json")
    if official:
        assert any(r["batch"] == 64 for r in pure)
    native = records(root, "native_prefill_records", "native_prefill_*.json")
    assert native and all(r["generated_kernel_sha256"] == "4a1f8bdfdb5b9399702941a0e499e62b52f104737aba6012a9056ef36f5c4d4e" for r in native)
    capacities = re.findall(r"GPU KV cache size: ([0-9,]+) tokens", (root / "server.log").read_text())
    assert capacities and int(capacities[-1].replace(",", "")) >= 327680
    preemptions = {}
    for name in ("before_metrics.txt", "after_metrics.txt"):
        values = re.findall(r"^vllm:num_preemptions_total(?:\{[^}]*\})? ([0-9.eE+\-]+)$", (root / name).read_text(), re.MULTILINE)
        assert values and all(float(value) >= 0 for value in values)
        preemptions[name] = sum(map(float, values))
    assert "No running processes found" in (root / "hardware_after.txt").read_text()
    observed = read(root / "observed_server_environment.json")
    assert observed["environment"] == state["profile_environment"] and len(observed["environment"]) == len(state["profile_environment"])
    assert observed["command"] == state["server_command"]
    assert state["server_command"] == official_command()
    defaults=[row for path in (root/'default_budget_records').glob('default_*.json') for row in read(path)]
    assert defaults and all(row['source_sha256']=='f1a32e2207e9ed2f0d954e3e6457effca17ca907cb0cf17bba2ca8bde855a719' and row['mode']=='candidate' for row in defaults)
    assert any(row['eligible'] and row['before_budget'] is None and row['after_budget']==4096 and row['max_num_seqs']==256 for row in defaults)
    assert all(row['RNG_not_read_or_modified'] and row['original_arguments_forwarded'] for row in defaults)
    assert any(row['default_applied'] for row in defaults)
    gemm=records(root,'prefill_gemm_records','mm_*.json')
    if True:
        assert {tuple(r['shape']) for r in gemm}=={(2048,2048,2048),(2048,12288,2048),(2048,2048,6144)}
        for r in gemm:
            assert r['enabled'] and not r['output_copy'] and not r['weights_reordered']
            assert r['configuration']==dict(bm=256,bn=256,bk=32,splits=1,warps=16,stages=2,group_m=8 if r['shape'][1]==12288 else 1)
            assert r['kernel_sha256']=='cb556d7a6025e50d95a87a7858154fd10e547d9c145a5a18b65133122ab8c880'
            assert r['vendor_sha256']=='4c2e23da2a6b65b89cbb8170967f14c5327d143e21137387f3907ccb4fcde090'
    engines=records(root,'soft8_engine_records','engine_*.json')
    assert engines and all(e['mode']=='candidate' and e['enabled']==('candidate'=='candidate') for e in engines)
    if 'candidate'=='candidate':
        for e in engines:
            if is_k8v16:
                assert e['layout']=='K8V16' and e['value_plane_records']
                for plane in e['value_plane_records'].values():
                    assert plane['same_data_pointer'] and plane['same_untyped_storage'] and plane['no_data_copy']
                    assert plane['actual_dtype']=='torch.bfloat16' and plane['actual_storage_end_bytes']<=plane['storage_bytes']
            assert e['cache_dtype_config']=='auto' and e['model_dtype']=='torch.bfloat16' and e['max_model_len']==131072
            assert e['prefix_cache'] is True and e['Python_counts_not_graph_replay_counts']
            assert e['specs'] and all(x['layers']==42 and x['dtypes']==['torch.uint8'] and x['before_page_bytes']==[16384] and x['after_page_bytes']==[expected_page] for x in e['specs'])
            assert e['availability'] and all(x['arena_reserved_inside_profile'] and x['KV_memory_bytes']>0 for x in e['availability'])
            assert 'determine_available_memory' in e['reservation_stack'] and e['profile_calls']>0
            reserve=e['scratch_reservation'];assert reserve['capacity_pages']==arena_pages and reserve['single_reusable_arena']
            assert reserve['scratch_bytes']==(arena_pages+1)*16384 and reserve['table_bytes']==arena_pages*4
            assert reserve['allocated_after']-reserve['allocated_before']>=reserve['scratch_bytes']+reserve['table_bytes']
            assert e['cache_views'] and len(e['cache_views'][-1])==42
            for view in e['cache_views'][-1]:
                assert view['dtype']=='torch.uint8' and view['shape'][1:]==[2,16,2,128] and view['strides'][1:]==[4096,256,128,1]
                assert view['shape'][0]*16==int(capacities[-1].replace(',',''))
                assert view['logical_bytes']==view['shape'][0]*8192 and view['strides'][0]>=expected_page
            if is_k8v16:
                for view in e['cache_views'][-1]:
                    assert view['physical_page_bytes']==12288 and view['physical_cache_bytes']==view['shape'][0]*12288
                    assert view['actual_V_dtype']=='torch.bfloat16' and view['actual_V_same_raw_storage']
                    assert view['actual_V_strides'][0]*2==view['strides'][0]
            assert e['dense_counts']['direct_byte_decode']>0 and e['byte_counts']['encoded_writer']>0
            assert e['dense_records'] and len({v['arena_ptr'] for v in e['dense_records'].values()})==1
            assert all(v['scratch_bytes']==(arena_pages+1)*16384 and v['n_spills']==0 for v in e['dense_records'].values())
            byte=list(e['producer_records'].values());assert byte
            if official:assert any(r['IS_3D'] and r['sequences']==64 for r in byte)
            for r in byte:
                if is_k8v16:assert r['value_dtype']=='torch.bfloat16'
                assert r['key_dtype']=='torch.uint8' and r['query_dtype']=='torch.bfloat16'
                if not r['IS_3D'] and r['pure_decode']:
                    assert r['BLOCK_M']==16 and r['BLOCK_Q']==2 and r['tile']==16 and r['warps']==1
                if r['IS_3D']:
                    assert r['tile']==16 and r['warps']==r['stages']==1 and 1<=r['sequences']<=64
                    n=r['sequences'];long=64 if n==1 else 32 if n<=3 else 16 if n<=8 else 8 if n<=16 else 16 if n<=24 else 4
                    if r['sequences']>24:
                        expected_segments=16 if state['mode']=='candidate' and 33<=r['sequences']<=56 else 4
                        assert r['segments']==expected_segments
                    else:
                        assert r['segments'] in (4,long)
            for name,h in e['bindings'].items():assert state['frozen_sources'][name]==h
            if is_k8v16:
                assert e['codec_proof']['physical_page_bytes']==12288
                own=e['codec_proof']['actual_writer_source'];assert state['frozen_sources'][own]==e['codec_proof']['actual_writer_sha256']
            proof=e['codec_proof'];assert proof['not_native_FP8'] and proof['format_fixed_scale']==1/16
            assert digest(artifact(proof['writer_path']))==proof['writer_sha256']
            assert digest(artifact(proof['reader']['generated_source']))==proof['reader']['generated_sha256']
    return dict(complete=True, warm_points=warm_points, budget=budget, max_num_seqs=expected_seqs, async_scheduling=True,
        BF16_reference_CV_KV16_S8_verified='candidate'=="control",software8_decode_policy_verified='candidate'=="candidate", native_TopP_source="64c66699cd398c2eafe9eb91f0e189162f83a6f83a209651048e163d7d6fcee8", original_sampler=True, strictTopP32Kgeo=True, M2048_three_projection_tiles=True,
        K8V16=is_k8v16,software8_KV='candidate'=='candidate',bounded_arena_accounted='candidate'=='candidate',stream_interval=1, actual_environment_replayed=True, zero_preemptions=all(value == 0 for value in preemptions.values()), recorded_preemptions=preemptions,
        B64_binding_scope="Official/service C64 validated separately; Math requests use C8.")
