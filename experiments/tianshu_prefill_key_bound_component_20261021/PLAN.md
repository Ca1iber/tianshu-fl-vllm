# Tighter prefill bound: isolated GPU gather proof, not model integration

Prerequisite: independent CPU helper2048metadata/119960logical addresses passed; current8192quality/formal queue owns GPU. This component MUST wait for all live services/evaluators to stop, requires ixsmi noGPUprocess, and hard-stops at2026-10-21T15:58UTC. No new GPU process is launched by CPU preparation.

Execute the actual unchanged pipeline BoundedDensePrefill.__call__ and gather_decode at capacity8192 in an isolated process. Eight explicit pure/mixed/prefix/boundary fixtures times three genuine K8/V-BF16 strided layouts=24cases. Per arm compare EVERY active K/V bit against independent CPU decoder/source values; verify ALL inactive cells and extra guard remain17, dense page IDs and poisoned inactive source table are safe. Original fullkey/prefix lengths retained. Different maxK changes only dense layout/grid in this test. Seven alternating CUDA-event groups over isolated component CUDA Graph replays measure gather cost while excluding Python launch gaps. Those graphs do not establish model capture/replay or end-to-end gain.

Launch later with:
CUDA_VISIBLE_DEVICES=0 python3 experiments/tianshu_prefill_key_bound_component_20261021/gpu_probe.py --output /workspace/benchmark-results/20261021_prefill_key_bound_component_tianshu --stop-at-utc 2026-10-21T15:58:00+00:00

Use fresh result directory for any genuine code correction after GPU has started. Existing completed/running source closures cannot be edited or extended. Model hook/scheduler/arena changes require an independent subsequent experiment and quality validation where applicable; no adoption from component timing alone.
