# Joint fitted QKV/O64 original full 4K validation

Run only if the completed fresh-input N128 paired gate qualifies (>=1% mean total throughput gain, both hot-three spreads <1%, per-round prefix hit <=1%, zero preemptions), both new C8/C64 Math runs exceed95% with all105 predictions counted, and actual GPU is idle. Use the exact997 model sources and original13 serve arguments already bound to these tests; no edits inside the frozen model directory.

Use the untouched original benchmark_throughput_serve.py (SHA0813eba574e9d1eb50c297c515baa426da1bcb61ce8c78a398b482f246005565), test case [[4096,1024,64,256]], served model minicpm, port9031. Preserve RUNS=4 and SKIP_FIRST=1; record all four raw rows, all1024 requests and the hot-three mean/spread. Record actual prefix-cache queries/hits and preemptions before/after the whole original script. This does not change the original script or inject per-round seeds. The preceding fresh-input N128 pair remains the paired evidence.

A standalone N256 score is compared explicitly against the historical r4 official6050.94 total tok/s, not the N128 values. Do not adopt merely from the best single round. Require successful full evaluation, independent profile/source/quality/controller audit, all requests successful and hot-three spread<1%, zero preemptions and prefix-hit fraction<=1%. Any gain claim must disclose the historical versus paired comparison scope.

Controller cleanup terminates only its own process groups, verifies actual GPU idle, and enforces hard stop2026-10-24 23:58 BJT. No concurrent independent CPU preparation/hashing/archiving/push during timed client runs. Raw data stays in container; code/report/key metrics go to GitHub. No16K test or automatic adoption in this controller.
