# Isolate V8 physical page stride

Previous V E5M2 component passed all codec/numerical guards but FP16_PV was ~32% slower than current K8/V16 at B64. This experiment tests a new layout hypothesis, without assuming a hardware cause.

Three arms in the same process: current K8/V BF16 (12288-byte page), unchanged V E5M2/FP16_PV in an8192-byte page, and the exact same V E5M2/FP16_PV in a12288-byte padded page. K starts atbyte0 and V atbyte4096 in both byte candidates; the padded candidate leaves the last4096 bytes untouched and guarded. The two byte arms share the same encoding, V/P precision, reader/source, writer, Q/K and launch policy. Only the physical block stride/allocation differs. Both receive the same logical inputs and block table per phase.

Reuse the GPU-qualified v2 codec and CPU gate via symlinks; rerun all65536 writer encodings/K-preservation checks and all256 decoder classifications anyway. Test only B64/4096, B64/5120 and B64 ragged/strided, each3armsx2changed-data same-Graph phases. Keep complete GPU FP32/selectedCPUFP64 references, allinput/cache/output/scratch/RNG guards and9alternating trialsx20replay.

Require at least5% less latency than current K8/V16 at both4K/5K before considering model work. If12KiB remains slower, stop this padding variant. If12KiB materially rescues8KiB and beats current control, then separately predeclare9/10/11KiB candidates to seek actual cache-capacity benefit. Do not attribute results to occupancy or memory partitions without further hardware evidence. No engine/Math/formal service run in this component. Rawdata stays in container; GitHub receives code/report/keyresults. Hard GPU stop2026-10-24 23:58BJT.
