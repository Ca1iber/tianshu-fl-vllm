# K8/V16 decode组件：B64减时28.68%，尚未接入完整模型

source4463e0f5f4c58d6717f4a649f73dd2bf023b08ad。10种batch/上下文×5路径、两次变数据/slots/长度同Graph，实际native FullAttentionSpec/page_size_padded/reshape产生的raw allocation与typed K/V共享存储。UInt8 K/BF16 V的page为12288 bytes，对比当前双UInt8的8192，访存量增加50%仍可能减少解码成本。未修改FullAttentionSpec的真实内存计数来伪造容量。

|B/ctx|当前8位原writer ms|8位自有writer ms|K16/V8 ms|K8/V16 ms|K8/V16变化|
|---|---:|---:|---:|---:|---:|
|1/33|0.018927|0.018531|0.018183|0.017326|-8.459%|
|1/4096|0.044884|0.044420|0.045945|0.035605|-20.674%|
|8/4096|0.121851|0.122036|0.204236|0.087503|-28.188%|
|16/4096|0.220140|0.220035|0.383533|0.157752|-28.340%|
|24/4096|0.330084|0.330001|0.427096|0.235368|-28.695%|
|32/4096|0.417075|0.416736|0.745771|0.298041|-28.540%|
|64/4096|0.811207|0.811054|1.174633|0.578566|-28.678%|
|64/5120|1.007012|1.006408|1.468874|0.716633|-28.836%|
|128/4096|1.645844|1.644571|2.940240|1.231391|-25.182%|
|256/4096|3.202456|3.200281|4.733245|2.440775|-23.784%|

B64/4K原byte 0.811207→K8/V16 0.578566ms（-28.6783%），5K -28.8357%。同次byte自有writer只快0.0188%，主要差额无法归因writer替换；所有路径计时包含实际writer+reader。保持当前byte的B64S4/T16/w1与相同reader源码，仅dtype专门化不同。B64寄存器245→142、spill均0；K16/V8为166寄存器但更慢，不能从寄存器数单独解释原因。BF16控制使用本probe记录的配置，尤其大B 2D是T128/w16；不把它误称为当前所有BF16最优配置，关键公平比较为byte与混合三路。

本地核验168源码/生成源、65536实际writer编码、256解码码值；100 GPU参考记录、1040个128元素CPU向量（133120标量）误差独立重算、350原始event组与全部median通过。记录的全raw物理存储/输出/scratch/输入/RNG检查均通过。没有序列化全GPU Tensor，不声称本机重算完整attention。

当前仅直接writer+unified decode组件，**没有安装真实ModelRunner混合cache生命周期，也未测prefill、新Math或服务吞吐**。后续只推进K8/V16，K16/V8负收益停止；正式best仍4259.87与同profile新Math102/105。不能把单层28.68%直接当整机收益。

原件service-results/20261019_tianshu/soft8_mixed_planes_v2_component/20261019_soft8_mixed_planes_v2_component_tianshu，7259219B/28parts与整包SHA通过，SHA256 `554c24de65c707ed818e0264d6113c6167e261badf31995f9e7b9f17738e30cc`。

```bash
python3 experiments/tianshu_soft8_mixed_planes_20261019/verify_downloaded.py --root ../service-results/20261019_tianshu/soft8_mixed_planes_v2_component/20261019_soft8_mixed_planes_v2_component_tianshu/micro
```
