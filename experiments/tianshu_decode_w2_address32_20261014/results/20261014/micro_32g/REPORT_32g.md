# 天数 BI-V150 32g：当前L0/KV32/两warp地址32主负载变慢

旧地址32测试基于KV16/warp4，当前正式最好已改L0/KV32/warp2-B8..64（其余warp4），因此重新检验不同编译布局的地址窄化。保持当前浮点源4a1f8b、KV32/stage1及原adaptive分段，仅28 stride注解和6整数offset casts改int32；生产与合并两阶段都用对应32位生成模块，物理指针64及FP source不变，不改installed source。

8长+14短/非连续/0-query填充共22项，全部通过GPU FP32、选定CPU独立参考、native容差、输出/S64 scratch尾部和Graph重放，22/22逐位native相同。所有实际tensor的元素offset加2048 masked余量均<2**31，最大369874943；包括实际Qstride/交错KV和BF16展开未用K/V descales。实际producer/reducer source module一致，baseline reducer为原64，避免仅改一阶段后误读数值。

|B|KV最大长度|warp|64位ms|32位ms|耗时变化%|
|---:|---:|---:|---:|---:|---:|
|1|4096|4|0.035458|0.035102|-1.003575|
|8|4096|2|0.149702|0.159099|+6.277613|
|16|4096|2|0.332618|0.354866|+6.688771|
|32|4096|2|0.533765|0.570308|+6.846199|
|64|4096|2|1.000435|1.065066|+6.460308|
|64|5120|2|1.244018|1.324880|+6.500072|
|16|16384|2|1.290431|1.373138|+6.409234|
|2|32768|4|0.415800|0.430082|+3.434750|

B8..64主4K/5K负载变慢约6.3%–6.8%，B1仅微小改善，没有进入服务实验的明确收益。不采用，不追加服务/Math/full；更窄整数不保证当前编译布局更快，原因需profile证据，不猜测硬件瓶颈。长16K/32K只是算子边界，不是16K服务成绩。

三份probe/helper/router与源snapshot逐字、L0原4a1f8b与生成32源sha、28/6替换数量、实际模块/warp/TILE与bound及9轮交替Graph/每图50次原始timing本地核验。两份原始包各10分片+整包SHA核验：

|包|bytes|SHA256|
|---|---:|---|
|long|2573479|295f89c3e10cde52b34d13717311f567e094f0768a3a9a07ba98b4cca575a5b9|
|short|2574241|db0d3ecc690857a44e9b332d31249148ace7be5b92054d6004112de8e9546a8b|

原件service-results/20261014_tianshu/decode_w2_address32_{long,short}。当前正式最好3640.16/728.03/TTFT4465.17/newMath103/tag tianshu-best-decode-kv32-w2-20261014保持。profile：TODO有效图内kernel覆盖；模型/权重/原benchmark不变，沐曦只读。
