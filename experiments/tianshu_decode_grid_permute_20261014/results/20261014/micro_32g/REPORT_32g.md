# 天数BI-V150 32g：Decode物理CTA轴排列无明确收益

实际.cv/S8/guarded KV16/w1基线上，保留所有原logical CTA和相同算术/reducer，比较原.cv与6个生成副本（含identity控制）。17场景×6候选共102候选全部通过FP32/CPU、原输入逐位比较、改变Q/K/V后的Graph重放逐位比较与尾部保护。1071原始设备event、源码快照/生成源逐字重现、B64七组实际编译产物均已完整下载复核。CPU96grid/37488坐标为补充证明，不冒充GPU验证。

|batch|KV最大长度|最快排列|全Attention耗时变化|
|---:|---:|---|---:|
|8|4096|[0, 1, 2]|-0.1506%|
|16|4096|[2, 1, 0]|+0.0653%|
|24|4096|[1, 2, 0]|-1.3931%|
|32|4096|[0, 1, 2]|-0.1072%|
|64|4096|[0, 2, 1]|-0.1729%|
|64|5120|[0, 2, 1]|-0.2327%|
|64|65|[0, 1, 2]|+0.1002%|

B64/4K最优[0,2,1]仅−0.1729%，5K−0.2327%；identity控制分别−0.0069%/+0.0584%，未建立足够局部收益，不接服务、不跑新Math或正式benchmark。更激进排列在B64/4K反而慢4.6%～4.8%，短KV亦可慢逾20%；不能只挑B24的−1.39%声称全模型有效。此计时是producer加reducer完整unified_attention，不冒充单producer或模型吞吐。

完整归档4001834bytes/16parts SHA256 `18d54af603464b65cc06e8729cd8bb264b91687102d42ee39260a744fba079de`。原件service-results/20261014_tianshu/decode_grid_permute/20261014_decode_grid_permute_tianshu；本目录verify_downloaded.py --root <raw>/micro可复核。原生安装源未修改，GPU串行接续maxseq服务完成后执行。当前采用4091.29/Math103保持。

下一值得核对的方向：reduce_segments独立launch配置。安装wrapper没有显式传num_warps，归档iluvatar backend默认4；既有producer warp1/2扫描没有扫描reducer warp。只有源码依据，尚无新GPU结果或收益结论。
