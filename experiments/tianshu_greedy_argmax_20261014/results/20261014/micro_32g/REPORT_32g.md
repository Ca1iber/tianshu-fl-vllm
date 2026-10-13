# 天数 BI-V150 32g：贪心Argmax微测正确且B64耗时明显降低

只读当前官方benchmark backend确认temperature0贪心；历史短测temperature1/top_p.95，采样热点不同。本实验对实际Sampler.greedy_sample静态函数（torch.argmax）作单独比较，尚未证明模型运行时实际provider及热点绑定。正式3640.16原脚本成绩有效，不把本micro写成新的整体吞吐。

新two-stage vocab分段最大值/索引归约，私有index32、返回int64；保留第一个并列最大值和第一个NaN优先规则，无输入写入/无RNG消耗。5形状/stride×6模式×5tile配置=30场景/150候选，CPU/native/Graph indices全部精确一致，NaNs/正负inf/ties/极值/非连续row均覆盖，输入位和CUDA RNG状态不变。

|B|vocab|strided|最好tile|native ms|candidate ms|耗时变化%|
|---:|---:|---|---|---:|---:|---:|
|1|130560|False|{'tile': 1024, 'warps': 4}|0.047002|0.012084|-74.291563|
|7|130560|False|{'tile': 2048, 'warps': 4}|0.148366|0.014263|-90.386653|
|64|130560|False|{'tile': 2048, 'warps': 4}|0.560375|0.063134|-88.733582|
|64|130560|True|{'tile': 2048, 'warps': 4}|0.570546|0.074344|-86.969631|
|3|13|True|{'tile': 8192, 'warps': 8}|0.011854|0.011838|-0.141461|

B64实际vocab130560的0.560375→0.063134ms（-88.733582%）只代表单次argmax，节省约0.50ms/步；不能等同于模型吞吐提升88%。5个normal形状保留7轮交替、每图50次重放原始样本并核验。小vocab13没有明显优势，应保留原生fallback。

source implementation23ace17a292f4f20ce2a8b69c3f65146af023cac352281bdda3143035e4f01d0，probe85c250aecc74098dac1f51c0fbdc119ecc0c671f593e88c4e0cae67f661431ad；actualSampler source1a2f18ddc91c89f3744ab5a93624be056bf20b7a8b2e6e13d0236182718a2253。三份源snapshot逐字核验。

全原始包2542191bytes/10分片，SHA256 f68af14638bd9577d76410c4ebe22a9ea91f8b5ab7afb3c10d285b0e73625a8c，全部分片/整包、150结果及源/原始timing已本地核验。原件service-results/20261014_tianshu/greedy_argmax/20261014_greedy_argmax_tianshu。

下一步：先用对齐正式temperature0/top_p1的贪心N64服务比较，实际请求阶段marker和双组调用记录证明绑定，不能把启动预热TopP/argmax记录当真实请求热点。保留其他Sampler处理和公开index类型。若端到端收益明确，仍需新独立官方Math105>95后正式4K。当前最好3640.16/728.03/TTFT4465.17/Math103不变；无新16K，硬件profile覆盖未证实不作占比推断。
