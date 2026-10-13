# 天数 BI-V150 32g：Decode KV32两warp服务小测提升4.05%

在当前TopP16/Decode L0 KV32-w4-stage1基础上，只将batch8..64的Decode生产kernel改warp2，其余batch保留warp4，合并tile32保持。同期4K输入/1K输出、C64/N64各3轮，预排首轮：控制3525.854514→候选3668.522003 total tok/s，提升4.046324%。候选后两轮极差/均值0.107640%。这份为小测，尚未新Math或正式评测，不能当作正式最佳。

|轮次|控制tok/s|候选tok/s|控制TTFT ms|候选TTFT ms|计分|
|---:|---:|---:|---:|---:|---|
|1|1316.725751|1310.067950|19091.020502|20688.018251|否|
|2|3525.056989|3666.547597|13252.692835|13221.506673|是|
|3|3526.652039|3670.496410|13289.303438|13214.987792|是|

384/384请求成功、实际usage4096/1024/5120及逐轮提示seed相同；7个greedy探针相同，不替代官方随机采样质量。两组六份安装源码和实际server argv完全一致，TopP16/nativePrefill L0/B75G1/budget2048保持，不混入IRRMS、RoPE、地址int32、量化或投机。两组都实际完成75点预热，六轮没有新增调优记录。

候选实际B8/B64 compiled metadata num_warps2、num_stages1、shared20480；B1记录warp4。全部生产与合并的batch/segments配对且tile32相同，generated L0源4a1f8bdfdb5b9399702941a0e499e62b52f104737aba6012a9056ef36f5c4d4e，kernel body/grid/指针与整数位宽没有新增修改。七份冻结源与snapshot逐字核验，主候选hook SHA6aeea3463a3f0177d49704cae190af9c1642d4c893091ed935f6347bb9b50a59，Decode hook SHA4d5cda1a9f8bc6c6df752a41a1d44be38d248ab2f574d34f3e86e7cf05b94ba6。

前置22场景/66候选的独立CPU/GPU FP32、native逐位、Graph与尾部检查见同目录micro报告。微测B64/4K耗时下降7.93%不能直接当作模型收益；本轮另证实服务提升4.05%。尚无硬件profile计数或有效图内kernel占比，不推测硬件机制，TODO。

全原始包4158890bytes/16分片，SHA256 50f757ae4f61dd892ed90b6306882289e1a11e3979cf4b566d6ab33c8ae00267。所有分片/整包、384响应、七份源与实际CLI/路由、75点预热及均值已本地核验。原件service-results/20261014_tianshu/decode_low_warps_service/20261014_decode_low_warps_service_tianshu。

结论：收益明确，进入新控制/候选各5轮4K/1K/C64/N64复测，预排前两轮。确认收益后新独立实际warp2/TopP16组合Math105严格>95%，再跑原正式4K N256四轮排首轮；不复用旧103/105，也不追加16K。正式最好3506.84/TTFT4494.12/Math103/tag tianshu-best-topp-w16-20261014保持，报告及Git保存，沐曦只读。
