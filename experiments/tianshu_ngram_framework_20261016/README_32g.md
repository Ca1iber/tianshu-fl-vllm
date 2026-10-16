# Native ngram 投机框架探索

一 token 草稿来自原生 CPU ngram，不加载额外模型。候选必须使用标准拒绝采样、正 FP32 随机阈值、真实接受计数；禁止 synthetic 接受率或 FP64 退化阈值。先重新运行真实拒绝核 CPU oracle 与小型 Q1/Q2 attention 测试，再决定是否上模型。原75点预热/hybrid 对 speculative 有 guard，不能以环境变量为证声称仍然生效。

tiny attention 只调整已检查 BF16/Q16/KV2/head128/page16/Q≤2/B≤64 的二维验证启动配置，保留 BLOCK_M16/BLOCK_Q2、grid、因果/分页/整数位置计算及 attention 算法；其他调用走原路。需要独立 FP32/CPU、真实 strided cache、ragged Q1/Q2、变更 live 输入后 Graph replay 检查。组件收益不能当服务成绩，fresh Math >95% 前不得采用或报正式分数。
