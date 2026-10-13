# 天数 BI-V150 32g：私有索引32服务及热路径核查

对应ops微测98aaffe：私有索引32、最终返回int64，120项GPU精确值与稳定索引检查通过，M64词表sort24.954239→21.397672ms（速度约+16.6%）。不改模型、浮点值、全局计数或installed source。

N64/C64/4K1K五轮、预先排除首两轮，复用原版control3253.050661。候选320请求成功、7greedy匹配，计算源与usage通过校验。该local筛选不是官方N256分数。

|轮次|total tok/s|耗时/s|计入|
|---|---:|---:|---|
|1|1271.269151|257.758162|否|
|2|3230.802138|101.423729|否|
|3|3247.438158|100.904154|是|
|4|3236.348798|101.249902|是|
|5|3259.731337|100.523622|是|

最后三轮3247.839431 tok/s，相对control差异-0.160195%。无清晰服务收益，不采用、不重复完整benchmark或Math，正式最好保持4K3222.52/16K2440.62/Math102/105。

实际routes仅M1..7，均在第一轮刚提交时（metric_before03:49:56UTC，routes03:49:58..03:50:04）。小批次使用private32/public64可证，不能当M64路由证明。

关键平台区别：topk_topp_sampler.py:356，HAS_TRITON且batch>=8直接调用原生Triton top-p，仅小batch回退PyTorch sort。天数保留此分支，MetaX补丁强制PyTorch版本。M64 sorter微测虽形状与词表相符，却不是天数大批次热路径；这解释了排序micro提升未转化服务吞吐，不能直接迁移平台收益。

进一步审计：原生Triton top-p已用int32/uint32；InputBatch token/top-k/positions、block table也多已32。slot_mapping数据、attention/KV地址strides与casts仍64，可在严格验证局部元素偏移范围后设计guarded32算术，物理指针和必要公共64接口保持原类型。尚未进行该新候选GPU测试。

原始归档3269537bytes、13parts，SHA256 2d1d98db4a30333acf568ddff277abf7969cc8923fa86943b5086a76556e2ded，本地完整核验。含320响应/usage、routes与源。分支原始源码另备份service-results/20261013_tianshu/sampling_route_audit。未采集硬件profiler，官方benchmark不改，沐曦工作副本只读。
