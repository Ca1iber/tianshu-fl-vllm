# QKV64 keep128 发布草稿（未获准发布）

本目录先准备可复现的发布工具，不代表新版本已合格或已替换r3。必须等待同版本新鲜完整C8/C64各105题均严格>95%、N64三轮排除第一轮同请求对照通过、N256四轮排除第一轮正式4K与独立CPU核验通过。没有此版OFFICIAL_GATE.promotion_passed时，build_release拒绝执行；没有QUALIFIED_RELEASE.sha256时，启动器拒绝启动。

本候选以r3原903项模型输入为精确子集，共928项；只增加QKV M64在FULL/64tokens/64requests/均匀解码/无LoRA条件下的原生INT8，其余M64与O64保留BF16，QKV128继续原生INT8。没有合入LM925 pad候选。原checkpoint和结构、13项serve参数、63项环境沿用；只允许输出记录根路径重定向，启动前清理继承实验环境。

发布门槛在已冻结的formal verifier中预先定义：原N256×4全部1024请求成功、热三轮极差/均值<1%、相对r3正式5950.75 total tok/s提高>0.5%、TTFT不高于3174.58ms×1.01、全部继承和新增数值/路由/显存/来源审计通过、零抢占、就绪后无Triton autotune。正式历史比较不冒充同机同时A/B。没有新16K结果，也不据此声称完整比赛资格已审核。

额外缓存清单只记录原384个严格二进制缓存文件及实际M64去量化的8个文件SHA，共392项；不上传缓存或权重。M64 CUBIN SHA a701127a8e57c1e53b9aaad83945dd152cea3d84485aa3c4d8eb9f5ffed5905a、实际Triton kernel hash 40c78275879dd786891ff407cfadd9c7e261c1d94cdcaedd9723a3ad2de592c4。原始模型、编译缓存、题目、预测及数组都只在容器。GitHub只包含代码、报告、关键JSON/CSV和六份必要控制JSON的gzip。

build_release通过后才产生QUALIFIED_RELEASE.sha256与实际分数README。打包不启动GPU；随后必须从仓库根和/tmp各做启动器dry run，注入错误实验环境确认被清理，核对928个模型输入、实际全部控制器、13参数和63环境。审计完成前不创建新合格标签。
