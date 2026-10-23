# Down64 + LM64 W4原始正式4K独立CPU核验

只读取本934来源原始N256×4/exclude1正式结果，重新核对全来源及快照、13argv/63环境、官方脚本SHA、1024请求与完整token数、官方summary为后3轮均值、零runtime autotune/抢占、全部继承GU/KV/attention/scheduler/sampling/LM8/其他Down和新W4实证、checkpoint和GPU退出后空闲。新鲜C8正常complete与C64原audit_failed经单独CPU校正必须都>95%，校正代码/数据/原失败SHA必须精确。不重跑GPU或修改raw。

执行前晋级阈值固定：相对已采用r3 5950.75 total tok/s增>0.5%，后3轮spread<1%，TTFT<=3174.58*1.01；必须先配对短测通过、完整同934质量独立复核。历史正式比较不冒充同次完整AB。计时期间不运行CPU准备。当前正式尚未开始；本脚本不启动GPU、不构建发布。无16K。
