# cadence16新质量与原正式4K验证准备

尚未启动，也没有新Math/正式成绩。只为当前独立N128/C64完整3+3筛选准备后续：预先固定门槛至少+1.0%（与官方质量>95%分开），后两轮candidate均高于control、各稳态极差<0.5%，均值从128*5120/duration重算且128响应usage正确、三轮全部实际补请求。旧N64 +0.95%、N128 +0.647%不满足本门槛，不以它们直接触发完整评测。此门槛适用于更大N128筛选，不修改历史GEO的>=2%门槛代码。

继承已验证GEO基底准备器，再要求新screen的70源完整包含该基底并字节相同，只重放screen candidate的40环境/原serve argv与cadence16 policy。新driver复用原105题Math解析/原N256四轮benchmark与严格>95质量门槛，两stage同一80源（screen70、5编排/门槛与5个EvalScope文件）/同一normalized环境；不沿用旧102/105。C1/C8 canary和真实恢复GPU token/position检查在评测前重新执行，关闭同步arm后核验journal不变。原temperature1/top_p.95/max32768、batch8、math500 Level3路径不变，无缓存复用。

Math启动至少需剩70分钟，正式至少25分钟，全部15:58UTC硬停。新Math需完整105/零错误、独立下载审计通过才会运行原4K/C64/N256x4，逐轮报告；无16K。保留全部原件和源，不修改旧冻结目录、最佳启动器、模型权重、安装或沐曦，不自动GitHubpush。

实际cadence16筛选只有+0.5425%，check_screen按预定>=1%门槛明确拒绝。此验证套件保持未运行GPU/新Math/正式状态，CPU历史103/105解析检查不当本候选成绩。
