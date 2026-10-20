# 当前K8V16/default4096正式16K补测：运行中

对应已过Math102/105及正式4K5101.45的同一205源、43项配置模板，原始serve命令与benchmark脚本均未修改。仅test-cases选择官方16384输入/1024输出、C64、N128；原脚本自带四轮排首轮。

直接Shell启动已冻结的最佳launcher，再调用原始benchmark；脚本和服务分别设超时，异常也清理服务。CPU原有两处WIP不暂存或改写。没有叠加今天失败的V压缩、页间隔或M64新配置，也没有新权重量化。

新结果尚未产生。此前正式16K为2026-10-13旧版本2665.42 total tok/s、156.79 output tok/s、TTFT184434.62ms。当前版结果必须单独审计成功数、四轮CSV、实际配置/容量/预热/源；不以4K外推16K，也不把4K专用零preemption门槛误当16K运行事实。

全部原件留容器 `/workspace/benchmark-results/20261020_k8v16_official16k_tianshu`，关键结果完成后提交GitHub。
