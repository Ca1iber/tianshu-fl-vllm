# 逐M GateUp融合服务同源4K对照

原qualified450版（正式5618.01）为两臂共同父版本，保持原M64静态FX融合、M8/LM量化政策、K8V16/attention/allocator/scheduler/75点捕获及四对index预热和原13参数serve命令。只改dynamic编译范围[1,4096]的GateUp linear+SiLU链；mode control观察不改，candidate替换为opaque op，按已完整42bank验证的正收益M配置融合，其余M调用原linear+从现用生成代码提取的同算术激活，实时取current stream。M64若进入dynamic仍委托旧融合。原compile_sizes [64]保持，不额外扩编译shape/权重bank。

23个M满足完整MLP>=1%收益，范围由实际24M全测归纳：M128；384；512..1024 step128；1152..2048 step128；2304..4096 step256。M256负收益排除，精度白名单不扩，未知M原路径回退。新kernel实际binary每M必须匹配3024场景组件元数据，startup在qualified capture后预热每个选定M并独立CPU真实W整数/激活验证，CPU/GPU RNG及CPU线程数保持。fallback activation在capture前预热且用动态stream。不能把第一版fixed-stream无效结果混用；原始已撤回。

运行时完整继承450-source审核：两臂legacy mode都candidate保留旧M64；只把prefill实际执行证据从旧NN MM改为真实新fused路线，原parent warmup/42 bank/物理layout/权重/allocator/KV/attention/scheduler/LM全部检查保留。自己的动态pass 43图、42真实bank替换，静态pass保持，要求全部42个M4096真实runtime路由且binary与组件一致。legacy动态不变/M8不变字段仅指旧M64 pass，新pass的显式更改另行记录，禁止冒称所有动态图未变。

CPU阶段实际synthetic FX matcher/改写guard与prepare两臂，不初始化CUDA；GPU前冻结全部入口/PLAN/CPU gate/selected plan；首次GPU后修复另目录。原4K/1K C64 N64三轮/排首轮，control和candidate各一次，逐轮向用户报数；0就绪后autotune、0preemption、请求全部成功，同源仅PYTHONPATH/本mode变化（两臂各自输出记录目录规范化后比较，原始路径差异单独保存）。两臂完成后独立CPU比较；候选有稳定>.5%整体收益才进行新官方Math105（严格>95%）及C64实际覆盖，质量通过再正式N256四轮。无例行16K。

开始时必须GPU空闲，当前serial_queue的single-dot另一个实验必须终态；单服务1800s上限/全程23:58北京硬停。此轮不声明正式新纪录。代码报告关键结果在天数容器/GitHub，不留host副本。未知或失败不得跳审核/重抽Math；仅metadata错误可独立CPU恢复保留原failed。


v2修复：v1因未启用的Projected诊断占位张量形状导致SME mask19、不同于组件83而被binary检查拦截。新调用复用已有2D BF16输出作未用Projected，恢复83，绝不改全局选项、内核源码、精度、额外storage或二进制检查。原v1失败完整保留，CPU诊断见tianshu_m_range_compiler_diag_20261021。
