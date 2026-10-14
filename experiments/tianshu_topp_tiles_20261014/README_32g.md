# 天数32g：已采用native TopP的tile和高warp对照

基线为已采用8192/4096/w16/s1，非旧w4/stage3。比较身份控制、4096/2048的w8/w16、2048/1024/w8、16384/8192的w16/w32及8192/4096/w32，共8配置。原native源SHA64c66699...e8，body/RNG/p=.95/k/vocab130560/grid/scratch分配不改，只覆盖BLOCK_SIZE、BLOCK_SIZE_TRUNC、num_warps和stage1。读取实际wrapper已证实scratch按vocab分配，不依赖tile。CPU假JIT验证真实Router所有8路由参数，非性能证据。

GPU7场景覆盖B8/32/64、normal/peaked/-inf/ties及k100+p，逐项与已采用基线相同，或独立CPU FP64 mass/minimality/order在1e-6内通过才允许计时。CPU先传回host再double，保留已知baseline部分严格误差，不能放宽容差。保存逐行概率质量、最小保留概率、保留/移除边界及CPU门槛供离线重算。翻转词表后重新生成独立oracle、直接执行并Graph重放精确比较，两输入都达相对门槛才计时。7轮交替Graph，每图10回放含相同输入恢复copy，保存B64编译产物和真实tiles/warps。

候选编译/资源错误完整记录；只有正常baseline/identity通过才能完成，未通过门槛的配置不写吞吐提升。当前没有新GPU/服务/Math/正式分数，当前4091.29最佳保持。若micro明确改善再独立4K服务对照，不把过滤速度当模型成绩。
