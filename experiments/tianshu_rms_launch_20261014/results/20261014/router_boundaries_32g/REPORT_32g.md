# 天数 BI-V150 32g：真实RMS小批次路由80项边界检查通过

候选使用与原版相同的Triton JIT函数体和grid，只在BF16、连续N2048、M1..64、相同输入/输出行列stride条件下改为w8/stage1；其他形状走原LibEntry。hook仅进程内替换module kernel对象，不修改安装源。与当前TopP正式最好3506.84的hook组合；本次仅数值测试，TopP及Decode开关设0避免加载模型。

调用实际flag_gems.modules.normalization.gems_rms_forward，80项：M1/7/16/64/65/128加N1024及FP32回退，分别随机/零/残差抵消/大/小数值，普通和fused两种路径。80/80通过CPU FP32参考（BF16 rtol=.016/atol=.016）及Graph回放与同配置eager一致。77/80与原版逐位相同；3个不同都为fused/M64/BF16/N2048，分布random/large/small。保留浮点差异，不声称所有输入bitwise一致；用户质量门槛仍是对应新独立Math准确率严格>95%。

实际8组(kind, M)记录只包含normal/fused的1/7/16/64，compiled metadata均w8/stage1；M65/128、N1024、FP32没有进入候选路由，覆盖guard回退。LibEntry自身cachekey不含launch kwargs，因此候选调用同body的原.fn保证参数确实生效，原LibEntry只用于回退和本次AB基线。

原始包2297107bytes/9分片，SHA256 7739c01b4efce844e597ef044a3a7238396266f5379a29ab65172f0c98fd906d；整包/分片、80唯一case、8真实路由、probe及hook快照均已本地核验。hook SHA 48fe65892c63c90a807cec9d21ae71fadaddfc95bc7a1966d0ea151a91873d28。原件service-results/20261014_tianshu/rms_router_boundaries/20261014_rms_router_boundaries_tianshu。

结论：可以进入同期小规模4K服务对照，尚无新的模型性能分数。若服务收益明确，再重新独立Math105及正式4K，不复用TopP旧103/105；不跑16K。当前最好仍3506.84/TTFT4494.12/Math103，tag tianshu-best-topp-w16-20261014。性能剖析：TODO，Graph内部覆盖尚未得到有效计时。安装源、权重和官方benchmark保持原件，沐曦只读。
