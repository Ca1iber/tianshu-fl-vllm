# r5量化KV消费者：独立BM8/direct候选

两组均保持已发布r5的997个来源和全部W4/W8/RMS/采样/模型权重，父prepare固定candidate；仅本候选的candidate装旧v3 M64-only BM8选择器，control保持r5成对K读取。K8/V BF16、writer、softmax/reducer、分段4不改；IS_3D/N64 query tile16→8、BLOCK_Q2→1、grid96→64，其它M/2D原样委托。旧8044544元素bitwise组件资格和严格机器码签名作为复用前提，不能借用旧模型质量或吞吐成绩。

源与控制器在新CPU_GATE冻结，原r5的997来源逐项保持。source_snapshot新增小文件，大SDK继续引用已经独立捕获的不可变SHM副本；严禁重建/覆盖其根。所有新大输出先放/dev/shm/tianshu_r5_BM8_runs_20261024，正式工作盘只保存代码、报告、关键结果和经校验的紧凑原始档案。禁止主机文件副本。

启动时必须核对实际BM8最终JIT源、网格、BLOCK_M/Q、零spill及非debug机器码签名，并与uniform64 Graph capture绑定；真实C64运行必须观察到>=100次Graph重放。旧pair记录在子映射之前，只可证明父路径，不能作为最终BM8核验证据。control与candidate相同模型来源和相同payload，实际QKVO fitted模式两组均candidate。

新进程分别跑完整C8/C64 Math Level3共105题，每题均计入、>95%，不得使用旧分数。C64质量测试沿用r5质量协议暂时禁用未来KV准入以实际到达64并发，此开关必须记录，不能默认为正式吞吐设置。性能均启用原r5未来准入。同场freshN128每轮独立seed、两组对应seed相同，prefix hit<=1%、零抢占/就绪后autotune、无计时期间CPU准备。预先排首轮，热3轮吞吐极差/均值<1%；候选总吞吐须>对照1%、平均TTFT不得退步>2%，不达标不正式。通过新质量和短测后才原始N256×4排首轮4K，不常规跑16K。23:58北京时间硬停。

本计划定义待验证候选，不代表已采用；最新正式成绩仍为r5 6123.31 total tok/s。旧短N64 +2.4325%及C64 103/105只作研究依据。
