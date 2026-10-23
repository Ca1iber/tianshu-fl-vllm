# r3 + dotsaddz Down64运行时W4，单一模型候选

903个r3模型来源、13项官方serve argv、63项最终环境（仅PYTHONPATH实现入口变化，C64仍按既有质量规则关闭FutureKV）原样继承。只将真实42层Down的M64从BF16改为已完整真实权重组件验证的group128 RTN W4A16 dotsaddz。其他M继续原nativeDown/bf16，QKV/O/GU/LM/KV/scheduler/sampling不改，原checkpoint和架构不改。LM W4组件有收益但不在本次模型中，避免精度归因混杂。

使用冻结5c79e84的全部42层packed文件，加载时核对原始GPU BF16权重、CPU文件及GPU packed/S/Z全SHA。注册非持久buffer，保留原BF16和W8供其他批次回退。所有新增buffer和42个native零输入预热在真实ModelRunner.load_model的DeviceMemoryProfiler内；在实际Worker.determine_available_memory的profile_run窗口显式执行42个M64 scratch探针（与真实M4096 profile分开记录，不伪称实际大批次执行），RNG不变、手动KV字节None，记录实际allocated/peak及模型所有唯一parameter/buffer storage字节。

新的Torch自定义op只接M64连续BF16输入，内部实际调用 aten._weight_int4pack_mm_dotsaddz；输出新分配BF16。捕获时记录真实42个bank/输入stride/graph上下文。图重放不产生Python计数，不能把CPU调用计数当replay，链接已有真实M64模型图执行证据。

审核唯一替换：旧native_down审计要求42个M64捕获为BF16，与本次精度变更矛盾；在本候选中复制旧审计到新文件，要求旧Down路由中完全没有M64 capture，交给新独立审计严格检查42个W4 bank/真实capture、加载/预热/内存/源和库SHA。所有其他旧Down检查、原84个attention M64 BF16检查，以及旧875+ GU/KV/scheduler/sampling/LM质量检查保持执行。不会伪造或改写旧record，不写原已冻结文件。

先冻结全部源码/控制器/CPU准备后执行新鲜C64完整Level3 105题，temperature1/top_p.95/max_tokens32768/C64，所有error/cap/空输出计入总数；严格>95%。若失败不挑选重跑、先分析而不是追分。通过才执行同模型新鲜C8，随后才能新短4K N64x3/exclude1。不得直接跑正式或16K。全部GPU在北京时间23:58前终止；代码/报告/关键评测结果GitHub，大数组容器保留。
