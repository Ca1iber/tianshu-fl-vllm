# 天数 BI-V150 32G：KV页号标量读取无收益，B64最终机器码相同

当前.cv/L0/KV16-w1/B64S8基础上，独立生成SCALAR_KV_PAGE副本。只在3D且TILE_SIZE==BLOCK_SIZE时把页号读成标量后广播，保留int64地址、原浮点计算、.cv、grid和reducer。关闭该标志的生成副本作为恒等控制。源码精确逆替换及CPU66810个页映射检查通过，源提交ac6bce0c0d96eb1ac6f339285b09ea4b628f9df1。

17场景×2候选共34例全部GPU FP32、选定CPU参考、原输入逐位对比、变Q/K/V后的Graph逐位对比及输出/scratch尾部保护通过。实际tile/warp/stage/S8和相同grid均核验；B24保持KV32/w2，标量守卫未生效。每个场景3个版本各9次交替event，全部459个原始值的中位数及变化比例已本机重算。

| batch | KV最大长度 | 启用标量页号的耗时变化 |
|---:|---:|---:|
|8|4096|+0.08858%|
|16|4096|-0.94494%|
|24|4096|+0.28768%|
|32|4096|+0.03896%|
|64|4096|+0.02355%|
|64|5120|+0.00138%|
|64|65|-0.11743%|

C64/4K对照0.846087875ms，候选约0.846287155ms；5K对照1.052348175ms，候选基本持平。不挑B16局部约-0.94%外推整模型收益，不做服务接入、新Math或正式benchmark。当前正式4K3992.97 total tok/s、Math103/105与最佳启动器保持。

## 编译产物证据

只对实际B64/4K编译保存了三个版本产物，故以下结论限定该场景。LLVM中`llvm.bi.load.kop.v1i32`静态调用数为原版36、恒等36、标量5；BF16 load调用均48。三者真实n_regs85、n_spills0。LLVM静态调用数不是运行时执行计数。

完整cubin SHA彼此不同，因为文件还含调试/来源信息；直接解析64位ELF section headers后，三者可执行`.text`均11272bytes且字节SHA相同：`03685e8545d29ef42c3194a1ceff445a8a9ba2bf11cce8bf3cced53546405086`。这直接证明被测B64候选最终没有改变可执行指令，解释为何未提速；不能只凭中间LLVM精简宣称改进。源码和产物完整SHA见RAW_AUDIT_32g.json、LOWERING_AUDIT_32g.json。

## 复核与保存

父runner34617与GPU子进程34619均终态，ixsmi确认GPU空闲；10份运行源码冻结及本机对应文件一致，最佳50份运行源未改、沐曦仓库只读。没有下一GPU任务排队。

原始包3996159bytes/16分片，整包SHA256 `102402322c046958c511a287d9601e988b60a448501f654ecda4db39cd838ca3`，每片及整包均核验。本机原件位于`service-results/20261016_tianshu/scalar_kv_page/20261016_scalar_kv_page_tianshu`。实际复核命令均通过：

```bash
python3 experiments/tianshu_scalar_kv_page_20261016/verify_downloaded.py \
  --root ../service-results/20261016_tianshu/scalar_kv_page/20261016_scalar_kv_page_tianshu
python3 experiments/tianshu_scalar_kv_page_20261016/audit_lowering.py \
  --root ../service-results/20261016_tianshu/scalar_kv_page/20261016_scalar_kv_page_tianshu
```
