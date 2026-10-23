# 两组服务之间整理未冻结FX缓存

磁盘可用约1.2GiB。仅暂停外层Bash进程1807，当前control Python/GPU子进程继续运行，不影响任何计时轮。等control三轮完成且原审计通过、GPU无进程后，备份/root/.cache/vllm/**/fxgraph/下的可重建对象至容器/dev/shm（不占本地磁盘），tar逐字节compare后才删除原缓存对象。禁止.py/cubin/so/IR/JSON/source；新925模型源无fxgraph条目，925源和旧r3严格cache384文件、新LM16cache文件前后SHA必须一致。原数据、模型、源码、生成源和已合格cache不动。完成后SIGCONT外层Bash才允许启动candidate。此操作不修改GPU数值策略/冻结控制源，不发生在任何计时轮中。
