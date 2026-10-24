# r5+BM8真实启动通过，新C8质量运行中

2026-10-24约16:48，北京时间。新API实际安装BM8且capture_completed，42层真实uniform64 Graph捕获全部切到资格内核5cc055...，actual_grid=[64,2,4]、BLOCK_M=8、BLOCK_Q=1、TILE_SIZE=16、segments=4；167寄存器、0 spill、6144 shared，warp/stage=1，严格非debug机器码签名匹配旧组件资格。STARTUP_OBSERVED_32g.json是本次新服务记录的精简启动快照，不是旧实验的标签。

原r5全部997源、84个fitted QKV/O权重bank和原有Down/LM/GU/KV/显存/采样审计由新控制器重新核验通过后，已实际进入C8完整105题评测，evaluator PID34055。只有启动捕获证明；C8期间不声称真实C64请求重放，更不声称新正确率或吞吐。后续须全题计入、完整结果>95%，再独立新API验证C64及>=100真实Graph重放。

工作盘启动前124MiB、启动后14MiB。为避免C64启动时无空间，只向已核对身份的Bash父启动器PID33228发SIGSTOP；C8控制器33233、API33235/Engine33463和evaluator34055继续运行。启动器不是GPU计算进程。C8结束并经其控制器正常清理GPU后，先备份、逐文件校验并只回收可重建FX缓存，保留384固定二进制缓存、生成Python模块、所有模型/原始档案和r5不可变SDK副本，再SIGCONT同一个33228让其收集C8证据并启动C64。不得另起重复C8、不得仅因轮询超时视为进程终止。

新鲜输入4K N128对照协议已单独提交f619925，尚未启动性能。正式r5版本与6123.31 total tok/s不变。
