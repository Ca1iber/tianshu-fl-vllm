# BI-V150 32G：线性 INT8 V 候选，源码准备阶段

## 已有基线与当前状态

正式最佳仍是 K8V16/default4096：4K 5101.45 total tok/s、1020.29 output tok/s、Mean TTFT 4164.20ms；同源码 Math Level3 102/105。最佳标签为 tianshu-original-command-4k-k8v16-budget4096-20261019，提交 5ea64d33e594fc9510f5a3062bbf84811161ba5f。以上均是旧版本已验收结果，不能作为本候选的验证结果。

2026-10-20 用户已确认容器运行且网页 IDE 可用，但指定 SSH 网关仍返回 Permission denied(publickey)。本轮尚无 GPU 测试、服务成绩、Math 或 profile 数据。GitHub CPU 检查已通过，见下方记录。已将准备代码保存到独立分支 exp/tianshu-v-int8-20261020，不修改当前最佳代码、标签及沐曦仓库，不建立本地项目副本。

## 本轮假设

保留 K 的已验收软件 8 位编码，将 V 改为线性有符号 INT8，UINT8 存储载荷使用二进制补码。固定缩放候选 1/32、1/16、1/8，不按题目或评测阶段选择缩放，不修改模型权重、采样设置或 benchmark 统计。

真实物理页目标为 12288→8192 字节，减少三分之一页存储；这不是吞吐预测。此前软件 FP8 同时压缩 K/V 更慢，本轮需比较实际线性 V 解码成本。

沐曦 10af519dc5c0 主要由排序回退切回原生 TopK/TopP；天数最好版本已有原生 TopP 和严格几何搜索，因此不把重复迁移当作新收益。

## 已准备的代码

- codec.py：独立标量格式规范；在现有 SHA 锁定的软件 8 位 reader 上只替换 V 的类型转换调用，保留 K helper、attention 数学与 reducer；在 SHA 锁定的已测 writer 中仅替换 UINT8 V 编码。生成模块不安装引擎 hook。
- V 写入采用 nearest/ties-to-even 后饱和到 [-128,127]；NaN 编码为 0、负/正无穷编码为 128/127。读取采用有符号扩展和常量缩放。K 的编码、负 slot 处理和地址计算不变。
- check_cpu.py：覆盖全部 65536 种 BF16 位模式、每档全部 256 个码点、每档 783 个舍入/饱和边界，使用独立 Fraction 有理数参考；校验生成源码的可逆性和 AST 保护。检查只覆盖 CPU 数值约定与源码变换，不证明 Triton 编译或 GPU 正确性。

在容器恢复后，先运行：

```bash
cd /workspace/vllm-plugin-FL
python3 experiments/tianshu_v_int8_20261020/check_cpu.py \
  --reader-source /workspace/benchmark-results/20261019_k8v16_budget_default_official4k_tianshu/decode_coherent_records/cache_generated/97987/generated_cache_policy.py
```

该冻结 reader 也存在于已发布最好版本的关键结果归档中。路径丢失时先恢复并核对摘要，不放松源码断言。

## 尚未实现与必须完成的门槛

1. 运行 CPU 检查，实际编译/运行新增 reader 与 writer；对照独立参考检查 GPU 码点、舍入/饱和、负 slot、packed offset、跨页 stride、尾部保护。禁止把源码检查当作 GPU 通过。
2. 本轮尚未接入服务。必须完成真实 8192 字节页布局、decode 路由、dense prefill 及容量 fallback 的 V 解码；不得沿用旧的 BF16 V 别名或偷偷保留完整 BF16 V 副本。保留有界 scratch，在内存 profile 内计入，capture 前分配。
3. 同一 CUDA Graph 改变 Q/K/V、block table 和序列长度后重放复查。组件对照必须使用同进程已验收 K8V16，覆盖 B1/8/64、4K/5K、ragged/mixed/prefill；独立正确性检查后交替计时，保存原始 event、编译参数、饱和比例和误差。
4. 组件有真实收益才筛选 C64/N64 的 4K/1K 服务；首轮单列并逐轮报告。稳态至少提升 2% 才晋级新 Math；正确率严格大于 95% 才运行原脚本 N256 正式 4K。不能借用旧 Math，不常规跑 16K。
5. 本分支在全部门槛完成前 adoption_allowed=false，不改变最佳启动器。

## 保存与运行边界

代码、报告和关键结果保存天数 GitHub；原始实验数据留 /workspace/benchmark-results；不上传完整原始历史和编译缓存，不修改历史测量源码。预约为北京时间 2026-10-20 06:00 至 2026-10-21 00:00，GPU 硬停止时间为 23:58。

## 已执行的 GitHub CPU 检查

检查运行：https://github.com/Ca1iber/tianshu-fl-vllm/actions/runs/39352140421 ，成功结束。执行提交 a10b55f2c896e9f2b5652ddcce489612891ca9d7；机器为 GitHub Ubuntu 24.04 CPU 运行器，不是 BI-V150 容器。原始 JSON 摘自该作业日志，保存在 CPU_GATE_GITHUB_32g.json；文件名表示目标卡，不能当作该卡 GPU 验证。

三档缩放全部通过：共 196608 个 BF16 位模式检查、768 个解码码点和 2349 个舍入边界；包含 NaN/Inf 明确定义的编码。独立 Fraction 参考、所有码点往返、生成 Python AST、K helper / attention / reducer 保护及可逆源码变换均通过。冻结 reader SHA256 为 8deb8967fdacf354f3b38e6a9687163493f6163764f3e8663dce916250cd1289，输入发布归档 SHA256 为 67f6faa0afe2658573ab7273dc665d8460523d2bb76f751aefc47273501e8207。

这不证明 GPU encoder 与标量规范一致，不证明 Triton 能编译，不包含物理页布局、attention 数值、Graph 重放、饱和率、吞吐或 Math 结论。adoption_allowed 仍为 false；正式最佳 5101.45 total tok/s 不变。下一步必须等 SSH 接入恢复，在天数卡执行 GPU 检查与接入门槛。

## GPU 编解码/写入 gate：程序已准备，尚未运行

probe_codecs_gpu.py 只用于真实卡上的编解码与 KV writer 验证，不安装服务 hook，不运行 attention forward。它核对已通过 CPU 检查的源码摘要；逐档编译并检查全部 BF16 K/V 编码、K/V 全部码点、FP32 ties/saturation 边界、尾部 guard，以及真实 UINT8 页、带偏移/填充/跨页 stride 的完整原始 storage。writer Graph 的三个阶段改变 K、V 和 slot，保存全部存储及原始输入，检查负 slot 与空洞保护。该 Graph 检查不覆盖 Q、block table 或 attention。

预定在容器空闲时运行：

```bash
cd /workspace/vllm-plugin-FL
export VLLM_PLUGINS=fl
python3 experiments/tianshu_v_int8_20261020/probe_codecs_gpu.py \
  --output /workspace/benchmark-results/20261020_v_int8_codecs_tianshu/micro \
  --reader-source /workspace/benchmark-results/20261019_k8v16_budget_default_official4k_tianshu/decode_coherent_records/cache_generated/97987/generated_cache_policy.py \
  --stop-at-utc 2026-10-20T15:58:00+00:00
```

程序只允许 /workspace/benchmark-results 下的新输出目录，确认空闲 BI-V150，设置最多 30 分钟且不越过指定截止时间的进程闹钟。未执行时没有 GPU 正确性结论。即便本 gate 通过，也仍须实现/验证 8192 字节 spec 接入、dense prefill/capacity fallback、attention 数值与 Graph、组件性能、服务及 Math 门槛。

## 后续检查与接入阻塞

新增 GPU 检查程序的 Python 语法及原 CPU 数值检查已在提交 7a1f21637512c1d4abc1b75ffe85903a741ee631 通过，运行 https://github.com/Ca1iber/tianshu-fl-vllm/actions/runs/39352795896 ，作业 118400654165 的所有步骤 success。此运行只 py_compile 三个脚本并执行 check_cpu.py，没有导入或执行 GPU 检查程序；不能把它当作 Triton/GPU 通过。

截至北京时间 2026-10-20 09:20，指定连接连续多轮在 Go SSH 网关的公钥询问阶段失败。显式选择本地 id_ed25519 后，实际公钥指纹 SHA256:IeJn0Xi8fTRgzVQZFCl2SpJHMKUFrIif1ys9sEh9B9M 与用户提供的平台公钥一致；服务器指纹仍是已知 SHA256:DZ7AQ1tcXbaYcLkuPByw0qs+GhV5WKHyr9ghLqPil9A。日志未出现 Server accepts key，最终 Permission denied(publickey)。这只证明该次网关认证未通过，不证明容器 sshd 的具体状态，也不能据此替换 proxy-authorized-keys。

用户确认网页 IDE 正常，正在等待 hostname、pgrep -a sshd 和两个 authorized_keys 文件指纹的当前输出。没有开启新的 GPU 进程。当前无进行中的 CI 作业；独立准备工作完成，下一步须先恢复 SSH，再在真实卡执行已准备 gate，根据 GPU 结果决定是否继续布局和服务接入。当前最佳分支仍是 82b547743e125a506f881d58e1a1a0bb6e7f1e7c，最佳标签与成绩不变。
