# BI-V150 32G：线性 INT8 V 候选，源码准备阶段

## 已有基线与当前状态

正式最佳仍是 K8V16/default4096：4K 5101.45 total tok/s、1020.29 output tok/s、Mean TTFT 4164.20ms；同源码 Math Level3 102/105。最佳标签为 tianshu-original-command-4k-k8v16-budget4096-20261019，提交 5ea64d33e594fc9510f5a3062bbf84811161ba5f。以上均是旧版本已验收结果，不能作为本候选的验证结果。

2026-10-20 用户已确认容器运行且网页 IDE 可用，但指定 SSH 网关仍返回 Permission denied(publickey)。本轮尚无 GPU 测试、CPU 检查执行、服务成绩、Math 或 profile 数据。已将准备代码保存到独立分支 exp/tianshu-v-int8-20261020，不修改当前最佳代码、标签及沐曦仓库，不建立本地项目副本。

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