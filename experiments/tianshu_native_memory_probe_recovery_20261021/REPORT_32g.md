# 静态编译原生显存分项观察与 CPU 归档恢复

2026-10-21 BI-V150 32G。本轮只启动 static420 候选加只读 observer，共 426 源/62 环境项；原 serve 13 参数、4096 token 预算与 0.85 显存比例保持不变。没有增加 profile_run、reset_peak 或修改 native KV 返回值。实际 compile_sizes=[8,64]，C8/C64 专用 range 确实进入真实 Graph 捕获。

这次原生 requested=29205777612、weights=6429220352、activation peak=402702336、non-Torch=1352663040、Graph estimate=0 bytes，返回 KV=21021191884、对应40731个K8V16 blocks。调用前后活跃Torch=6448255488/6691412480 bytes。该预算与近期独立 KV V3 control 的原生观测一致；这不是同一轮配对因果实验。

此前 static420 正式小测候选仅40143块、比当时control少588块=303108608 bytes；本次没有复现。不能把静态编译固定地解释成289MiB永久损失，也没有证据通过reset峰值就能安全回收。当前不据此改变显存计算或推荐版本。

8条无计时 canary 均128输入/16输出，finish_reason=length。原controller以完整usage字典等于三字段字典，遇到合法的prompt_tokens_details:null后断言；原失败state和426源全部保留。独立CPU verifier逐个验证完整冻结源码/快照、环境、native公式、真实捕获与8条响应，只恢复归档，未重跑GPU。GPU已回收。本轮没有吞吐成绩、完整C64请求复测或Math精度成绩。

GitHub只保留本报告、CPU复核代码及关键gate；完整响应与日志在容器原始目录。RECOVERY_GATE记录原始文件SHA，未修改原始state。
