# 天数 BI-V150 W4 Down64 + LM1..64，r4

正式4K：**6050.94 total tok/s / 1210.19 output tok/s，TTFT3076.66ms**。原始4096输入/1024输出/C64/N256，4轮排除首轮；有效三轮6044.63、6055.14、6053.05，极差/均值0.17369%。1024请求全部成功。相比上版r3正式5950.75，吞吐+1.68365%、TTFT-3.08450%；这是历史正式比较，非同场完整AB。相对官方2028.01基线吞吐约2.984倍（+198.37%）。同场N64短测的实际净增为0.63703%，须区别于正式比较。

新鲜完整Math-500 Level3：C8和C64各100/105（95.238095%），均>95%，合计3次max_tokens全部计入105分母。原始checkpoint和结构不变，没有训练或微调。仅Down M64、LM M1..64改为group128对称RTN原生W4；LM1/2直算，其余补齐32/64。其他M、QKV/O64 BF16、GU/KV/attention/scheduler/sampling和75点预热等r3内容保留。原BF16/W8副本继续保留，新增W4权重与scratch计入真实原生memory profile，不宣称净显存节省。

审计事实：C8、短测候选、正式运行的原始result均保留audit_failed，分别因旧NT8捕获检查、旧LM8 BF16调用要求、正式C64检查错误默认取quality_batch=8。三者均单独CPU复核通过且GPU_rerun=false；C64 Math原始运行正常complete。正式CPU复核保留M64>=100门槛（实际15384次），显式M8>=1（实际8次），未删减其他运行或性能要求。完整源/快照、所有105x2评审、四轮CSV、原始checkpoint、真实Graph回放与W4/内存检查、零抢占和就绪后零autotune均独立重算通过。正式脚本无逐请求trace，仅对其实际提供的聚合CSV/时延汇总核验；短测另核验全部384条请求记录。

只在当前已准备好的天数容器运行，所有模型、W4打包副本、依赖、固定算子缓存和原始评测保存在容器；GitHub保存代码、报告、关键JSON/CSV及SHA，不上传模型或大数组。沿用r3的生成源恢复工具，先检查全部944模型输入和W4副本，再直接执行原13参数命令：

```bash
cd /workspace/vllm-plugin-FL
bash experiments/tianshu_native_int8_attention_prefill_release_20261022/restore_generated_sources.sh
bash experiments/tianshu_native_int8_attention_prefill_release_20261022/restore_frozen_cache.sh --check
sha256sum --check experiments/tianshu_W4_Down64_LM1to64_release_20261023/MODEL_SOURCES.sha256
sha256sum --check experiments/tianshu_W4_Down64_LM1to64_release_20261023/W4_PAYLOADS.sha256
source experiments/tianshu_W4_Down64_LM1to64_release_20261023/environment.sh
vllm serve /workspace/MiniCPM5-2B \
  --port 9031 \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
  --served-model-name minicpm \
  --gpu-memory-utilization 0.85 \
  --max-model-len 131072
```

environment.sh仅把原63项环境中的记录路径重定向到新的时间戳目录，其余值不变；可在source时传入绝对输出目录。GPU_CONTROLLER_SOURCES.sha256记录实际133项GPU控制器来源；独立CPU适配及三次复核来源另由各门禁精确绑定。源码、944输入和W4副本SHA、384固定缓存及shell导出的63项原配置已在CPU上检查，无额外GPU重跑。

启动前于20:39:36归档校验后清理172个旧FX缓存文件，归档保存在容器/dev/shm/tianshu-old-fx-20261023.QIaOdf；正式20:42开始计时，此后本代理未做代码准备、推送、哈希或缓存整理。模型/原始证据/固定384项算子缓存未动。

本版没有新16K成绩，不作完整双项比赛资格声明。旧r3标签保留。
