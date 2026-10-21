# 13:38：QKV prefill小测+.293%，未采用；GPU空闲，继续研究

预约10-22 00:00止，GPU23:58北京硬停。推荐仍tag tianshu-original-command-4k-prefill-int8-20261021 /521c154，407源/62env/原13argv，正式4K5586.28total/1117.25out/TTFT3780.63ms，Math102/105。TopP99/105失败不重跑；Down SplitK/grid-only/完整integer-PV拒绝。IntegerPV最终CPU gate在tianshu_k8vi8_pv_full_cpu_recovery_20261021，2184向量/1104整数向量，4K .539900→1.450207ms，理想参考两项失败披露。

QKV V1 loading前routing模块遮蔽失败，保留。V2 tianshu_qkv_prefill_int8_service_v2_20261021已完整六轮384成功与独立audit，control排首后5481.311164，candidate5497.381391，+.293182%，TTFT−1.347920%，spread .028933/.080455%，tune0。实际42层/24warm尺寸，活跃M128/2048/4096，原生load profiler新增220631040B、全部BF16保留，native模型6429220352→6650899968B，KV40133→39586。候选未过事前>.5%门槛，不做Math/N256。每轮成绩已向用户报告。原runner232213终态Z，API233355/core233513退出，13:37:56 ixsmi空闲，FlagGems clean。不要再当GPU live或重跑。

独立exact-reference已92974e7推送，只CPU准备，证明407是432完全相同子集。因共同opaque wrapper不能据委托原方法证明实际BF16 kernel逐位不变，预先要求若QKV有>.5%才直接跑真正未包装407。当前未达门槛，该参考GPU不启动。报告/SCREEN gate/CSV/RUNTIME_BINDING在V2目录；无新Math质量claim。

下一方向尚未决定，不要重复LM NN（已1fab1c7证明负收益）、单纯V8、pair大tile或两个KV heads打包。可检查是否已做FP16 QK/PV分支，先find历史再设计。此前integerPV大tile明显慢；当前实际reader f1df731b，C64/4K S4/T16/w1/176regs/0spill，physicalK8V16 12288B，不是旧S8 reference。

也正在核对元数据复用：只读沐曦5217a6d将K累计长度从逐层tolist+D2H挪到builder一次GPU cumsum。天数iluvatar_hybrid_attention.route_from_common已在builder生成Q路由/归一化cu，所有层共享；这只是Q元数据证据，不能直接当K累计长度已缓存。真实路径pipeline/routing→BoundedDensePrefill→hybrid bf16_dispatch→flag_gems.flash_attn_varlen_func，paged BF16 K/V+seqused_k/block_table；需继续读/usr/local/lib/python3.12/site-packages/flag_gems/ops/attention.py:1284函数，确认paged路径是否根本无需cuK或仍逐层生成。低层flash_api.py不是实际wrapper入口，别再猜文件。所有源只读，尚无该方向GPU试验。

参考bare仓/workspace/reference-repos/metax-latest/flagos-{contest,ops-lib}.git的HEAD master为空，使用refs/remotes/origin/exp/packed-bundle-service-20261007，latest ed3bcfe/8eb57f17。SQL reflect_one已采用。代码/报告/关键结果仅容器+GitHub，无本地副本，不动用户dirty gateup_silu两文件，FlagGems clean4cdc3b7，不改沐曦，无子agent。SSH master/tmp/tianshu-20261021-control，16:27前刷新。

进一步已读FlagGems ops/attention.py:1284..1445：paged接口同时传seqused_k/block_table，cu_seqlens_k明确仅非paged使用；Python分支只创建empty_like(cuQ) dummy传低层，不做K cumsum或tolist/D2H。C-extension分支也直接透传seqused_k。天数不具备沐曦逐层K累计D2H那条问题；至多研究无用dummy分配，预计很小，不可宣称已找到大框架收益。find历史没有fp16/half目录（只有V8 pv32旧试验），若探索FP16 QK/PV先查Git/原生kernels以防重复。
