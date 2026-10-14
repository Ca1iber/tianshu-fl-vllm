# 天数BI-V150 32g：当前KV16/warp1+.cv地址32未取得收益

以当前已采用.cv/S8/guarded KV16/w1为对照，只改28个stride整数注解和6个整数element offset cast，物理pointer64/浮点AST/tiles/warps/stages/grid不变。分别测producer32+原reducer64和两阶段32，以真实调用的模块记录区分。旧KV32/warp2负结果没有被直接当作本轮结论。

|batch|KV最大长度|producer32耗时变化|两阶段32|
|---:|---:|---:|---:|
|8|4096|-0.2640%|-0.3724%|
|16|4096|+0.0684%|+0.7113%|
|24|4096|+6.7642%|+6.8004%|
|32|4096|+0.5125%|+0.3040%|
|64|4096|+0.0022%|+0.0356%|
|64|5120|+0.0692%|+0.0393%|
|64|65|-0.8940%|-1.2778%|

B64/4K与5K均在约0.07%以内，基本持平；B24（本来回退KV32/warp2）两种地址32仍慢约6.8%。没有接入服务的明确收益，不采用、不新跑Math/正式。候选源afcb由实际.cv源8deb生成，整数类型归一后的完整AST相同，原始来源链与实际双阶段模块已核验。

17场景/34候选全部FP32/选定CPU、原输入逐位对比、改变Q/K/V后的Graph与逐位对比及尾部通过。每候选launch对实际tensor元素offset加2048 stride masked余量检查<2**31，最大记录369874943；本地复核源码实现与实际记录，未将其写成独立保存全部tensor重算。459原始event、实际producer/reducer模块、reducer默认4warp/stage3、源快照、生成源字节重现及3组B64编译产物均已完整下载核验。

归档3629687bytes/14parts SHA256 `f6ae5f71348633bc1c8a647744f0e26774f58ccc0862410cf4e50a0bca625a20`。原件service-results/20261014_tianshu/cv_address32/20261014_cv_address32_tianshu/micro，verify_downloaded.py --root <raw>/micro可复核。当前正式最佳4091.29及其独立Math103保持，安装源与沐曦仓库未改。

下一未测试方向：Decode KV循环显式展开。已只读确认本机triton.language.range支持loop_unroll_factor（core.py3290/3311/3324），当前.cv在406行使用for j in range(loop_lo,loop_hi)。可在新独立副本只为3D使用tl.range(...loop_unroll_factor=1/2/4/8)，保留原逐token计算、.cv、guardedtile/warp/S8/reducer，不等同于此前明确负收益的num_stages2/3软件流水。尚无源码生成/CPU/GPU实测，不宣称收益。
