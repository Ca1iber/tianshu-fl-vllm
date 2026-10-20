# index_adjacent配置构造缓存：单项服务对照

来自当前版本CPU诊断：每步索引候选重新构造17个Config。native load_all只预装vendor YAML键，备用index_adjacent配置从fallback取得却未进loaded_triton_config，随后每次get_tuned_config都重新deepcopy默认字典。

候选在原capture返回后，仅把原生构造出来的同一列表存入原有loaded_triton_config['index_adjacent']；对照不存。没有修改任何配置值、autotune算法、内核源码、RNG或模型计算。两组均观察实际lookup/cache miss，记录稀疏powers-of-two调用快照；对照应每次miss，候选应全部hit，真实配置签名与CPU证明逐项一致。

组件已用实际原生选择/生成函数验证16种索引shape/dtype，生成内核文本和选中配置完全一致；7轮交替×200次CPU lookup约108.407→0.197微秒，属于CPU局部，不是模型吞吐。首次两次CPU夹具失败及原件SHA保留在组件目录，最终v3通过。

两组继承合格LM/KV/profile，原13项serve参数、62项环境；不接入新采样预热，不降低日志级别，保持其它变量。4K/1K C64 N64，各3轮首轮剔除，若有稳定服务收益再独立质量和正式4K；不例行完整16K。所有GPU在2026-10-21 23:58北京时间前停。
