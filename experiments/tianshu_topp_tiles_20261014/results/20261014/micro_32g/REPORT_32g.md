# 天数BI-V150 32g：native TopP tile和warp32未形成可采用组合

以已采用8192/4096/w16/s1为基线，8配置×7输入共56次尝试，无编译/资源错误。p=.95、top-k、RNG、vocab130560、native64c666源和grid不变，只改tile/warp启动参数。原输入及翻转词表输入与已采用基线精确相同，或各自独立严格CPU mass/minimality/order1e-6通过，才进入计时；未放宽CPU容差。

|输入|w16基线 ms|8192/w32耗时变化|
|---|---:|---:|
|B8 normal k=False|1.152335|+15.6031%|
|B32 normal k=False|2.222963|+13.4586%|
|B64 normal k=False|4.583795|+13.2356%|
|B64 peaked k=False|2.826202|+17.4755%|
|B8 masked k=False|1.152229|+15.4181%|
|B8 ties k=False|0.766970|未通过相对门槛，不计时|
|B32 normal k=True|0.558860|+29.1738%|

仅adopted与identity两配置在全部7输入通过相对门槛；identity耗时差<0.14%。8192/w32正常B64慢13.24%、peaked慢17.48%，ties不通过。16384/w16在masked B8减时27.99%，但其它普通/peaked或翻转ties改变mask且未满足严格CPU门槛，不能凭这一行采用或声称模型提速。暂不接服务、不新Math/正式。

复核还观察到现有native缺陷：baseline普通B64最大概率shortfall约5.80e-5，peaked约5.91e-4；ties部分行保留质量0.99518，最小集合超额约0.04518，而请求p约0.95。完整逐行CPU向量在raw，最坏行摘要另存KNOWN_NATIVE_BASELINE_ERRORS_32g.json。与自己精确相同不是严格CPU语义通过；报告保留baseline误差，不宣称本轮所有配置数学正确。当前已采用模型Math103/105是另一条完整质量证据，本轮没有新模型准确率结论。

全部2600条逐行原始/改变输入CPU门槛记录重新计算，实际计时30组共210event重算，三冻结源/native字节/B64编译产物SHA均完整下载验证。56次尝试不等于56项通过，未通过候选没有性能分数。

归档4268723bytes/17parts SHA256 `fc56b82bd95f453cd73de5c571955bc81e99a1c199c838b240e603c8251833a8`，原件service-results/20261014_tianshu/topp_tiles/20261014_topp_tiles_tianshu/micro。verify_downloaded.py --root <raw>/micro可复核。当前正式最佳4091.29保持。

新源码假设（尚未实现/验证）：_update_min_larger_stats对概率使用abs差<1e-9，把非常接近但不同的概率算作重复；standalone TopP还以概率绝对区间1e-9或18次停止，并用log/exp逆变换回logit后以1e-9判断重复，可能引入边界/重复计数问题。可只为standalone TopP新增exact相等helper、将停止标准改1e-12/32次、最终在概率域判断pivot与重复，保留top-k分支原行为；须按严格CPU重新验证，不把假设写成已定位根因。修复后再测8K/16K tile，不能直接放宽旧门槛。补BF16往返logits场景更接近模型输出。
