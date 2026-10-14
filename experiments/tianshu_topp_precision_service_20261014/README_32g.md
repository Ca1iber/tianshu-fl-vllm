# 天数BI-V150 32g：TopP精度修复/16K tile与max-num-seqs64组合服务对照

当前采用.cv/4096预算/83捕获点/S8/KV16/原生Prefill配置作为control，原生TopP 8192/4096/w16/s1，默认max_num_seqs。candidate仅组合修复后的纯TopP 16384/8192/w16/s1与max-num-seqs64；TopK、其他输入保护路径仍原生。采样参数1/.95、模型最大长度131072、请求4K输入/1K输出/C64/N64不变。

前置独立算子10输入×6配置，翻转输入和Graph/严格CPU1e-6验证通过；具体报告见tianshu_topp_precision_20261014。先前maxseq64单独服务只+.6721%，未采用。本次组合收益不归因单项。每边新服务3轮，预排除首轮；达到2%且区间不交叠才独立5+5确认，然后新Math/正式4K。当前配置及正式成绩不因micro采用。

CPU实际嵌套CLI与Loader/组合hook执行检查通过；CPU fixture不是GPU验证。安装源码不修改。冻结源码可完整追溯。
