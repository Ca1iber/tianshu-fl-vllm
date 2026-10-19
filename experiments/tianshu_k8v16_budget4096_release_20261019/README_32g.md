# BI-V150 32G：原命令K8V16与框架默认预算4096

此配置已通过全新Math102/105及正式4K：5101.45 total tok/s、1020.29 output tok/s、TTFT4164.20ms；详见OFFICIAL4K_REPORT_32g.md。官方serve argv不变，原生框架默认max_num_batched_tokens从2048选为4096，max_num_seqs=256、async=True；显式用户值优先。75点预热随4096预算扩展到83点。

模板记录205份源文件和43项进程环境，启动前逐一核验；其中包含原容器安装文件和历史生成源码，尚未验证全新克隆或独立提交包复现。不得删改这些文件绕过校验。

CPU干跑使用TIANSHU_K8V16_PROFILE_DRY_RUN=1，不启动GPU；真实启动执行本目录serve_tianshu_minicpm.sh，端口9031。记录目录可通过TIANSHU_K8V16_PROFILE_RECORD_DIR设置为一个尚不存在的目录。该脚本清除继承的实验开关，再设置模板的43项参数。运行时请先检查GPU和端口空闲。
