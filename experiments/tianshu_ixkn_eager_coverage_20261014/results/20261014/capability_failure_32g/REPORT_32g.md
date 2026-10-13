# 天数 BI-V150 32g：ixkn eager剖析能力失败证据

使用当前ixkn-cli4.4.0、time_duration、已知A/B两个简单Triton kernel，profile-from-start=off，launch-count100避免预热误计。no-start负对照输出正确且工具明确No kernels were profiled，trace0bytes；正阶段应用计算结果也exact，但无法得到可用性能报告。

实际控制器阶段：failed，正阶段工具退出-11，原始审计记录{"complete": false, "coverage_verified": false, "status": "eager_profiler_heap_corruption", "positive_tool_exit_code": -11, "positive_application_exit_code": 134, "positive_report_import_exit_code": 255, "no_start_tool_no_kernels": true, "workload_both_numeric_exact": true, "Graph_not_constructed": true, "therefore_Graph_specific_cause_not_proven": true}。不能把workload complete/numeric_exact当作profiler成功，也不能把预期叶子数当实际捕获数。Graph正阶段预期2eager+2Graph=4，未证实内部覆盖；eager无Graph对照也在退出时堆损坏，因此没有证明失败仅由Graph导致。

原始log、工具版本/help、源2份快照、trace和导入错误全部保留，实际失败phase未伪装complete。全包2533455bytes/10分片，SHA256 b6a9d2307fbcd94bff63d4879016b4355419bd1cf4e6c8ab91f956f0bc192586；整包/分片、源逐字与两阶段原始记录本地核验。原件service-results/20261014_tianshu/ixkn_eager_failure。没有模型吞吐新分数，没有可用Graph内kernel时间占比，不基于此猜测硬件瓶颈。

当前正式最好3640.16/728.03/TTFT4465.17/Math103保持，模型/安装源/benchmark未改。继续通过已验证的整体Graph事件计时、直接算子测试和实际正式请求路径选择优化点。无新16K服务。
