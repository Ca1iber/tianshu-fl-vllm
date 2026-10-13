# 天数 Prefill 地址位宽探索

8场景全部CPU/FP32参考、逐位一致和Graph通过，但地址32相对64的微测改善仅约0.17–0.44%，不替换已验证服务。见 results/20261013/micro_32g/REPORT_32g.md。零sink分母L0与原浮点计算完全保留，指针仍64。这里只有process-only原型和探针，没有安装源修改。
