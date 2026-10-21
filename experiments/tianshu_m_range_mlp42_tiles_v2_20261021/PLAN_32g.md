# 按M选tile的42-bank确认v2：实际CPU入口预检查

原tiles入口仅CPU读selection顺序错误，GPU尚未import/root未建；原failed记录保留。v2先读selection再核对hash，新增--preflight-only实际走全部前置代码，assert torch/flag_gems未import且GPU root未建。CPU实际预检通过方可GPU；不是只AST语法检查。逐M配置与selected_plan字节完全不变、24M/42层/三动态Graph-eager/整链poison/阈值/真实权重前后hash/锚点binary匹配等科学部分均未改变。1小时或23:58北京硬停，源GPU前冻结。
