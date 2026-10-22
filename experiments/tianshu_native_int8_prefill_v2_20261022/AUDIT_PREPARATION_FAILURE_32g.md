# 首次CPU缓存准备未通过

原audit_cache_cpu.py仅检查/tmp/torchinductor_root/triton/0，M128符合资格cubin，后续M512/4096不符，assert停止，未启动本V2 GPU，也未覆盖旧cache。保持原脚本。后续独立GPU来源诊断以两个AST相同来源重编译六次，按每M严格资格SHA选择；新audit_cache_cpu_v2.py核对选择及所有cache/group实际依赖。不是放宽binary要求。
