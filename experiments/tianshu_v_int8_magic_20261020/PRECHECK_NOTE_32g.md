# GPU gate 预检查修正

首个尝试在运行 CUDA kernel 前终止：新增对父 codec.py 的冻结后，按 basename 查找当前 codec.py 会同时匹配新旧两个文件，len(matches)==1 失败。原日志和修正前 GPU probe 留 /workspace/benchmark-results/20261020_v_int8_magic_codecs_tianshu/micro 与 launch.log。

修正仅将 basename 匹配改成当前脚本目录中的完整绝对路径匹配，仍要求唯一匹配且 SHA256 相同；父 codec 的 hash 保留在冻结输入中。未改变编码、解码、CPU 数值、GPU 码点、写入或 Graph 断言。重试使用新输出 micro_v2 和独立日志，不能把首次失败记录当成 GPU 成功。
