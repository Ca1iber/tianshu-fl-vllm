# 天数32g：当前.cv KV循环显式展开

当前已采用.cv/S8/guarded KV16/warp1配置，比较原.cv及新副本的KV_LOOP_UNROLL1/2/4/8。只添加一个constexpr并将唯一KV循环头从range改为带loop_unroll_factor的tl.range，循环体、整数宽度、tile/warp/stage、grid、reducer及.cv不变，逆替换可逐字恢复实际8deb源。1作为新range写法的控制，不将传hint等同于编译器已实际展开，保存B64生成IR/LLVM与真实nregs/spills以便后验核对。

本机triton.language.core.py3290/3311/3324已只读确认支持loop_unroll_factor并将在GPU实验保存完整源。CPU用真实adapter/原LaunchRouter与假JIT检查五组实际kwargs、producer设置和reducer未改、真实.cv源生成可逆；不是GPU正确性或性能证据。GPU计划17场景，FP32/CPU/变QKV Graph/原输入逐位对照/尾部/9交替全Attention event。当前没有新GPU/服务/Math/正式分数，保持4091.29最佳。
