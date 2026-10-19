# K8/V16真实4K筛选：+17.2446%，等待新质量/正式

4K/1K、C64/N64、3+3各排首轮，双UInt8控制稳态4167.760031→K8/V16候选4886.475495 total tok/s，+17.244646%。不是N256正式分数，不能替换正式最佳4259.87。新Math105已启动，不复用旧102/105。

|模式|轮次|Total tok/s|Mean TTFT ms|新tune|
|---|---:|---:|---:|---:|
|control|1|3963.249948|16475.36|0|
|control|2|4169.158400|12799.95|0|
|control|3|4166.361661|12814.23|0|
|candidate|1|4476.066894|17814.62|0|
|candidate|2|4888.745391|12459.82|0|
|candidate|3|4884.205599|12478.07|0|

384/384性能请求、4个within-mode seeded canary、16份零抢占metrics、194冻结源/生成源、实际41环境及原serve argv全本地独立核验；6轮零新autotune。输入每轮改变seed，两模式同轮完全配对。模型BF16、原RNG/strict TopP32Kgeo、M2048三投影、2048budget/256seq/asyncTrue/75warm都保持。

唯一开关TIANSHU_K8V16_SERVICE_MODE：control原V2真实双UInt8 engine，candidate真实12KiB K8/V16 engine；42层FullAttentionSpec、物理规划和同storage BF16 V views、profile内512MiB arena、cache容量、实际reader/writer/mixed/native prefill均核验。不是传入一段未执行的模拟KV缓存。

source5b679a70dc017fa4fb5292cfa16c44f4a066dc11；运行期仅新增smoke报告/独立验证脚本同步到0e8，目标194源SHA不变。原件包4bb4c47f41400e7e7991678d397f3cc58444f936034f0521175326143a6ebbb8，31parts/整包核验；raw在service-results/20261019_tianshu/k8v16_engine_service/20261019_k8v16_engine_service_tianshu。

```bash
python3 experiments/tianshu_k8v16_pipeline_20261019/verify_service_downloaded.py --root ../service-results/20261019_tianshu/k8v16_engine_service/20261019_k8v16_engine_service_tianshu
```
