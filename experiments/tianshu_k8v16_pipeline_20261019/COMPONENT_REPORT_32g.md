# K8V16完整attention组件21场景通过

来源446c568de3906b2a9f3fa533480d522b07b686dc；直接通过真实TritonAttentionImpl写入/forward、原metadata builder/native prefills与decode router验证新的12KiB物理布局。184冻结源/原生成码、65536实际writer编码与256解码、4组真实native spec/reshape CPU fixture、42变数据Graph相位/1044参考记录、16 dense逐位检查（34765824元素）及294原始event已本地核验。typed V无复制、同data_ptr/同raw storage/所有访问落在已分配字节内。

BF16→K8V16同次计时：prefill2048 .712546→.705374ms、chunk2048 1.842340→1.835162；mixed64 2.893834→2.444940；decode64 .882857→.590170；decode64/5K 1.091447→.729859；B256 6.805228→2.458146。小prefill32/33仍慢约22%/19.5%，ragged慢8.77%、unreordered慢.755%。强制容量fallback .020888→.071181ms，正确但慢约3.4倍；共享arena限制仍保留，不将它隐去。

上列是BF16组件对照，不是新旧KV服务收益；之前同reader/launch的五路decode对照才是byte→K8V16约-28.7%的证据。任何组件均不替代完整模型吞吐和新Math。尚未安装实际ModelRunner，本目录新增engine_hooks等只是CPU准备；当前正式最好4259.87+freshMath102/105不变。

原始包5383591bytes；SHA256 dde409b3c9caade675e52f5d0e369e49bbf4a87f2728011c2288c997deeea015，21parts及整包已校验。

```bash
python3 experiments/tianshu_k8v16_pipeline_20261019/verify_component_downloaded.py --root ../service-results/20261019_tianshu/k8v16_pipeline_component/20261019_k8v16_pipeline_component_tianshu/micro
```
