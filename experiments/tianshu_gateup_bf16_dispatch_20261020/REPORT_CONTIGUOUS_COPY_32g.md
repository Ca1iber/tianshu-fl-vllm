# 显式连续化修正验证

原M565/M1537行stride输入在两条原始路径上存在已保存的数值超差。将a.contiguous()放入真实Graph调用内后，4个case×3阶段×2方法全部通过原容差，独立CPU选择点复算通过，失败列表为空。原始输入位、保护区和RNG不变，未降低门槛。两次诊断完整原始weight SHA均匹配正式100/105 Math实际layer0 GateUp。

复制成本包含在Graph计时。连续M565 mm.541365ms/linear.436136ms，row-stride M565 .559873/.453221ms；连续M1537 1.240679/1.061523ms，row-stride M1537 1.289254/1.100355ms。修正后linear相对mm组件吞吐仍分别高24.13%、23.53%、16.88%、17.17%。尚未接入服务，不能当作新增模型吞吐。

实际site-packages SQL与CPU验证用的仓库sql.py文件已单独SHA核对，均6dc19cb581d70154f14b95ed9112d682c6b5919b6cf3f8dea01a78a75ec0efd7；后续接入会针对实际加载的方法，避免只改未执行的checkout。
