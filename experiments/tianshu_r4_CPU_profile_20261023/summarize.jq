def grouped($rows):
 $rows|group_by([.file,.line,.name])|map({file:.[0].file,line:.[0].line,name:.[0].name,
 mean_local_ms:(map(.local_wall_seconds)|add*1000/16),mean_cumulative_ms:(map(.cumulative_wall_seconds)|add*1000/16),total_calls:(map(.total_calls)|add)})|sort_by(-.mean_local_ms)|.[0:30];
{
 complete:($result[0].phase=="complete"),intrusive_diagnostic_only:true,official_benchmark:false,
 raw_root:$result[0].output_root,source_count:($result[0].frozen_sources|length),actual_environment_count:($result[0].profile_environment|length),
 model_runtime_sources_before_additional_observed_stack_sources:364,
 summary:$result[0].summary,runtime_audit:$result[0].runtime_audit,
 actual_bound_schedule_module:$s[0].actual_bound_schedule_module,actual_bound_schedule_name:$s[0].actual_bound_schedule_name,
 execute_top_local:grouped([ $c[0].frames[].execute_model.rows[] ]),
 sample_top_local:grouped([ $c[0].frames[].sample_tokens.rows[] ]),
 scheduler_top_local:grouped([ $s[0].frames[].rows[] ]),
 caveat:"Host cProfile only, its instrumentation inflates calls. Cumulative rows overlap. Separate processes/windows; not GPU kernel time or benchmark score."
}
