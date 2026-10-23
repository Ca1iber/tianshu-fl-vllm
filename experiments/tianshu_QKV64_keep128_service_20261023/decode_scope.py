# SPDX-License-Identifier: Apache-2.0
def eligible_m64(mode_name,num_tokens,num_reqs,uniform,has_lora):
 return mode_name=='FULL' and num_tokens==64 and num_reqs==64 and uniform is True and has_lora is False
