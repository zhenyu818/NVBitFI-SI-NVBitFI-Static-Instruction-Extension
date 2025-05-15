/*
 * Copyright 2020, NVIDIA CORPORATION.
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *     http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <stdint.h>
#include <stdio.h>

#include "utils/utils.h"
#include "arch.h"

// Global counters are incremented once per warp 
extern "C" __device__ __noinline__ void count_instrs(uint64_t pcounters, int index, int grp_index, int predicate, int num_counters,uint64_t pstatic_inst_threads_count, int static_inst_id, uint64_t pstatic_inst_type) {    
	uint64_t *counters = (uint64_t*)pcounters;
	uint64_t *static_inst_threads_count = (uint64_t *)pstatic_inst_threads_count;
    uint64_t *static_inst_type = (uint64_t *)pstatic_inst_type;

	// Optimization: Instead of all the threads in a warp performing atomicAdd,
	// let's count the number of active threads with predicate=1 in a warp and let just one thread
	// (leader) in the warp perform the atomicAdd
	const int active_mask = __ballot_sync(__activemask(), 1);
	const int leader = __ffs(active_mask) - 1;
	const int laneid = get_laneid();

	// compute the predicate mask
	const int predicate_mask = __ballot_sync(__activemask(), predicate);
	const int num_threads = __popc(predicate_mask);

	if (laneid == leader) { // Am I the leader thread
		atomicAdd((unsigned long long *)&static_inst_threads_count[static_inst_id], num_threads);
        atomicExch((unsigned long long *)&static_inst_type[static_inst_id], grp_index);
		atomicAdd((unsigned long long *)&counters[index], num_threads);
		atomicAdd((unsigned long long *)&counters[NUM_ISA_INSTRUCTIONS+grp_index], num_threads);
		atomicAdd((unsigned long long *)&counters[num_counters-2], num_threads*(grp_index != G_NODEST));
		atomicAdd((unsigned long long *)&counters[num_counters-1], num_threads*(1 - ((grp_index == G_NODEST) || (grp_index == G_PR))));
	}
}
