//
// Created by wei on 4/9/18.
//

#include "common/logging.h"
#include "common/sanity_check.h"
#include "core/warp_solver/solver_constants.h"
#include "core/warp_solver/geometry_icp_jacobian.cuh"
#include "core/warp_solver/PreconditionerRhsBuilder.h"

/* The interface function for JtResidual
 */
void surfelwarp::PreconditionerRhsBuilder::ComputeJtResidual(cudaStream_t stream) {
	ComputeJtResidualIndexed(stream);
	//Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	cudaSafeCall(cudaStreamSynchronize(stream));
	cudaSafeCall(cudaGetLastError());
#endif
}




