//
// Created by wei on 4/11/18.
//

#include "common/sanity_check.h"
#include "core/warp_solver/ApplyJtJMatrixFreeHandler.h"

void surfelwarp::ApplyJtJHandlerMatrixFree::AllocateBuffer() {
	m_jacobian_dot_x.AllocateBuffer(kMaxNumScalarResidualTerms);
}

void surfelwarp::ApplyJtJHandlerMatrixFree::ReleaseBuffer() {
	m_jacobian_dot_x.ReleaseBuffer();
}


void surfelwarp::ApplyJtJHandlerMatrixFree::SetInputs(
	Node2TermMap node2term, 
	DenseDepthTerm2Jacobian dense_depth_term, 
	NodeGraphSmoothTerm2Jacobian smooth_term, 
	DensityMapTerm2Jacobian density_map_term, 
	ForegroundMaskTerm2Jacobian foreground_mask_term, 
	Point2PointICPTerm2Jacobian sparse_feature_term,
	PenaltyConstants constants
) {
	m_node2term_map = node2term;
	
	m_term2jacobian_map.dense_depth_term = dense_depth_term;
	m_term2jacobian_map.smooth_term = smooth_term;
	m_term2jacobian_map.density_map_term = density_map_term;
	m_term2jacobian_map.foreground_mask_term = foreground_mask_term;
	m_term2jacobian_map.sparse_feature_term = sparse_feature_term;
	
	m_penalty_constants = constants;
}

void surfelwarp::ApplyJtJHandlerMatrixFree::ApplySpMV(DeviceArrayView<float> x, DeviceArraySlice<float> spmv_x, cudaStream_t stream) {
	ApplyJtJ(x, spmv_x, stream);
}













