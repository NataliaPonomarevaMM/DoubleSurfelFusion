//
// Created by wei on 4/11/18.
//

#pragma once
#include "common/macro_utils.h"
#include "core/warp_solver/solver_types.h"
#include "core/warp_solver/PenaltyConstants.h"
#include "core/warp_solver/Node2TermsIndex.h"
#include "pcg_solver/ApplySpMVBase.h"
#include <Eigen/Core>
#include <memory>

namespace surfelwarp {
	
	class ApplyJtJHandlerMatrixFree : public ApplySpMVBase<6> {
	private:
		Term2JacobianMaps m_term2jacobian_map;
		
		//The map from node to terms
		using Node2TermMap = Node2TermsIndex::Node2TermMap;
		Node2TermMap m_node2term_map;
		
		//The penalty constants
		PenaltyConstants m_penalty_constants;
	
	public:
		using Ptr = std::shared_ptr<ApplyJtJHandlerMatrixFree>;
		SURFELWARP_DEFAULT_CONSTRUCT_DESTRUCT(ApplyJtJHandlerMatrixFree);
		SURFELWARP_NO_COPY_ASSIGN(ApplyJtJHandlerMatrixFree);
		
		//Explicit allocation, release and input
		void AllocateBuffer();
		void ReleaseBuffer();
		
		void SetInputs(
			Node2TermMap node2term,
			DenseDepthTerm2Jacobian dense_depth_term,
			NodeGraphSmoothTerm2Jacobian smooth_term,
			DensityMapTerm2Jacobian density_map_term = DensityMapTerm2Jacobian(),
			ForegroundMaskTerm2Jacobian foreground_mask_term = ForegroundMaskTerm2Jacobian(),
			Point2PointICPTerm2Jacobian sparse_feature_term = Point2PointICPTerm2Jacobian(),
			PenaltyConstants constants = PenaltyConstants()
		);
		
		//The processing interface
		void ApplyJtJ(DeviceArrayView<float> x, DeviceArraySlice<float> jtj_dot_x, cudaStream_t stream = 0);
		
		//The inherted interface
		size_t MatrixSize() const override { return 6 * (m_node2term_map.offset.Size() - 1); }
		void ApplySpMV(DeviceArrayView<float> x, DeviceArraySlice<float> spmv_x, cudaStream_t stream) override;

		/* Compute JtJ using node2term index or directly using atomic operation
		 */
	public:
		void ApplyJtJIndexed(DeviceArrayView<float> x, DeviceArraySlice<float> jtj_dot_x, cudaStream_t stream = 0);
		void ApplyJtJAtomic(DeviceArrayView<float> x, DeviceArraySlice<float> jtj_dot_x, cudaStream_t stream = 0);
		

		/* First apply Jacobian, then apply Jt
		 */
	private:
		const static unsigned kMaxNumScalarResidualTerms;
		DeviceBufferArray<float> m_jacobian_dot_x;
		void applyJacobianDot(DeviceArrayView<float> x, cudaStream_t stream = 0);
		void applyJacobianTranposeDot(DeviceArraySlice<float> jtj_dot_x, cudaStream_t stream);
	public:
		void ApplyJtJSeparate(DeviceArrayView<float> x, DeviceArraySlice<float> jtj_dot_x, cudaStream_t stream = 0);
	};
	
}