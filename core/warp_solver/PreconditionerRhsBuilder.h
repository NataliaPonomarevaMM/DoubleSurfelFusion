//
// Created by wei on 4/6/18.
//

#pragma once

#include "common/macro_utils.h"
#include "core/warp_solver/solver_types.h"
#include "core/warp_solver/Node2TermsIndex.h"
#include "core/warp_solver/PenaltyConstants.h"
#include "pcg_solver/BlockDiagonalPreconditionerInverse.h"
#include <memory>

namespace surfelwarp {
	
	class PreconditionerRhsBuilder {
	private:
		//The map from term to jacobian, will also be accessed on device
		Term2JacobianMaps m_term2jacobian_map;
		
		//The map from node to terms
		using Node2TermMap = Node2TermsIndex::Node2TermMap;
		Node2TermMap m_node2term_map;
		
		//The penalty constants
		PenaltyConstants m_penalty_constants;
	public:
		using Ptr = std::shared_ptr<PreconditionerRhsBuilder>;
		SURFELWARP_DEFAULT_CONSTRUCT_DESTRUCT(PreconditionerRhsBuilder);
		SURFELWARP_NO_COPY_ASSIGN(PreconditionerRhsBuilder);
		
		//Explicit allocation, release and input
		void AllocateBuffer();
		void ReleaseBuffer();
		
		void SetInputs(
			Node2TermMap node2term,
			DenseDepthTerm2Jacobian dense_depth_term,
			NodeGraphSmoothTerm2Jacobian smooth_term,
            NodeGraphBindTerm2Jacobian bind_term,
			DensityMapTerm2Jacobian density_map_term = DensityMapTerm2Jacobian(),
			ForegroundMaskTerm2Jacobian foreground_mask_term = ForegroundMaskTerm2Jacobian(),
			Point2PointICPTerm2Jacobian sparse_feature_term = Point2PointICPTerm2Jacobian(),
			PenaltyConstants constants = PenaltyConstants()
		);
		
		//The processing interface
		void ComputeDiagonalPreconditioner(cudaStream_t stream = 0);
		void ComputeDiagonalPreconditionerGlobalIteration(cudaStream_t stream = 0);
		DeviceArrayView<float> InversedPreconditioner() const { return m_preconditioner_inverse_handler->InversedDiagonalBlocks(); }
		DeviceArrayView<float> JtJDiagonalBlocks() const { return m_block_preconditioner.ArrayView(); }
		
		/* The buffer and method to compute the diagonal blocked pre-conditioner
		 */
	private:
		DeviceBufferArray<float> m_block_preconditioner;
		//The actual processing methods
	public:
		void ComputeDiagonalBlocks(cudaStream_t stream = 0);
		

		/* The buffer and method to inverse the preconditioner
		 */
	private:
		BlockDiagonalPreconditionerInverse<6>::Ptr m_preconditioner_inverse_handler;
	public:
		void ComputeDiagonalPreconditionerInverse(cudaStream_t stream = 0);
		/* The buffer and method to compute Jt.dot(Residual)
		 */
	private:
		DeviceBufferArray<float> m_jt_residual;

	public:
		void ComputeJtResidual(cudaStream_t stream = 0);
		void ComputeJtResidualGlobalIteration(cudaStream_t stream = 0);
		DeviceArrayView<float> JtDotResidualValue() const { return m_jt_residual.ArrayView(); }
	};
	
}