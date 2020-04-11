//
// Created by wei on 3/28/18.
//

#pragma once

#include "common/macro_utils.h"
#include "common/common_types.h"
#include "common/CameraObservation.h"
#include "common/ArraySlice.h"
#include "common/surfel_types.h"
#include "core/render/Renderer.h"
#include "core/SurfelGeometry.h"
#include "core/WarpField.h"
#include "core/smpl/smpl.h"
#include "core/warp_solver/SolverIterationData.h"
#include "core/warp_solver/ImageTermKNNFetcher.h"
#include "core/warp_solver/DenseDepthHandler.h"
#include "core/warp_solver/DensityForegroundMapHandler.h"
#include "core/warp_solver/SparseCorrespondenceHandler.h"
#include "core/warp_solver/Node2TermsIndex.h"
#include "core/warp_solver/NodePair2TermsIndex.h"
#include "core/warp_solver/PreconditionerRhsBuilder.h"
#include "core/warp_solver/ResidualEvaluator.h"
#include "core/warp_solver/JtJMaterializer.h"
#include "core/warp_solver/ApplyJtJMatrixFreeHandler.h"
#include "core/warp_solver/NodeGraphSmoothHandler.h"
#include "pcg_solver/BlockPCG.h"
#include "math/DualQuaternion.hpp"
#include <memory>

namespace surfelwarp {
	
	class WarpSolver {
	private:
		//Default parameters
		int m_image_width;
		int m_image_height;
		
		//The inputs to the solver
		CameraObservation m_observation;
		Renderer::SolverMaps m_rendered_maps;
		SurfelGeometry::SolverInput m_geometry_input;
		WarpField::SolverInput m_warpfield_input;
        SMPL::SolverInput m_smpl_input;
		mat34 m_world2camera;
		
		//The interation data maintained by the solver
		SolverIterationData m_iteration_data;
	public:
		using Ptr = std::shared_ptr<WarpSolver>;
		WarpSolver();
		~WarpSolver() = default;
		SURFELWARP_NO_COPY_ASSIGN_MOVE(WarpSolver);
		
		//Matrix-free solver might override these methods;
		void AllocateBuffer();
		void ReleaseBuffer();
		
		//The maps and arrays accessed by solver
		void SetSolverInputs(
			CameraObservation observation,
			Renderer::SolverMaps rendered_maps,
			SurfelGeometry::SolverInput geometry_input,
			WarpField::SolverInput warpfield_input,
            SMPL::SolverInput smpl_input,
			const Matrix4f& world2camera
		);
		void SetSolverInputs(
			CameraObservation observation,
			Renderer::SolverMaps rendered_maps,
			SurfelGeometry::SolverInput geometry_input,
			WarpField::SolverInput warpfield_input,
            SMPL::SolverInput smpl_input,
			const mat34& world2camera
		);
		
		//The access interface
		DeviceArrayView<DualQuaternion> SolvedNodeSE3() const { return m_iteration_data.CurrentWarpFieldInput(); }

    private:
		/* Query the KNN for pixels given index map
		 * The knn map is in the same resolution as image */
		DeviceArray2D<KNNAndWeight> m_knn_map;
		void QueryPixelKNN(cudaStream_t stream = 0);

		/* Fetch the potential valid image term pixels, knn and weight*/
		ImageTermKNNFetcher::Ptr m_image_knn_fetcher;
		
		/* Hand in the geometry maps to
		 * depth correspondence finder
		 * Depends: QueryPixelKNN, FetchPotentialDenseImageTermPixels*/
		DenseDepthHandler::Ptr m_dense_depth_handler;
		void setDenseDepthHandlerFullInput();
	public:
		//The method to compute alignment error after solving
		void ComputeAlignmentErrorMapDirect(cudaStream_t stream = 0);
		void ComputeAlignmentErrorOnNodes(cudaStream_t stream = 0);
		void ComputeAlignmentErrorMapFromNode(cudaStream_t stream = 0);
		cudaTextureObject_t GetAlignmentErrorMap() const { return m_dense_depth_handler->GetAlignmentErrorMap(); }
		NodeAlignmentError GetNodeAlignmentError() const { return m_dense_depth_handler->GetNodeAlignmentError(); }

    private:
		/* Hand in the color and foreground
		 * mask to valid pixel compactor
		 * Depends: QueryPixelKNN*/
		DensityForegroundMapHandler::Ptr m_density_foreground_handler;
		void setDensityForegroundHandlerFullInput();
		
		/* Hand in the vertex maps and pixel pairs
		 * to sparse feature handler*/
		SparseCorrespondenceHandler::Ptr m_sparse_correspondence_handler;
		void SetSparseFeatureHandlerFullInput();

		/* Hand in the value to node graph term handler*/
		NodeGraphSmoothHandler::Ptr m_graph_smooth_handler;
		void computeSmoothTermNode2Jacobian(cudaStream_t stream);

		/* Build the node to term index
		 * Depends: correspond depth, valid pixel, node graph, sparse feature*/
		Node2TermsIndex::Ptr m_node2term_index;
		NodePair2TermsIndex::Ptr m_nodepair2term_index;
		void SetNode2TermIndexInput();
		void BuildNodePair2TermIndexBlocked(cudaStream_t stream = 0);
		
		/* Compute the jacobians for all terms*/
		void ComputeTermJacobiansFreeIndex(
			cudaStream_t dense_depth = 0,
			cudaStream_t density_map = 0,
			cudaStream_t foreground_mask = 0,
			cudaStream_t sparse_feature = 0
		);
		void ComputeTermJacobianFixedIndex(
			cudaStream_t dense_depth = 0,
			cudaStream_t density_map = 0,
			cudaStream_t foreground_mask = 0,
			cudaStream_t sparse_feature = 0
		);
		
		/* Construct the preconditioner and rhs of the method*/
		PreconditionerRhsBuilder::Ptr m_preconditioner_rhs_builder;
        /* Materialize the JtJ matrix*/
        JtJMaterializer::Ptr m_jtj_materializer;
        /* The method to apply JtJ to a vector*/
        ApplyJtJHandlerMatrixFree::Ptr m_apply_jtj_handler;
		void SetPreconditionerBuilderAndJtJApplierInput();

		/* The pcg solver*/
		BlockPCG<6>::Ptr m_pcg_solver;
		void SolvePCGMatrixFree();
		void SolvePCGMaterialized(int pcg_iterations = 10);

		/* The solver interface for streamed solver*/
		cudaStream_t m_solver_stream[4];
		void initSolverStream();
		void releaseSolverStream();
		void syncAllSolverStream();

		void buildSolverIndexStreamed();
		void solverIterationGlobalIterationStreamed();
		void solverIterationLocalIterationStreamed();
	public:
		void SolveStreamed();
	};
}
