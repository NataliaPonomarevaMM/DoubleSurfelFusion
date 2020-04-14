#include "core/warp_solver/WarpSolver.h"

void surfelwarp::WarpSolver::initSolverStream() {
	//Create the stream
	cudaSafeCall(cudaStreamCreate(&m_solver_stream[0]));
	cudaSafeCall(cudaStreamCreate(&m_solver_stream[1]));
	cudaSafeCall(cudaStreamCreate(&m_solver_stream[2]));
	cudaSafeCall(cudaStreamCreate(&m_solver_stream[3]));
	
	//Hand in the stream to pcg solver
    m_pcg_solver->UpdateCudaStream(m_solver_stream[0]);
}

void surfelwarp::WarpSolver::releaseSolverStream() {
	//Update 0 stream to pcg solver
    m_pcg_solver->UpdateCudaStream(0);

	cudaSafeCall(cudaStreamDestroy(m_solver_stream[0]));
	cudaSafeCall(cudaStreamDestroy(m_solver_stream[1]));
	cudaSafeCall(cudaStreamDestroy(m_solver_stream[2]));
	cudaSafeCall(cudaStreamDestroy(m_solver_stream[3]));

	//Assign to null stream
	m_solver_stream[0] = 0;
	m_solver_stream[1] = 0;
	m_solver_stream[2] = 0;
}

void surfelwarp::WarpSolver::syncAllSolverStream() {
	cudaSafeCall(cudaStreamSynchronize(m_solver_stream[0]));
	cudaSafeCall(cudaStreamSynchronize(m_solver_stream[1]));
	cudaSafeCall(cudaStreamSynchronize(m_solver_stream[2]));
	cudaSafeCall(cudaStreamSynchronize(m_solver_stream[3]));
}

void surfelwarp::WarpSolver::SolveStreamed() {
	//Sync before compuation
	syncAllSolverStream();

	//Actual computation
	buildSolverIndexStreamed();
	for(auto i = 0; i < Constants::kNumGaussNewtonIterations; i++) {
		if (m_iteration_data.IsGlobalIteration())
			solverIterationGlobalIterationStreamed();
		else
			solverIterationLocalIterationStreamed();
	}

	//Sync again for debug
	syncAllSolverStream();
}

void surfelwarp::WarpSolver::buildSolverIndexStreamed() {
	QueryPixelKNN(m_solver_stream[0]); //Sync is required here
	cudaSafeCall(cudaStreamSynchronize(m_solver_stream[0]));

	//FetchPotentialDenseImageTermPixelsFixedIndex
	m_image_knn_fetcher->SetInputs(m_knn_map, m_rendered_maps.index_map);
	m_image_knn_fetcher->MarkPotentialMatchedPixels(m_solver_stream[0]);
	m_image_knn_fetcher->CompactPotentialValidPixels(m_solver_stream[0]);

	//FindPotentialForegroundMaskPixelSynced();
	setDensityForegroundHandlerFullInput();
	m_density_foreground_handler->MarkValidColorForegroundMaskPixels(m_solver_stream[1]);
	m_density_foreground_handler->CompactValidMaskPixel(m_solver_stream[1]);

	SetSparseFeatureHandlerFullInput();
	m_sparse_correspondence_handler->ChooseValidPixelPairs(m_solver_stream[2]);
	m_sparse_correspondence_handler->CompactQueryPixelPairs(m_solver_stream[2]);

	//The sync group
	m_image_knn_fetcher->SyncQueryCompactedPotentialPixelSize(m_solver_stream[0]); //Sync is inside the method
	m_density_foreground_handler->QueryCompactedMaskPixelArraySize(m_solver_stream[1]); //Sync is inside the method
	m_sparse_correspondence_handler->QueryCompactedArraySize(m_solver_stream[2]); //Sync is inside the method
    computeBindTermNode2Jacobian(m_solver_stream[3]);
    cudaSafeCall(cudaStreamSynchronize(m_solver_stream[3]));

	setDenseDepthHandlerFullInput();
	setDensityForegroundHandlerFullInput();

	//Before the index part: A sync happened here
	SetNode2TermIndexInput();
    m_node2term_index->BuildIndex(m_solver_stream[0]); //This doesnt block
	BuildNodePair2TermIndexBlocked(m_solver_stream[1]); //This will block
	cudaSafeCall(cudaStreamSynchronize(m_solver_stream[0]));
}


void surfelwarp::WarpSolver::solverIterationGlobalIterationStreamed() {
	//Hand in the new SE3 to handlers
	m_dense_depth_handler->UpdateNodeSE3(m_iteration_data.CurrentWarpFieldInput());
	m_density_foreground_handler->UpdateNodeSE3(m_iteration_data.CurrentWarpFieldInput());
	m_sparse_correspondence_handler->UpdateNodeSE3(m_iteration_data.CurrentWarpFieldInput());

	//The computation of jacobian
	ComputeTermJacobianFixedIndex(m_solver_stream[0], m_solver_stream[1], m_solver_stream[2], m_solver_stream[3]); //A sync should happened here
	cudaSafeCall(cudaStreamSynchronize(m_solver_stream[0]));
	cudaSafeCall(cudaStreamSynchronize(m_solver_stream[1]));
	cudaSafeCall(cudaStreamSynchronize(m_solver_stream[2]));
	cudaSafeCall(cudaStreamSynchronize(m_solver_stream[3]));

	//The computation of diagonal blks JtJ and JtError
	SetPreconditionerBuilderAndJtJApplierInput();
    m_preconditioner_rhs_builder->ComputeDiagonalPreconditionerGlobalIteration(m_solver_stream[0]);
    m_preconditioner_rhs_builder->ComputeJtResidualGlobalIteration(m_solver_stream[1]);
    m_jtj_materializer->BuildMaterializedJtJNondiagonalBlocksGlobalIteration(m_solver_stream[2]);
	cudaSafeCall(cudaStreamSynchronize(m_solver_stream[0]));
	cudaSafeCall(cudaStreamSynchronize(m_solver_stream[1]));
	cudaSafeCall(cudaStreamSynchronize(m_solver_stream[2]));

	//The assemble of matrix: a sync here
	const auto diagonal_blks = m_preconditioner_rhs_builder->JtJDiagonalBlocks();
	m_jtj_materializer->AssembleBinBlockCSR(diagonal_blks, m_solver_stream[0]);

	//Solve it and update
	SolvePCGMaterialized();
	m_iteration_data.ApplyWarpFieldUpdate(m_solver_stream[0]);
	cudaSafeCall(cudaStreamSynchronize(m_solver_stream[0]));
}

void surfelwarp::WarpSolver::solverIterationLocalIterationStreamed() {
	//Hand in the new SE3 to handlers
	m_dense_depth_handler->UpdateNodeSE3(m_iteration_data.CurrentWarpFieldInput());
	m_density_foreground_handler->UpdateNodeSE3(m_iteration_data.CurrentWarpFieldInput());
	m_sparse_correspondence_handler->UpdateNodeSE3(m_iteration_data.CurrentWarpFieldInput());

	//The computation of jacobian
	ComputeTermJacobianFixedIndex(m_solver_stream[0], m_solver_stream[1], m_solver_stream[2], m_solver_stream[3]); // A sync should happend here
	cudaSafeCall(cudaStreamSynchronize(m_solver_stream[0]));
	cudaSafeCall(cudaStreamSynchronize(m_solver_stream[1]));
	cudaSafeCall(cudaStreamSynchronize(m_solver_stream[2]));
	cudaSafeCall(cudaStreamSynchronize(m_solver_stream[3]));

	//The computation of diagonal blks JtJ and JtError
	SetPreconditionerBuilderAndJtJApplierInput();
    m_preconditioner_rhs_builder->ComputeDiagonalPreconditioner(m_solver_stream[0]);
    m_preconditioner_rhs_builder->ComputeJtResidual(m_solver_stream[1]);
    m_jtj_materializer->BuildMaterializedJtJNondiagonalBlocks(m_solver_stream[2]);
	cudaSafeCall(cudaStreamSynchronize(m_solver_stream[0]));
	cudaSafeCall(cudaStreamSynchronize(m_solver_stream[1]));
	cudaSafeCall(cudaStreamSynchronize(m_solver_stream[2]));

	//The assemble of matrix: a sync here
	const auto diagonal_blks = m_preconditioner_rhs_builder->JtJDiagonalBlocks();
	m_jtj_materializer->AssembleBinBlockCSR(diagonal_blks, m_solver_stream[0]);

	//Solve it and update
	SolvePCGMaterialized();
	m_iteration_data.ApplyWarpFieldUpdate(m_solver_stream[0]);
	cudaSafeCall(cudaStreamSynchronize(m_solver_stream[0]));
}