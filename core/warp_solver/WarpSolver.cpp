//
// Created by wei on 3/28/18.
//

#include <core/WarpField.h>
#include "common/ConfigParser.h"
#include "common/CameraObservation.h"
#include "core/warp_solver/WarpSolver.h"
#include "core/warp_solver/solver_constants.h"


/* The constructor only zero-init
 * The caller ensures allocate/release
 */
surfelwarp::WarpSolver::WarpSolver() : m_iteration_data() {
	//Query the image
	const auto config = ConfigParser::Instance();
	m_image_height = config.clip_image_rows();
	m_image_width = config.clip_image_cols();
	
	memset(&m_observation, 0, sizeof(m_observation));
	memset(&m_rendered_maps, 0, sizeof(m_rendered_maps));
	memset(&m_geometry_input, 0, sizeof(m_geometry_input));
}

void surfelwarp::WarpSolver::AllocateBuffer() {
	m_knn_map.create(m_image_height, m_image_width);
	
	//The correspondence depth and albedo pixel pairs
    m_image_knn_fetcher = std::make_shared<ImageTermKNNFetcher>();
    m_dense_depth_handler = std::make_shared<DenseDepthHandler>();
    m_dense_depth_handler->AllocateBuffer();
    m_density_foreground_handler = std::make_shared<DensityForegroundMapHandler>();
    m_density_foreground_handler->AllocateBuffer();
    m_sparse_correspondence_handler = std::make_shared<SparseCorrespondenceHandler>();
    m_sparse_correspondence_handler->AllocateBuffer();
    m_graph_smooth_handler = std::make_shared<NodeGraphSmoothHandler>();
    m_graph_bind_handler = std::make_shared<NodeGraphBindHandler>();
    m_node2term_index = std::make_shared<Node2TermsIndex>();
    m_node2term_index->AllocateBuffer();
    m_nodepair2term_index = std::make_shared<NodePair2TermsIndex>();
	m_nodepair2term_index->AllocateBuffer();
    m_preconditioner_rhs_builder = std::make_shared<PreconditionerRhsBuilder>();
    m_preconditioner_rhs_builder->AllocateBuffer();
    m_jtj_materializer = std::make_shared<JtJMaterializer>();
	m_jtj_materializer->AllocateBuffer();
    const auto max_matrix_size = 6 * Constants::kMaxNumNodes;
    m_pcg_solver = std::make_shared<BlockPCG<6>>(max_matrix_size);
	
	//Init the stream for cuda
	initSolverStream();
}

void surfelwarp::WarpSolver::ReleaseBuffer() {
	m_knn_map.release();

	//Destroy the stream
	releaseSolverStream();
	
	//Release the corresponded buffer
    m_dense_depth_handler->ReleaseBuffer();
    m_density_foreground_handler->ReleaseBuffer();
    m_sparse_correspondence_handler->ReleaseBuffer();
    m_node2term_index->ReleaseBuffer();
    m_nodepair2term_index->ReleaseBuffer();
    m_preconditioner_rhs_builder->ReleaseBuffer();
    m_jtj_materializer->ReleaseBuffer();
}

//The input interface
void surfelwarp::WarpSolver::SetSolverInputs(
        CameraObservation observation,
        Renderer::SolverMaps rendered_maps,
        SurfelGeometry::SolverInput geometry_input,
        WarpField::SolverInput warpfield_input,
        SMPL::SolverInput smpl_input,
        const mat34 &world2camera
) {
    m_observation = observation;
    m_rendered_maps = rendered_maps;
    m_geometry_input = geometry_input;
    m_warpfield_input = warpfield_input;
    m_world2camera = world2camera;
    m_smpl_input = smpl_input;

    //The iteration data
    m_iteration_data.SetWarpFieldInitialValue(warpfield_input.node_se3);
}


void surfelwarp::WarpSolver::SetSolverInputs(
        CameraObservation observation,
        Renderer::SolverMaps rendered_maps,
        SurfelGeometry::SolverInput geometry_input,
        WarpField::SolverInput warpfield_input,
        SMPL::SolverInput smpl_input,
        const Matrix4f& world2camera
) {
    SetSolverInputs(
            observation,
            rendered_maps,
            geometry_input,
            warpfield_input,
            smpl_input,
            mat34(world2camera)
    );
}

/* The buffer and method for correspondence finder
 */
void surfelwarp::WarpSolver::setDenseDepthHandlerFullInput() {
	const auto node_se3 = m_iteration_data.CurrentWarpFieldInput();
	
	//Construct the input
	m_dense_depth_handler->SetInputs(
		node_se3,
		m_knn_map,
        m_smpl_input.smpl_vertices,
        m_smpl_input.smpl_normals,
        m_smpl_input.knn,
        m_smpl_input.knn_weight,
        m_smpl_input.onbody,
		m_observation.vertex_config_map,
		m_observation.normal_radius_map,
		m_rendered_maps.reference_vertex_map,
		m_rendered_maps.reference_normal_map,
		m_rendered_maps.index_map,
		m_world2camera,
		m_image_knn_fetcher->GetImageTermPixelAndKNN()
	);
}

void surfelwarp::WarpSolver::ComputeAlignmentErrorMapDirect(cudaStream_t stream) {
	const auto node_se3 = m_iteration_data.CurrentWarpFieldInput();
	m_dense_depth_handler->ComputeAlignmentErrorMapDirect(
		node_se3,
		m_world2camera,
		m_observation.filter_foreground_mask,
		stream
	);
}


void surfelwarp::WarpSolver::ComputeAlignmentErrorOnNodes(cudaStream_t stream) {
	const auto node_se3 = m_iteration_data.CurrentWarpFieldInput();
	m_dense_depth_handler->ComputeNodewiseError(
		node_se3,
		m_world2camera,
		m_observation.filter_foreground_mask,
		stream
	);
}


void surfelwarp::WarpSolver::ComputeAlignmentErrorMapFromNode(cudaStream_t stream) {
	const auto node_se3 = m_iteration_data.CurrentWarpFieldInput();
	m_dense_depth_handler->ComputeAlignmentErrorMapFromNode(
		node_se3, 
		m_world2camera,
		m_observation.filter_foreground_mask,
		stream
	);
}


/* The buffer and method for density and foreground mask pixel finder
 */
void surfelwarp::WarpSolver::setDensityForegroundHandlerFullInput() {
	//The current node se3 from iteraion data
	const auto node_se3 = m_iteration_data.CurrentWarpFieldInput();

	//Hand in the information to handler
#if defined(USE_RENDERED_RGBA_MAP_SOLVER)
	m_density_foreground_handler->SetInputs(
		node_se3,
		m_knn_map,
		m_observation.foreground_mask,
		m_observation.filter_foreground_mask,
		m_observation.foreground_mask_gradient_map,
		m_observation.density_map,
		m_observation.density_gradient_map,
		m_rendered_maps.reference_vertex_map,
		m_rendered_maps.reference_normal_map,
		m_rendered_maps.index_map,
		m_rendered_maps.normalized_rgb_map,
		m_world2camera,
		m_image_knn_fetcher->GetImageTermPixelAndKNN()
	);
#else
	m_density_foreground_handler->SetInputs(
		node_se3,
		m_knn_map,
		m_observation.foreground_mask,
		m_observation.filter_foreground_mask,
		m_observation.foreground_mask_gradient_map,
		m_observation.density_map,
		m_observation.density_gradient_map,
		m_rendered_maps.reference_vertex_map,
		m_rendered_maps.reference_normal_map,
		m_rendered_maps.index_map,
		m_observation.normalized_rgba_prevframe,
		m_world2camera,
		m_image_knn_fetcher->GetImageTermPixelAndKNN()
	);
#endif
}

/* The method to filter the sparse feature term
 */
void surfelwarp::WarpSolver::SetSparseFeatureHandlerFullInput() {
	//The current node se3 from iteraion data
	const auto node_se3 = m_iteration_data.CurrentWarpFieldInput();

	m_sparse_correspondence_handler->SetInputs(
		node_se3,
		m_knn_map,
		m_observation.vertex_config_map,
		m_observation.correspondence_pixel_pairs,
		m_rendered_maps.reference_vertex_map,
		m_rendered_maps.index_map,
		m_world2camera
	);
}

/* The method for smooth term handler
 */
void surfelwarp::WarpSolver::computeSmoothTermNode2Jacobian(cudaStream_t stream) {
	//Prepare the input
	m_graph_smooth_handler->SetInputs(
		m_iteration_data.CurrentWarpFieldInput(),
		m_warpfield_input.node_graph,
		m_warpfield_input.reference_node_coords
	);
	
	//Do it
	m_graph_smooth_handler->BuildTerm2Jacobian(stream);
}

/* The method for bind term handler
 */
void surfelwarp::WarpSolver::computeBindTermNode2Jacobian(cudaStream_t stream) {
    //Prepare the input
    m_graph_bind_handler->SetInputs(
            m_iteration_data.CurrentWarpFieldInput(),
            m_warpfield_input.reference_node_coords,
            m_warpfield_input.node_index,
            m_smpl_input
    );

    //Do it
    m_graph_bind_handler->BuildTerm2Jacobian(stream);
}

/* The index from node to term index
 */
void surfelwarp::WarpSolver::SetNode2TermIndexInput() {
	const auto dense_depth_knn = m_image_knn_fetcher->DenseImageTermKNNArray();
	const auto density_map_knn = DeviceArrayView<ushort4>(); //Empty
	const auto node_graph = m_warpfield_input.node_graph;
	const auto node_bind_index = m_graph_bind_handler->GetIndex();
	const auto foreground_mask_knn = m_density_foreground_handler->ForegroundMaskTermKNN();
	const auto sparse_feature_knn = m_sparse_correspondence_handler->SparseFeatureKNN();
	m_node2term_index->SetInputs(
		dense_depth_knn,
		node_graph,
        node_bind_index,
		m_warpfield_input.node_se3.Size(),
		foreground_mask_knn,
		sparse_feature_knn
	);

	const auto num_nodes = m_warpfield_input.node_se3.Size();
	m_nodepair2term_index->SetInputs(
		num_nodes,
		dense_depth_knn,
		node_graph,
		foreground_mask_knn,
		sparse_feature_knn
	);
}

void surfelwarp::WarpSolver::BuildNodePair2TermIndexBlocked(cudaStream_t stream) {
	m_nodepair2term_index->BuildHalfIndex(stream);
	m_nodepair2term_index->QueryValidNodePairSize(stream); //This will blocked
	
	//The later computation depends on the size
	m_nodepair2term_index->BuildSymmetricAndRowBlocksIndex(stream);
}

//Assume the SE3 for each term expepted smooth term is updated
void surfelwarp::WarpSolver::ComputeTermJacobianFixedIndex(
	cudaStream_t dense_depth,
	cudaStream_t density_map,
	cudaStream_t foreground_mask,
	cudaStream_t sparse_feature
) {
	m_dense_depth_handler->ComputeJacobianTermsFixedIndex(dense_depth);
	computeSmoothTermNode2Jacobian(sparse_feature);
	m_density_foreground_handler->ComputeTwistGradient(density_map, foreground_mask);
	m_sparse_correspondence_handler->BuildTerm2Jacobian(sparse_feature);
}

/* Compute the preconditioner and linear equation rhs for later use
 */
void surfelwarp::WarpSolver::SetPreconditionerBuilderAndJtJApplierInput() {
    //Map from node to term
	const auto node2term = m_node2term_index->GetNode2TermMap();
    //Map from nodepair to term
    const auto nodepair2term = m_nodepair2term_index->GetNodePair2TermMap();
	//The dense depth term
	const auto dense_depth_term2jacobian = m_dense_depth_handler->Term2JacobianMap();
	//The node graph term
	const auto smooth_term2jacobian = m_graph_smooth_handler->Term2JacobianMap();
    const auto bind_term2jacobian = m_graph_bind_handler->Term2JacobianMap();
	//The image map term
	DensityMapTerm2Jacobian density_term2jacobian;
	ForegroundMaskTerm2Jacobian foreground_term2jacobian;
	m_density_foreground_handler->Term2JacobianMaps(density_term2jacobian, foreground_term2jacobian);
	//The sparse feature term
	const auto feature_term2jacobian = m_sparse_correspondence_handler->Term2JacobianMap();
	//The penalty constants
	const auto penalty_constants = m_iteration_data.CurrentPenaltyConstants();

    //Hand in the input to preconditioner builder
	m_preconditioner_rhs_builder->SetInputs(
		node2term,
		dense_depth_term2jacobian,
		smooth_term2jacobian,
        bind_term2jacobian,
		density_term2jacobian,
		foreground_term2jacobian,
		feature_term2jacobian,
		penalty_constants
	);
    //Hand in to materializer
    m_jtj_materializer->SetInputs(
            nodepair2term,
            dense_depth_term2jacobian,
            smooth_term2jacobian,
            density_term2jacobian,
            foreground_term2jacobian,
            feature_term2jacobian,
            node2term,
            penalty_constants
    );
}




