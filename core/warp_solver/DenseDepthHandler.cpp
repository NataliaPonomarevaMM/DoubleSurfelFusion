#include "common/logging.h"
#include "common/Constants.h"
#include "common/ConfigParser.h"
#include "common/sanity_check.h"
#include "common/data_transfer.h"
#include "common/common_texture_utils.h"
#include "core/warp_solver/DenseDepthHandler.h"

surfelwarp::DenseDepthHandler::DenseDepthHandler() {
	const auto& config = ConfigParser::Instance();
	m_image_height = config.clip_image_rows();
	m_image_width = config.clip_image_cols();
	m_project_intrinsic = config.rgb_intrinsic_clip();
	
	memset(&m_depth_observation, 0, sizeof(m_depth_observation));
	memset(&m_geometry_maps, 0, sizeof(m_geometry_maps));
}


void surfelwarp::DenseDepthHandler::AllocateBuffer() {
	const auto num_pixels = m_image_height * m_image_width;
	//The buffer to match the pixel pairs
	m_pixel_match_indicator.create(num_pixels);
	m_pixel_pair_maps.create(num_pixels);
	
	//The buffer to compact the pixel pairs
	m_indicator_prefixsum.AllocateBuffer(num_pixels);
	m_valid_pixel_pairs.AllocateBuffer(num_pixels);
	m_dense_depth_knn.AllocateBuffer(num_pixels);
	m_dense_depth_knn_weight.AllocateBuffer(num_pixels);
	
	//The buffer for gradient
	m_term_residual.AllocateBuffer(num_pixels);
	m_term_twist_gradient.AllocateBuffer(num_pixels);
	
	//The buffer for alignment error
	createFloat1TextureSurface(m_image_height, m_image_width, m_alignment_error_map);
	m_node_accumlate_error.AllocateBuffer(Constants::kMaxNumNodes);
	m_node_accumlate_weight.AllocateBuffer(Constants::kMaxNumNodes);
}


void surfelwarp::DenseDepthHandler::ReleaseBuffer() {
	m_pixel_match_indicator.release();
	m_pixel_pair_maps.release();
	
	m_valid_pixel_pairs.ReleaseBuffer();
	m_dense_depth_knn.ReleaseBuffer();
	m_dense_depth_knn_weight.ReleaseBuffer();
	
	m_term_residual.ReleaseBuffer();
	m_term_twist_gradient.ReleaseBuffer();
}

void surfelwarp::DenseDepthHandler::SetInputs(
	const DeviceArrayView<DualQuaternion>& node_se3,
	const DeviceArrayView2D<KNNAndWeight>& knn_map,
	//smpl
    const DeviceArrayView<float3> &smpl_vertices,
    const DeviceArrayView<float3> &smpl_normals,
    const DeviceArrayView<ushort4> &smpl_knn,
    const DeviceArrayView<float4> &smpl_knn_weight,
    const DeviceArrayView<int> &onbody,
	cudaTextureObject_t depth_vertex_map, cudaTextureObject_t depth_normal_map,
	//The rendered maps
	cudaTextureObject_t reference_vertex_map,
	cudaTextureObject_t reference_normal_map,
	cudaTextureObject_t index_map,
	const mat34& world2camera,
	//The potential pixels and knn
	const ImageTermKNNFetcher::ImageTermPixelAndKNN& pixels_knn
) {
	m_node_se3 = node_se3;
	m_knn_map = knn_map;
	
	m_world2camera = world2camera;
	m_camera2world = world2camera.inverse();
	
	m_depth_observation.vertex_map = depth_vertex_map;
	m_depth_observation.normal_map = depth_normal_map;
	
	m_geometry_maps.reference_vertex_map = reference_vertex_map;
	m_geometry_maps.reference_normal_map = reference_normal_map;
	m_geometry_maps.index_map = index_map;
	
	m_potential_pixels_knn = pixels_knn;

    //smpl
    m_smpl_vertices = smpl_vertices;
    m_smpl_normals = smpl_normals;
    m_smpl_knn = smpl_knn;
    m_smpl_knn_weight = smpl_knn_weight;
    m_onbody = onbody;
}

void surfelwarp::DenseDepthHandler::UpdateNodeSE3(surfelwarp::DeviceArrayView<surfelwarp::DualQuaternion> node_se3) {
	SURFELWARP_CHECK_EQ(node_se3.Size(), m_node_se3.Size());
	m_node_se3 = node_se3;
}