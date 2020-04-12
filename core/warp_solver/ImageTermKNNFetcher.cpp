#include "common/ConfigParser.h"
#include "core/warp_solver/ImageTermKNNFetcher.h"

surfelwarp::ImageTermKNNFetcher::ImageTermKNNFetcher() {
	//The initialization part
	const auto& config = ConfigParser::Instance();
	m_image_height = config.clip_image_rows();
	m_image_width = config.clip_image_cols();
	memset(&m_geometry_maps, 0, sizeof(m_geometry_maps));
	
	//The malloc part
	const auto num_pixels = m_image_height * m_image_width;
	m_potential_pixel_indicator.create(num_pixels);

	//For compaction
	m_indicator_prefixsum.InclusiveSum(num_pixels);
	m_potential_pixels.AllocateBuffer(num_pixels);
	m_dense_image_knn.AllocateBuffer(num_pixels);
	m_dense_image_knn_weight.AllocateBuffer(num_pixels);

	//The page-locked memory
	cudaSafeCall(cudaMallocHost((void**)&m_num_potential_pixel, sizeof(unsigned)));
}

surfelwarp::ImageTermKNNFetcher::~ImageTermKNNFetcher() {
	m_potential_pixel_indicator.release();

	m_potential_pixels.ReleaseBuffer();
	m_dense_image_knn.ReleaseBuffer();
	m_dense_image_knn_weight.ReleaseBuffer();

	cudaSafeCall(cudaFreeHost(m_num_potential_pixel));
}

void surfelwarp::ImageTermKNNFetcher::SetInputs(
	const DeviceArrayView2D<surfelwarp::KNNAndWeight> &knn_map,
	cudaTextureObject_t index_map
) {
	m_geometry_maps.knn_map = knn_map;
	m_geometry_maps.index_map = index_map;
}

