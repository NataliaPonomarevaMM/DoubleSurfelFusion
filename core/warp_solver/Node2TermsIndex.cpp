//
// Created by wei on 4/2/18.
//

#include "common/ConfigParser.h"
#include "common/logging.h"
#include "common/Constants.h"
#include "core/warp_solver/Node2TermsIndex.h"
#include "core/warp_solver/term_offset_types.h"
#include "Node2TermsIndex.h"
#include <vector>

surfelwarp::Node2TermsIndex::Node2TermsIndex() {
	memset(&m_term2node, 0, sizeof(m_term2node));
	memset(&m_term_offset, 0, sizeof(TermTypeOffset));
	m_num_nodes = 0;
}

void surfelwarp::Node2TermsIndex::AllocateBuffer() {
	const auto& config = ConfigParser::Instance();
	const auto num_pixels = config.clip_image_cols() * config.clip_image_rows();
	const auto max_dense_image_terms = num_pixels;
	const auto max_node_graph_terms = Constants::kMaxNumNodes * Constants::kNumNodeGraphNeigbours;
	const auto max_foreground_terms = num_pixels / 2; //Only part on boundary
	const auto max_feature_terms = Constants::kMaxMatchedSparseFeature;
	
	//The total maximum size of kv buffer
	const auto kv_buffer_size = 4 * (max_dense_image_terms + max_node_graph_terms + max_foreground_terms + max_feature_terms);
	
	//Allocate the key-value pair
	m_node_keys.AllocateBuffer(kv_buffer_size);
	m_term_idx_values.AllocateBuffer(kv_buffer_size);

	//Allocate the sorting and compaction buffer
	m_node2term_sorter.AllocateBuffer(kv_buffer_size);
	m_node2term_offset.AllocateBuffer(Constants::kMaxNumNodes + 1);
}

void surfelwarp::Node2TermsIndex::ReleaseBuffer() {
	m_node_keys.ReleaseBuffer();
	m_term_idx_values.ReleaseBuffer();
}

void surfelwarp::Node2TermsIndex::SetInputs(
	DeviceArrayView<ushort4> dense_image_knn,
	DeviceArrayView<ushort2> node_graph, unsigned num_nodes,
    DeviceArrayView<int> node_bind_index,
	DeviceArrayView<ushort4> foreground_mask_knn,
	DeviceArrayView<ushort4> sparse_feature_knn
) {
	m_term2node.dense_image_knn = dense_image_knn;
	m_term2node.node_graph = node_graph;
    m_term2node.node_bind_index = node_bind_index;
	m_term2node.foreground_mask_knn = foreground_mask_knn;
	m_term2node.sparse_feature_knn = sparse_feature_knn;
	m_num_nodes = num_nodes;
	
	//build the offset of these terms
	size2offset(
		m_term_offset,
		dense_image_knn,
		node_graph,
		node_bind_index,
		foreground_mask_knn,
		sparse_feature_knn
	);
}

void surfelwarp::Node2TermsIndex::BuildIndex(cudaStream_t stream) {
	buildTermKeyValue(stream);
	sortCompactTermIndex(stream);
}


/* The size query methods
 */
unsigned surfelwarp::Node2TermsIndex::NumTerms() const
{
	return 
	  m_term2node.dense_image_knn.Size()
	+ m_term2node.node_graph.Size()
	+ m_term2node.node_bind_index.Size()
	+ m_term2node.foreground_mask_knn.Size() 
	+ m_term2node.sparse_feature_knn.Size();
}


unsigned surfelwarp::Node2TermsIndex::NumKeyValuePairs() const
{
	return 
	  m_term2node.dense_image_knn.Size() * 4
	+ m_term2node.node_graph.Size() * 2
	+ m_term2node.node_bind_index.Size()
	+ m_term2node.foreground_mask_knn.Size() * 4 
	+ m_term2node.sparse_feature_knn.Size() * 4;
}