//
// Created by wei on 4/16/18.
//
#include "common/ConfigParser.h"
#include "common/Constants.h"
#include "core/warp_solver/NodePair2TermsIndex.h"
#include "core/warp_solver/solver_encode.h"


surfelwarp::NodePair2TermsIndex::NodePair2TermsIndex() {
	memset(&m_term2node, 0, sizeof(m_term2node));
	memset(&m_term_offset, 0, sizeof(TermTypeOffset));
}

void surfelwarp::NodePair2TermsIndex::AllocateBuffer() {
	const auto& config = ConfigParser::Instance();
	const auto num_pixels = config.clip_image_cols() * config.clip_image_rows();
	const auto max_dense_depth_terms = num_pixels;
	const auto max_node_graph_terms = Constants::kMaxNumNodes * Constants::kNumNodeGraphNeigbours;
	const auto max_density_terms = num_pixels;
	const auto max_foreground_terms = num_pixels / 2; //Only part on boundary
	const auto max_feature_terms = Constants::kMaxMatchedSparseFeature;
	
	const auto kv_buffer_size = 6 * (max_dense_depth_terms + max_density_terms + max_foreground_terms + max_feature_terms) + 1 * max_node_graph_terms;
	
	//Allocate the key-value pair
	m_nodepair_keys.AllocateBuffer(kv_buffer_size);
	m_term_idx_values.AllocateBuffer(kv_buffer_size);
	
	//Allocate the sorter and compaction
	m_nodepair2term_sorter.AllocateBuffer(kv_buffer_size);
	m_segment_label.AllocateBuffer(kv_buffer_size);
	m_segment_label_prefixsum.AllocateBuffer(kv_buffer_size);
	const auto max_unique_nodepair = Constants::kMaxNumNodePairs;
	m_half_nodepair_keys.AllocateBuffer(max_unique_nodepair);
	m_half_nodepair2term_offset.AllocateBuffer(max_unique_nodepair);
	
	//The buffer for symmetric index
	m_compacted_nodepair_keys.AllocateBuffer(2 * max_unique_nodepair);
	m_nodepair_term_range.AllocateBuffer(2 * max_unique_nodepair);
	m_symmetric_kv_sorter.AllocateBuffer(2 * max_unique_nodepair);
	
	//For blocked offset and length of each row
	m_blkrow_offset_array.AllocateBuffer(Constants::kMaxNumNodes + 1);
	m_blkrow_length_array.AllocateBuffer(Constants::kMaxNumNodes);
	
	//For offset measured in bin
	const auto max_bins = divUp(Constants::kMaxNumNodes * 6, 32);
	m_binlength_array.AllocateBuffer(max_bins);
	m_binnonzeros_prefixsum.AllocateBuffer(max_bins + 1);
	m_binblocked_csr_rowptr.AllocateBuffer(32 * (max_bins + 1));
	
	//For the colptr of bin blocked csr format
	m_binblocked_csr_colptr.AllocateBuffer(6 * Constants::kMaxNumNodePairs);
}

void surfelwarp::NodePair2TermsIndex::ReleaseBuffer() {
	m_nodepair_keys.ReleaseBuffer();
	m_term_idx_values.ReleaseBuffer();
	
	m_segment_label.ReleaseBuffer();
	
	m_compacted_nodepair_keys.ReleaseBuffer();
	m_nodepair_term_range.ReleaseBuffer();
}

void surfelwarp::NodePair2TermsIndex::SetInputs(
	unsigned num_nodes,
	surfelwarp::DeviceArrayView<ushort4> dense_image_knn,
	surfelwarp::DeviceArrayView<ushort2> node_graph,
    DeviceArrayView<int> node_bind_index,
	surfelwarp::DeviceArrayView<ushort4> foreground_mask_knn,
	surfelwarp::DeviceArrayView<ushort4> sparse_feature_knn)
{
	m_num_nodes = num_nodes;
	
	m_term2node.dense_image_knn = dense_image_knn;
	m_term2node.node_graph = node_graph;
	m_term2node.foreground_mask_knn = foreground_mask_knn;
	m_term2node.sparse_feature_knn = sparse_feature_knn;
	
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

void surfelwarp::NodePair2TermsIndex::BuildHalfIndex(cudaStream_t stream) {
	buildTermKeyValue(stream);
	sortCompactTermIndex(stream);
}

void surfelwarp::NodePair2TermsIndex::BuildSymmetricAndRowBlocksIndex(cudaStream_t stream) {
	buildSymmetricCompactedIndex(stream);
	
	//The map from row to blks
	computeBlockRowLength(stream);
	computeBinLength(stream);
	computeBinBlockCSRRowPtr(stream);
	
	//Compute the column ptr
	nullifyBinBlockCSRColumePtr(stream);
	computeBinBlockCSRColumnPtr(stream);
}

unsigned surfelwarp::NodePair2TermsIndex::NumTerms() const {
	return
		  m_term2node.dense_image_knn.Size()
		+ m_term2node.node_graph.Size()
		+ m_term2node.foreground_mask_knn.Size()
		+ m_term2node.sparse_feature_knn.Size();
}

unsigned surfelwarp::NodePair2TermsIndex::NumKeyValuePairs() const {
	return
		m_term2node.dense_image_knn.Size() * 6
		+ m_term2node.node_graph.Size() * 1
		+ m_term2node.foreground_mask_knn.Size() * 6
		+ m_term2node.sparse_feature_knn.Size() * 6;
}

surfelwarp::NodePair2TermsIndex::NodePair2TermMap surfelwarp::NodePair2TermsIndex::GetNodePair2TermMap() const {
	NodePair2TermMap map;
	map.encoded_nodepair = m_symmetric_kv_sorter.valid_sorted_key;
	map.nodepair_term_range = m_symmetric_kv_sorter.valid_sorted_value;
	map.nodepair_term_index = m_nodepair2term_sorter.valid_sorted_value;
	map.term_offset = m_term_offset;
	
	//For bin blocked csr format
	map.blkrow_offset = m_blkrow_offset_array.ArrayView();
	map.binblock_csr_rowptr = m_binblocked_csr_rowptr.ArrayView();
	map.binblock_csr_colptr = m_binblocked_csr_colptr.Ptr(); //The size is not required, and not queried
	return map;
}