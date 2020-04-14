//
// Created by wei on 4/2/18.
//

#pragma once

#include "common/macro_utils.h"
#include "common/common_types.h"
#include "common/ArrayView.h"
#include "common/DeviceBufferArray.h"
#include "common/algorithm_types.h"
#include "core/warp_solver/term_offset_types.h"
#include <memory>

namespace surfelwarp {

	class Node2TermsIndex {
	private:
		//The input map from terms to nodes, the input might be empty for dense_density, foreground mask and sparse feature
		struct {
			DeviceArrayView<ushort4> dense_image_knn; //Each depth scalar term has 4 nearest neighbour
			DeviceArrayView<ushort2> node_graph;
            DeviceArrayView<int> node_bind_index;
			DeviceArrayView<ushort4> foreground_mask_knn; //The same as density term
			DeviceArrayView<ushort4> sparse_feature_knn; //Each 4 nodes correspond to 3 scalar cost
		} m_term2node;
		//The number of nodes
		unsigned m_num_nodes;
		//The term offset of term2node map
		TermTypeOffset m_term_offset;

        DeviceBufferArray<unsigned short> m_node_keys;
        DeviceBufferArray<unsigned> m_term_idx_values;
        unsigned NumTerms() const;
        unsigned NumKeyValuePairs() const;
        void buildTermKeyValue(cudaStream_t stream = 0);

        KeyValueSort<unsigned short, unsigned> m_node2term_sorter;
        DeviceBufferArray<unsigned> m_node2term_offset;
        void sortCompactTermIndex(cudaStream_t stream = 0);
	public:
		//Accessed by pointer, default construct/destruct
		using Ptr = std::shared_ptr<Node2TermsIndex>;
		Node2TermsIndex();
		~Node2TermsIndex() = default;
		SURFELWARP_NO_COPY_ASSIGN_MOVE(Node2TermsIndex);
	
		//Explicit allocate/de-allocate
		void AllocateBuffer();
		void ReleaseBuffer();
		
		//The input
		void SetInputs(
			DeviceArrayView<ushort4> dense_image_knn,
			DeviceArrayView<ushort2> node_graph,  unsigned num_nodes,
			DeviceArrayView<int> node_bind_index,
			//These costs might be empty
			DeviceArrayView<ushort4> foreground_mask_knn = DeviceArrayView<ushort4>(),
			DeviceArrayView<ushort4> sparse_feature_knn  = DeviceArrayView<ushort4>()
		);
		
		//The main interface
		void BuildIndex(cudaStream_t stream = 0);

		struct Node2TermMap {
			DeviceArrayView<unsigned> offset;
			DeviceArrayView<unsigned> term_index;
			TermTypeOffset term_offset;
		};
		
		//Return the outside-accessed index
		Node2TermMap GetNode2TermMap() const {
			Node2TermMap map;
			map.offset = m_node2term_offset.ArrayReadOnly();
			map.term_index = m_node2term_sorter.valid_sorted_value;
			map.term_offset = m_term_offset;
			return map;
		}
	};

}