#pragma once

#include "common/macro_utils.h"
#include "common/common_types.h"
#include "common/ArrayView.h"
#include "common/SynchronizeArray.h"
#include "common/algorithm_types.h"
#include <memory>

namespace surfelwarp {
	
	class PairsSorting {
	public:
		using Ptr = std::shared_ptr<PairsSorting>;
        PairsSorting();
        ~PairsSorting();
        SURFELWARP_NO_COPY_ASSIGN_MOVE(PairsSorting);

		void PerformSorting(
            const surfelwarp::DeviceArrayView<ushort4> &knn,
            const surfelwarp::DeviceArrayView<float> &dist,
            surfelwarp::SynchronizeArray<ushort2> &pairs,
			cudaStream_t stream = 0
		);
	private:
		DeviceBufferArray<ushort> m_smpl_key;
        DeviceBufferArray<ushort> m_surfel_ind;
		void buildKeyForPoints(const DeviceArrayView<ushort4>& knn, cudaStream_t stream = 0);

		KeyValueSort<ushort, ushort> m_smpl_key_sort;
		DeviceBufferArray<unsigned> m_smpl_label;
		PrefixSum m_smpl_label_prefixsum;
		DeviceBufferArray<ushort> m_compacted_smpl_key;
		DeviceBufferArray<int> m_compacted_smpl_offset;
		void sortCompactKeys(cudaStream_t stream = 0);

		void collectSynchronizePairs(
            const surfelwarp::DeviceArrayView<float> &dist,
            surfelwarp::SynchronizeArray<ushort2> &pairs,
			cudaStream_t stream = 0
		);
	};
}

