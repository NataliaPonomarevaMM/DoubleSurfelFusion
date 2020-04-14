//
// Created by wei on 3/23/18.
//

#pragma once

#include "common/macro_utils.h"
#include "common/common_types.h"
#include "common/ArrayView.h"
#include "common/SynchronizeArray.h"
#include "common/algorithm_types.h"
#include "core/geometry/VoxelSubsampler.h"
#include <memory>

namespace surfelwarp {
	
	class VoxelSubsamplerSorting : public VoxelSubsampler {
	public:
		using Ptr = std::shared_ptr<VoxelSubsamplerSorting>;
		SURFELWARP_DEFAULT_CONSTRUCT_DESTRUCT(VoxelSubsamplerSorting);
		
		//Again, explicit malloc
		void AllocateBuffer(unsigned max_input_points) override;
		void ReleaseBuffer() override;

		//Assume PRE-ALLOCATRED buffer and the
		//AllocateBuffer has been invoked
		void PerformSubsample(
			const DeviceArrayView<float4>& points,
			SynchronizeArray<float4>& subsampled_points,
            SynchronizeArray<int>& subsampled_points_ind,
			const float voxel_size,
			cudaStream_t stream = 0
		) override;
	
		/* Take the input as points, build voxel key for each point
		 * Assume the m_point_key is in correct size
		 */
	private:
		DeviceBufferArray<int> m_point_key;
        DeviceBufferArray<int> m_point_key_ind;
        DeviceBufferArray<int> m_point_ind;
		void buildVoxelKeyForPoints(const DeviceArrayView<float4>& points, const float voxel_size, cudaStream_t stream = 0);

		/* Perform sorting and compaction on the voxel key
		 */
		KeyValueSort<int, float4> m_point_key_sort;
        KeyValueSort<int, int> m_point_ind_sort;
		DeviceBufferArray<unsigned> m_voxel_label;
		PrefixSum m_voxel_label_prefixsum;
		DeviceBufferArray<int> m_compacted_voxel_key;
		DeviceBufferArray<int> m_compacted_voxel_offset;
		void sortCompactVoxelKeys(const DeviceArrayView<float4>& points, cudaStream_t stream = 0);

		//Collected the subsampled points and sync it to host
		void collectSynchronizeSubsampledPoint(
			SynchronizeArray<float4>& subsampled_points,
            surfelwarp::SynchronizeArray<int> &subsampled_points_ind,
			const float voxel_size,
			cudaStream_t stream = 0
		);
	};
}

