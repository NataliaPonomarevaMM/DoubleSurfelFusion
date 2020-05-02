#include "common/encode_utils.h"
#include "common/Constants.h"
#include "math/vector_ops.hpp"
#include "core/volumetric/PairsSorting.h"
#include <device_launch_parameters.h>

namespace surfelwarp { namespace device {
	
	__global__ void createSMPLKeyKernel(
		DeviceArrayView<ushort4> knn,
		ushort* smpl_key,
        ushort* surfel_ind
	) {
        const auto idx = threadIdx.x + blockDim.x * blockIdx.x;
        if (idx >= knn.Size())
            return;
		smpl_key[idx] = knn[idx].x;
        surfel_ind[idx] = idx;
	}

	__global__ void labelSortedKeyKernel(
		const PtrSz<const ushort> sorted_smpl_key,
		unsigned* key_label
	) {
		int idx = threadIdx.x + blockDim.x * blockIdx.x;
		if (idx == 0) key_label[0] = 1;
		else {
			if (sorted_smpl_key[idx] != sorted_smpl_key[idx - 1])
				key_label[idx] = 1;
			else
				key_label[idx] = 0;
		}
	}


	__global__ void compactedKeyKernel(
		const PtrSz<const ushort> sorted_smpl_key,
		const unsigned* smpl_key_label,
		const unsigned* prefixsumed_label,
		ushort* compacted_key,
		DeviceArraySlice<int> compacted_offset
	) {
		const auto idx = threadIdx.x + blockDim.x * blockIdx.x;
		if (idx >= sorted_smpl_key.size) return;
		if (smpl_key_label[idx] == 1) {
			compacted_key[prefixsumed_label[idx] - 1] = sorted_smpl_key[idx];
			compacted_offset[prefixsumed_label[idx] - 1] = idx;
		}
		if (idx == 0) {
			compacted_offset[compacted_offset.Size() - 1] = sorted_smpl_key.size;
		}
	}

	__global__ void samplingPointsKernel(
		const DeviceArrayView<ushort> compacted_key,
		const int* compacted_offset,
		const ushort* sorted_surfel_ind,
        const float* dist,
		ushort2* pairs
	) {
		const auto idx = threadIdx.x + blockDim.x * blockIdx.x;
		if (idx >= compacted_key.Size()) return;

		const ushort smpl_ind = compacted_key[idx]; // smpl point

		// Find surfel closed to the smpl point
		float min_dist = 1e5;
		int min_dist_idx = compacted_offset[idx];
		for (int i = compacted_offset[idx]; i < compacted_offset[idx + 1]; i++) {
			const ushort surfel_ind = sorted_surfel_ind[i];
            const int dist_ind = surfel_ind * 6890 + smpl_ind;
			if (dist[dist_ind] < min_dist) {
				min_dist = dist[dist_ind];
				min_dist_idx = i;
			}
		}

		// Store the result to global memory
        pairs[idx] = make_ushort2(smpl_ind, sorted_surfel_ind[min_dist_idx]);
	}

}; // namespace device
}; // namespace surfelwarp


surfelwarp::PairsSorting::PairsSorting() {
	m_smpl_key.AllocateBuffer(Constants::kMaxNumSurfels);
    m_surfel_ind.AllocateBuffer(Constants::kMaxNumSurfels);
	m_smpl_key_sort.AllocateBuffer(Constants::kMaxNumSurfels);

	m_smpl_label.AllocateBuffer(Constants::kMaxNumSurfels);
	m_smpl_label_prefixsum.AllocateBuffer(Constants::kMaxNumSurfels);

	m_compacted_smpl_key.AllocateBuffer(6890);
	m_compacted_smpl_offset.AllocateBuffer(6891);
}

surfelwarp::PairsSorting::~PairsSorting() {
	//Constants::kMaxNumSurfels
	m_smpl_key.ReleaseBuffer();
    m_surfel_ind.ReleaseBuffer();

	m_smpl_label.ReleaseBuffer();
	
	//smaller buffer
	m_compacted_smpl_key.ReleaseBuffer();
	m_compacted_smpl_offset.ReleaseBuffer();
}

void surfelwarp::PairsSorting::PerformSorting(
    const surfelwarp::DeviceArrayView<ushort4> &knn,
    const surfelwarp::DeviceArrayView<float> &dist,
    surfelwarp::SynchronizeArray<ushort2> &pairs,
	cudaStream_t stream
) {
	buildKeyForPoints(knn, stream);
	sortCompactKeys(stream);
	collectSynchronizePairs(dist, pairs, stream);
}

void surfelwarp::PairsSorting::buildKeyForPoints(
	const surfelwarp::DeviceArrayView<ushort4> &knn,
	cudaStream_t stream
) {
	//Correct the size of arrays
	m_smpl_key.ResizeArrayOrException(knn.Size());
    m_surfel_ind.ResizeArrayOrException(knn.Size());
	
	//Call the method
	dim3 blk(256);
	dim3 grid(divUp(knn.Size(), blk.x));
	device::createSMPLKeyKernel<<<grid, blk, 0, stream>>>(
		knn,
		m_smpl_key,
		m_surfel_ind
	);
}

void surfelwarp::PairsSorting::sortCompactKeys(
	cudaStream_t stream
) {
	//Perform sorting
	m_smpl_key_sort.Sort(m_smpl_key.ArrayReadOnly(), m_surfel_ind.ArrayReadOnly(), stream);
	//Label the sorted keys
	m_smpl_label.ResizeArrayOrException(m_smpl_key.ArraySize());
	dim3 blk(128);
	dim3 grid(divUp(m_smpl_key.ArraySize(), blk.x));
	device::labelSortedKeyKernel<<<grid, blk, 0, stream>>>(
		m_smpl_key_sort.valid_sorted_key,
		m_smpl_label.ArraySlice()
	);
	
	//Prefix sum
	m_smpl_label_prefixsum.InclusiveSum(m_smpl_label.ArrayView(), stream);

	//Query the number of smpl points
	unsigned num_smpl_points;
	const auto& prefixsum_label = m_smpl_label_prefixsum.valid_prefixsum_array;
	cudaSafeCall(cudaMemcpyAsync(
		&num_smpl_points,
		prefixsum_label.ptr() + prefixsum_label.size() - 1,
		sizeof(unsigned),
		cudaMemcpyDeviceToHost,
		stream
	));
	cudaSafeCall(cudaStreamSynchronize(stream));
	
	//Construct the compacted array
	m_compacted_smpl_key.ResizeArrayOrException(num_smpl_points);
	m_compacted_smpl_offset.ResizeArrayOrException(num_smpl_points + 1);
	device::compactedKeyKernel<<<grid, blk, 0, stream>>>(
		m_smpl_key_sort.valid_sorted_key,
		m_smpl_label,
		prefixsum_label,
		m_compacted_smpl_key,
		m_compacted_smpl_offset.ArraySlice()
	);
}

void surfelwarp::PairsSorting::collectSynchronizePairs(
    const surfelwarp::DeviceArrayView<float> &dist,
	surfelwarp::SynchronizeArray<ushort2> &pairs,
	cudaStream_t stream
) {
	//Correct the size
	const auto num_smpl_points = m_compacted_smpl_key.ArraySize();
	pairs.ResizeArrayOrException(num_smpl_points);
	auto pairs_slice = pairs.DeviceArrayReadWrite();
	dim3 sample_blk(128);
	dim3 sample_grid(divUp(num_smpl_points, sample_blk.x));
	device::samplingPointsKernel<<<sample_grid, sample_blk, 0, stream>>>(
		m_compacted_smpl_key.ArrayView(),
		m_compacted_smpl_offset,
		m_smpl_key_sort.valid_sorted_value,
		dist.RawPtr(),
		pairs_slice
	);
	
	//Sync it to host
	pairs.SynchronizeToHost(stream);
}

