//
// Created by wei on 5/10/18.
//

#include "common/Constants.h"
#include "core/geometry/WarpFieldInitializer.h"
#include "core/geometry/WarpFieldUpdater.h"
#include "core/geometry/VoxelSubsamplerSorting.h"

surfelwarp::WarpFieldInitializer::WarpFieldInitializer() {
	m_vertex_subsampler = std::make_shared<VoxelSubsamplerSorting>();
	m_vertex_subsampler->AllocateBuffer(Constants::kMaxNumSurfels);
	m_node_candidate.AllocateBuffer(Constants::kMaxSubsampleFrom * Constants::kMaxNumNodes);
}


surfelwarp::WarpFieldInitializer::~WarpFieldInitializer() {
	m_vertex_subsampler->ReleaseBuffer();
}

void surfelwarp::WarpFieldInitializer::InitializeReferenceNodeAndSE3FromVertex(
	const DeviceArrayView<float4>& onbody_vertex,
    const DeviceArrayView<float4>& farbody_vertex,
	WarpField::Ptr warp_field,
	cudaStream_t stream
) {
	//First subsampling
    SynchronizeArray<float4> onbody_node_candidates;
    onbody_node_candidates.AllocateBuffer(Constants::kMaxSubsampleFrom * Constants::kMaxNumNodes);
	if (onbody_vertex.Size() > 0)
	    m_vertex_subsampler->PerformSubsample(onbody_vertex, onbody_node_candidates,
            0.7f * Constants::kNodeRadius, stream);

    SynchronizeArray<float4> farbody_node_candidates;
	farbody_node_candidates.AllocateBuffer(Constants::kMaxSubsampleFrom * Constants::kMaxNumNodes);
	if (farbody_vertex.Size() > 0)
   		m_vertex_subsampler->PerformSubsample(farbody_vertex, farbody_node_candidates,
            2.0f * Constants::kNodeRadius, stream);

    auto h_onbody = onbody_node_candidates.HostArray();
    auto h_farbody = farbody_node_candidates.HostArray();
    std::copy(h_farbody.begin(), h_farbody.end(), std::back_inserter(h_onbody));

	WarpFieldUpdater::InitializeReferenceNodesAndSE3FromCandidates(*warp_field, h_onbody, stream);
}
