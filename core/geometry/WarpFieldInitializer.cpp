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
	const DeviceArrayView<float4>& reference_vertex,
	WarpField::Ptr warp_field, SMPL::Ptr smpl,
    const DeviceArray<float> &beta,
	cudaStream_t stream
) {
	//First subsampling
	std::cout << "start subsampling\n";
    DeviceArray<float4> onbody, farbody;
	smpl->SplitOnBodyVertices(reference_vertex, beta, onbody, farbody, stream);
	auto onbody_read = DeviceArrayView<float4>(onbody);
    auto farbody_read  = DeviceArrayView<float4>(farbody);

cudaSafeCall(cudaStreamSynchronize(stream));

    SynchronizeArray<float4> onbody_node_candidates;
    m_vertex_subsampler->PerformSubsample(onbody_read, onbody_node_candidates,
            0.7f * Constants::kNodeRadius, stream);
cudaSafeCall(cudaStreamSynchronize(stream));
	std::cout << "perform subsample 1\n";
    SynchronizeArray<float4> farbody_node_candidates;
    m_vertex_subsampler->PerformSubsample(farbody_read, farbody_node_candidates,
            2.0f * Constants::kNodeRadius, stream);
	std::cout << "perform subsample 2\n";
    auto h_onbody = onbody_node_candidates.HostArray();
    auto h_farbody = farbody_node_candidates.HostArray();
    std::copy(h_farbody.begin(), h_farbody.end(), std::back_inserter(h_onbody));

	WarpFieldUpdater::InitializeReferenceNodesAndSE3FromCandidates(*warp_field, h_onbody, stream);

	std::cout << "end subsample\n";
}
