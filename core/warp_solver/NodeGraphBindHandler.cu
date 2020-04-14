#include "core/warp_solver/NodeGraphBindHandler.h"
#include "common/Constants.h"
#include "core/warp_solver/solver_constants.h"
#include "core/smpl/cuda/apply.cuh"

#include <device_launch_parameters.h>

namespace surfelwarp { namespace device {
        __global__ void forwardWarpBinderNodeKernel(
                DeviceArrayView<float4> reference_node_array,
                const DualQuaternion* node_se3,
                const int* node_index,
                const int* onbody,
                const float3* smpl_vertices,
                const ushort4* knn,
                const float4* knn_weight,
                float3* Ti_xi_array,
                float3* xi_array,
                PtrSz<int> index
        ) {
            const auto idx = threadIdx.x + blockIdx.x * blockDim.x;
            if (idx < reference_node_array.Size()) {
                const auto smpl_idx = onbody[node_index[idx]];
                if (smpl_idx == -1)
                    return;
                const auto xi = reference_node_array[idx];
                DualQuaternion dq_i = node_se3[idx];
                const mat34 Ti = dq_i.se3_matrix();
                auto Ti_xi = Ti.rot * xi + Ti.trans;
                auto xi_ = apply(smpl_vertices, knn[smpl_idx], knn_weight[smpl_idx]);
                //Save all the data
                Ti_xi_array[idx] = Ti_xi;
                xi_array[idx] = xi_;
            }
        }

        __global__ void fill_index(
                DeviceArrayView<int> node_index,
                const int* onbody,
                PtrSz<int> index
        ) {
            int on = 0;
            for (int i = 0; i < node_index.Size(); i++)
                if (onbody[node_index[i]])
                    index[on++] = i;
        }
    } // device
} // surfelwarp


surfelwarp::NodeGraphBindHandler::NodeGraphBindHandler() {
    const auto num_bind_terms = Constants::kMaxNumNodes;
    Ti_xi_.AllocateBuffer(num_bind_terms);
    xi_.AllocateBuffer(num_bind_terms);
}

surfelwarp::NodeGraphBindHandler::~NodeGraphBindHandler() {
    Ti_xi_.ReleaseBuffer();
    xi_.ReleaseBuffer();
}

void surfelwarp::NodeGraphBindHandler::SetInputs(
        const DeviceArrayView<DualQuaternion>& node_se3,
        const DeviceArrayView<float4>& reference_nodes,
        const DeviceArrayView<int> &node_index,
        const SMPL::SolverInput &smpl_input
) {
    m_node_se3 = node_se3;
    m_reference_node_coords = reference_nodes;
    m_node_index = node_index;
    m_smpl_vertices = smpl_input.smpl_vertices;
    m_knn = smpl_input.knn;
    m_knn_weight = smpl_input.knn_weight;
    m_onbody = smpl_input.onbody;
}


/* The method to build the term2jacobian
 */
void surfelwarp::NodeGraphBindHandler::BuildTerm2Jacobian(cudaStream_t stream) {
    std::vector<int> h_node_ind, h_onbody;
    m_node_index.Download(h_node_ind);
    m_onbody.Download(h_onbody);

    int count = 0;
    for (int i = 0; i < h_node_ind.size(); i++) {
        const auto smpl_idx = h_onbody[h_node_ind[i]];
        if (smpl_idx != -1)
            count++;
    }
    m_index = DeviceArray<int>(count);
    device::fill_index<<<1,1,0,stream>>>(m_node_index, m_onbody.RawPtr(), m_index);

    Ti_xi_.ResizeArrayOrException(count);
    xi_.ResizeArrayOrException(count);

    dim3 blk(128);
    dim3 grid(divUp(m_reference_node_coords.Size(), blk.x));
    device::forwardWarpBinderNodeKernel<<<grid, blk, 0, stream>>>(
            m_reference_node_coords,
            m_node_se3.RawPtr(),
            m_node_index.RawPtr(),
            m_onbody.RawPtr(),
            m_smpl_vertices.RawPtr(),
            m_knn.RawPtr(),
            m_knn_weight.RawPtr(),
            Ti_xi_.Ptr(), xi_.Ptr(),
            m_index
    );

    //Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
    cudaSafeCall(cudaStreamSynchronize(stream));
	cudaSafeCall(cudaGetLastError());
#endif
}

surfelwarp::NodeGraphBindTerm2Jacobian surfelwarp::NodeGraphBindHandler::Term2JacobianMap() const
{
    NodeGraphBindTerm2Jacobian map;
    map.Ti_xi = Ti_xi_.ArrayView();
    map.xi = xi_.ArrayView();
    return map;
}
