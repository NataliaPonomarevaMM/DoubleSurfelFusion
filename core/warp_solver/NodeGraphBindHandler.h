#ifndef SURFELWARP_NODEGRAPHBINDHANDLER_H
#define SURFELWARP_NODEGRAPHBINDHANDLER_H

#include "common/macro_utils.h"
#include "common/ArrayView.h"
#include "common/DeviceBufferArray.h"
#include "math/DualQuaternion.hpp"
#include "core/warp_solver/solver_types.h"
#include "core/smpl/smpl.h"
#include "core/smpl/cuda/apply.cuh"

#include <memory>

namespace surfelwarp {

    class NodeGraphBindHandler {
    private:
        //The input data from solver
        DeviceArrayView<DualQuaternion> m_node_se3;
        DeviceArrayView<float4> m_reference_node_coords;

        //index of node in surfel array
        DeviceArrayView<int> m_node_index;
        //index in array of nodes
        DeviceArray<int> m_index;

        DeviceArrayView<float3> m_smpl_vertices;
        DeviceArrayView<ushort4> m_knn;
        DeviceArrayView<float4> m_knn_weight;
        DeviceArrayView<int> m_onbody;

        //Do a forward warp on nodes
        DeviceBufferArray<float3> Ti_xi_;
        DeviceBufferArray<float3> xi_;
    public:
        using Ptr = std::shared_ptr<NodeGraphBindHandler>;
        NodeGraphBindHandler();
        ~NodeGraphBindHandler();
        SURFELWARP_NO_COPY_ASSIGN_MOVE(NodeGraphBindHandler);

        //The input interface from solver
        void SetInputs(
                const DeviceArrayView<DualQuaternion>& node_se3,
                const DeviceArrayView<float4>& reference_nodes,
                const DeviceArrayView<int> &node_index,
                const SMPL::SolverInput &smpl_input
        );
        void BuildTerm2Jacobian(cudaStream_t stream = 0);
        NodeGraphBindTerm2Jacobian Term2JacobianMap() const;
        DeviceArrayView<int> GetIndex() const;
    };
} // surfelwarp

#endif //SURFELWARP_NODEGRAPHBINDHANDLER_H
