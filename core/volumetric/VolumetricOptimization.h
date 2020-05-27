//
// Created by Natalia on 17/04/2020.
//

#ifndef SURFELWARP_VOLUMETRICOPTIMIZATION_H
#define SURFELWARP_VOLUMETRICOPTIMIZATION_H

#include "common/macro_utils.h"
#include "common/common_types.h"
#include "common/CameraObservation.h"
#include "common/ArraySlice.h"
#include "common/surfel_types.h"
#include "core/render/Renderer.h"
#include "core/SurfelGeometry.h"
#include "core/WarpField.h"
#include "core/smpl/smpl.h"
#include "pcg_solver/BlockPCG.h"
#include "math/DualQuaternion.hpp"
#include <memory>

namespace surfelwarp {

    class VolumetricOptimization {
    private:
        SMPL::Ptr m_smpl_handler;
        DeviceArray<int> m_knn;
        DeviceArray<float4> m_right_vert;

        void CheckPoints(
                DeviceArrayView<float4> &live_vertex,
                cudaStream_t stream
        );
        void ComputeJacobian(
                DeviceArrayView<float4> &live_vertex,
                float *j,
                cudaStream_t stream
        );
        void ComputeResidual(
                DeviceArrayView<float4> &live_vertex,
                float *r,
                cudaStream_t stream
        );
    public:
        using Ptr = std::shared_ptr<VolumetricOptimization>;
        VolumetricOptimization() = default;
        ~VolumetricOptimization() = default;
        SURFELWARP_NO_COPY_ASSIGN_MOVE(VolumetricOptimization);


        void Initialize(
                SMPL::Ptr smpl_handler,
                DeviceArrayView<float4> &live_vertex,
                cudaStream_t stream = 0
        );
        void Solve(
                DeviceArrayView<float4> &m_live_vertex,
                cudaStream_t stream = 0
        );
    };
}


#endif //SURFELWARP_VOLUMETRICOPTIMIZATION_H
