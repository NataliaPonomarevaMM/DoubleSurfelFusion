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
    public:
        using Ptr = std::shared_ptr<VolumetricOptimization>;
        VolumetricOptimization() = default;
        ~VolumetricOptimization() = default;
        SURFELWARP_NO_COPY_ASSIGN_MOVE(VolumetricOptimization);

        void Solve(
                SMPL::Ptr smpl_handler,
                DeviceArrayView<float4> &m_live_vertex,
                DeviceArrayView<float4> &m_live_normal
        );
    };
}


#endif //SURFELWARP_VOLUMETRICOPTIMIZATION_H
