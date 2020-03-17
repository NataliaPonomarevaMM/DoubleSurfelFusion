#pragma once
#include "common/common_types.h"

namespace surfelwarp {
    namespace device {
        __global__ float3 apply(
                        const PtrSz<const float> smpl_vertices,
                        const ushort4 knn,
                        const float4 knn_weight,
                        cudaStream_t stream
        ) {
            float coord[3];
            for (int k = 0; k < 3; k++) {
                coord[k] += smpl_vertices[knn.x * 3 + k] * knn_weight.x;
                coord[k] += smpl_vertices[knn.y * 3 + k] * knn_weight.y;
                coord[k] += smpl_vertices[knn.z * 3 + k] * knn_weight.z;
                coord[k] += smpl_vertices[knn.w * 3 + k] * knn_weight.w;
            }
            return make_float3(coord[0], coord[1], coord[2]);
        }
    }
}
