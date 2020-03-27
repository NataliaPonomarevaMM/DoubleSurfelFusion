#pragma once
#include "math/vector_ops.hpp"

namespace surfelwarp {
    namespace device {
        __host__ __device__ __forceinline__ float3 apply(
                        const float3* smpl_vertices,
                        const ushort4& knn,
                        const float4& knn_weight
        ) {
            return smpl_vertices[knn.x] * knn_weight.x +
                smpl_vertices[knn.y] * knn_weight.y +
                smpl_vertices[knn.z] * knn_weight.z +
                smpl_vertices[knn.w] * knn_weight.w;
        }

        __host__ __device__ __forceinline__ float3 apply_normal(
                        const float3* smpl_normals,
                        const ushort4& knn,
                        const float4& knn_weight
        ) {
           float3 norm = smpl_normals[knn.x] * knn_weight.x +
                            smpl_normals[knn.y] * knn_weight.y +
                            smpl_normals[knn.z] * knn_weight.z +
                            smpl_normals[knn.w] * knn_weight.w;
	   return normalized(norm);
        }
    }
}
