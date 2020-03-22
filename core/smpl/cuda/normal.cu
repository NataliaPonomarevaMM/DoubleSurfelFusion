#include <cmath>
#include "core/smpl/def.h"
#include "core/smpl/smpl.h"
#include "common/common_types.h"
#include <device_launch_parameters.h>

namespace surfelwarp {
    namespace device {
        __global__ void count_normal(
                const PtrSz<const float> vertices,
                const PtrSz<const int> face_ind,
                PtrSz<float3> normal
        ) {
            const auto ind = threadIdx.x + blockDim.x * blockIdx.x;
            if (ind >= face_ind.size)
                return;

            float3 vertex[3];
            for (int k = 0; k < 3; k++)
                vertex[k] = vertices[face_ind[ind * 3 + k] - 1];
            auto norm = cross((vertex[1] - vertex[0]), (vertex[2] - vertex[0]));

            for (int k = 0; k < 3; k++) {
                int vec_ind = face_ind[ind * 3 + k] - 1;
                normal[vec_ind] += normal;
            }
        }

        __global__ void normalize_normal(
                PtrSz<float3> normal
        ) {
            normal[ind] = normalize(normal[ind]);
        }
    }

    void SMPL::countNormals(
            cudaStream_t stream
    ) {
        m_smpl_normals = DeviceArray<float>(VERTEX_NUM);

        int num_triangles = 13776 / 3;
        dim3 blk(128);
        dim3 grid(divUp(num_triangles, blk.x));
        device::count_normal<<<grid, blk,0,stream>>>(m_smpl_vertices,
                m__faceIndices, m_smpl_normals);

        dim3 blk(128);
        dim3 grid(divUp(VERTEX_NUM, blk.x));
        device::normalize_normal<<<grid, blk,0,stream>>>(m_smpl_normals);
    }
}


