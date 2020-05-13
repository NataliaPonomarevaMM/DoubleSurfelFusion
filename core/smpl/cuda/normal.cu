#include <cmath>
#include "math/vector_ops.hpp"
#include "core/smpl/def.h"
#include "core/smpl/smpl.h"

namespace surfelwarp {
    namespace device {
        __global__ void count_normal(
                const PtrSz<const float3> vertices,
                const PtrSz<const int> face_ind,
                PtrSz<float3> normal
        ) {
            const auto ind = threadIdx.x + blockDim.x * blockIdx.x;
            if (ind >= face_ind.size)
                return;

		const float3 v0 = vertices[face_ind[ind * 3] - 1];
            const float3 v1 = vertices[face_ind[ind * 3 + 1] - 1];
            const float3 v2 = vertices[face_ind[ind * 3 + 2] - 1];
            auto n = cross((v1 - v0), (v2 - v0));

            for (int k = 0; k < 3; k++) {
                int vec_ind = face_ind[ind * 3 + k] - 1;
                normal[vec_ind] += n;
            }
        }

        __global__ void normalize_normal(
                PtrSz<float3> normal
        ) {
		const auto ind = threadIdx.x + blockDim.x * blockIdx.x;
            if (ind >= normal.size)
                return;
            normal[ind] = normalized(normal[ind]);
        }

        __global__ void transform(
                PtrSz<float3> normal,
                PtrSz<float3> vertex,
                const mat34 world2camera
        ) {
            const auto ind = threadIdx.x + blockDim.x * blockIdx.x;
            if (ind >= normal.size)
                return;

            auto x = 0.993545 * vertex[ind].x + -0.0299532 * vertex[ind].y + 0.109421 * vertex[ind].z + 0.0068873;
            auto y = 0.0173843 * vertex[ind].x + 0.993324 * vertex[ind].y + 0.114065 * vertex[ind].z + 0.643488;
            auto z =-0.112107 * vertex[ind].x - 0.111426 * vertex[ind].y + 0.987431 * vertex[ind].z + 1.6093;

            vertex[ind] = make_float3(x,y,z);
            //vertex[ind] = world2camera.rot * vertex[ind] + world2camera.trans;
            //normal[ind] = world2camera.rot * normal[ind];
        }
    }

    void SMPL::countNormals(
            cudaStream_t stream
    ) {
        m_smpl_normals = DeviceArray<float3>(VERTEX_NUM);

        int num_triangles = 13776 / 3;
        dim3 blk(128);
        dim3 grid(divUp(num_triangles, blk.x));
        device::count_normal<<<grid, blk,0,stream>>>(m_smpl_vertices,
                m__faceIndices, m_smpl_normals);

        blk = dim3(128);
        grid = dim3(divUp(VERTEX_NUM, blk.x));
        device::normalize_normal<<<grid, blk,0,stream>>>(m_smpl_normals);
    }

    void SMPL::CameraTransform(mat34 world2camera, cudaStream_t stream) {
        dim3 blk = dim3(128);
        dim3 grid = dim3(divUp(VERTEX_NUM, blk.x));
        device::transform<<<grid, blk,0,stream>>>(m_smpl_normals, m_smpl_vertices, world2camera);
    }

    void SMPL::transform(cudaStream_t stream) {
        dim3 blk = dim3(128);
        dim3 grid = dim3(divUp(VERTEX_NUM, blk.x));
        device::transform<<<grid, blk,0,stream>>>(m_smpl_normals, m_smpl_vertices, init_mat);
    }
}


