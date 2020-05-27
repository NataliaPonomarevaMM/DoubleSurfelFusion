#include "common/ConfigParser.h"
#include "common/sanity_check.h"
#include "common/logging.h"
#include "core/warp_solver/DenseDepthHandler.h"
#include "core/warp_solver/solver_constants.h"
#include "core/warp_solver/geometry_icp_jacobian.cuh"
#include "core/smpl/cuda/apply.cuh"
#include "core/smpl/smpl.h"
#include <device_launch_parameters.h>
#include <Eigen/Dense>
#include "common/Constants.h"
#include "math/vector_ops.hpp"
#include "core/volumetric/VolumetricOptimization.h"

namespace surfelwarp {
    namespace device {
        __global__ void computeJacobianKernel2(
                const float4 *live_vertex,
                const float3 *smpl_vertices,
                const float3 *smpl_vertices_jac,
                const PtrSz<const int> knn_ind,
                //Output
                float *gradient,
                float *residual
        ) {
            const auto idx = threadIdx.x + blockDim.x * blockIdx.x;
            if (idx >= knn_ind.size)
                return;

            if (knn_ind[idx] != -1) {
                const auto smpl = smpl_vertices[idx];
                const auto lv4 = live_vertex[knn_ind[idx]];
                const auto lv = make_float3(lv4.x, lv4.y, lv4.z);

                residual[idx] = (smpl.x - lv.x) * (smpl.x - lv.x) +
                                (smpl.y - lv.y) * (smpl.y - lv.y) +
                                (smpl.z - lv.z) * (smpl.z - lv.z);
                for (int i = 0; i < 72; i++)
                    gradient[i * 6891 + idx] = (smpl.x - lv.x) * smpl_vertices_jac[idx * 72 + i].x +
                                               (smpl.y - lv.y) * smpl_vertices_jac[idx * 72 + i].y +
                                               (smpl.z - lv.z) * smpl_vertices_jac[idx * 72 + i].z;
            }
        }

        __global__ void copy_right(
                const float4 *live_vertex,
                const PtrSz<const int> knn_ind,
                const PtrSz<const int> d_body,
                const PtrSz<const int> d_left_arm,
                const PtrSz<const int> d_right_arm,
                int *knn_body,
                int *knn_left_arm,
                int *knn_right_arm,
                float4 *right
        ) {
            const auto idx = threadIdx.x + blockDim.x * blockIdx.x;
            if (idx > knn_ind.size)
                return;
            right[idx] = live_vertex[knn_ind[idx]];

            bool t_body = false;
            for (int i = 0; i < d_body.size; i++)
                if (d_body[i] == idx)
                    t_body = true;
            knn_body[idx] = t_body ? knn_ind[idx] : -1;

            bool t_left_arm = false;
            for (int i = 0; i < d_left_arm.size; i++)
                if (d_left_arm[i] == idx)
                    t_left_arm = true;
            knn_left_arm[idx] = t_left_arm ? knn_ind[idx] : -1;

            bool t_right_arm = false;
            for (int i = 0; i < d_right_arm.size; i++)
                if (d_right_arm[i] == idx)
                    t_right_arm = true;
            knn_right_arm[idx] = t_right_arm ? knn_ind[idx] : -1;
        }

        __host__ __device__ __forceinline__ bool check(
                const float4 *live_vertex,
                const PtrSz<const int> knn,
                const float4 *right,
                const int idx
        ) {
            return abs(right[idx].x - live_vertex[knn[idx]].x) > 0.03 ||
                   abs(right[idx].y - live_vertex[knn[idx]].y) > 0.03 ||
                   abs(right[idx].z - live_vertex[knn[idx]].z) > 0.03;
        }

        __global__ void check_right(
                const float4 *live_vertex,
                const PtrSz<const int> knn_body,
                const PtrSz<const int> knn_left_arm,
                const PtrSz<const int> knn_right_arm,
                const float4 *right,
                bool *ans
        ) {
            const auto idx = threadIdx.x + blockDim.x * blockIdx.x;
            if (idx > knn_body.size)
                return;

            if (knn_body[idx] != -1 && check(live_vertex, knn_body, right, idx) ||
                knn_left_arm[idx] != -1 && check(live_vertex, knn_left_arm, right, idx) ||
                knn_right_arm[idx] != -1 && check(live_vertex, knn_right_arm, right, idx))
                *ans = true;
        }
    }
}

void surfelwarp::VolumetricOptimization::Initialize(
        SMPL::Ptr smpl_handler,
        DeviceArrayView<float4> &live_vertex,
        cudaStream_t stream
) {
    int body_size = 85;
    int left_arm_size = 11;
    int right_arm_size = 11;
//    int right_arm[right_arm_size];
//    int left_arm[left_arm_size];
//    int body[body_size];
//    for (int i = 0; i < left_arm_size; i++)
//        left_arm[i] = rand() % 1500 + 1000;
//    for (int i = 0; i < right_arm_size; i++)
//        right_arm[i] = rand() % 1500 + 4000;
//    for (int i = 0; i < body_size; i++)
//        body[i] = rand() % 6890;

    int right_arm[11] =
            //4873, 4879, 4790, 5144, 5125, 5213, 5211, 5050, 5028, 5573, 5042
            {5684, 5696, 5408, 5028, 5066, 5108, 4878, 4873, 4122, 5041, 5210};
    int left_arm[] =
            // 2112, 1919, 1559, 1551, 1742, 1615, 1652, 1678, 2819, 1343, 1842
            {1400, 1407, 1311, 628, 1665, 1597, 1579, 2874, 2112, 1572, 1314};
    int body[] = {7, 17, 20, 22, 40, 109, 122, 136, 156, 166,
                  193, 195, 237, 239, 260, 261, 264, 310, 318, 322,
                  341, 351, 358, 383, 410, 484, 485, 495, 499, 540,
                  557, 576, 590, 667, 862, 867, 1225, 1448, 1741, 1864,
                  1872, 1897, 1898, 1925, 2780, 2794, 2922, 3008, 3128, 3131,
                  3517, 3533, 3546, 3729, 3747, 3800, 3821, 3827, 3832, 4061,
                  4081, 4115, 4152, 4181, 4183, 4343, 4689, 4723, 4780, 4781,
                  4809, 4818, 4821, 4871, 4882, 4948, 4981, 5012, 5234, 5269,
                  6280, 6316, 6497, 6500, 6556};
    d_body.upload(body, body_size);
    d_left_arm.upload(left_arm, left_arm_size);
    d_right_arm.upload(right_arm, right_arm_size);

    m_smpl_handler = smpl_handler;

    auto knn_ind = DeviceArray<int>(VERTEX_NUM);
    auto dist = DeviceArray<float>(VERTEX_NUM);
    m_smpl_handler->count_dist_to_smpl(live_vertex, knn_ind, dist, 0);

    m_knn_body = DeviceArray<int>(VERTEX_NUM);
    m_knn_right_arm = DeviceArray<int>(VERTEX_NUM);
    m_knn_left_arm = DeviceArray<int>(VERTEX_NUM);
    m_right_vert = DeviceArray<float4>(VERTEX_NUM);

    dim3 blk(128);
    dim3 grid(divUp(knn_ind.size(), blk.x));
    device::copy_right<<<grid, blk, 0, stream>>>(
            live_vertex.RawPtr(), knn_ind,
            d_body, d_left_arm, d_right_arm,
            m_knn_body.ptr(), m_knn_left_arm.ptr(), m_knn_right_arm.ptr(),
            m_right_vert.ptr()
    );
}

void surfelwarp::VolumetricOptimization::CheckPoints(
        DeviceArrayView<float4> &live_vertex,
        cudaStream_t stream
) {
    dim3 blk(128);
    dim3 grid(divUp(VERTEX_NUM, blk.x));

    bool h_answer;
    bool *d_answer;
    cudaMalloc((void **)(&d_answer), sizeof(bool));
    cudaMemset(d_answer, 0, sizeof(bool));
    device::check_right<<<grid, blk, 0, stream>>>(
            live_vertex.RawPtr(),
            m_knn_body, m_knn_left_arm, m_knn_right_arm,
            m_right_vert.ptr(),
            d_answer
    );
    cudaMemcpy(&h_answer, d_answer, sizeof(bool), cudaMemcpyDeviceToHost);
    cudaFree(d_answer);
    if (h_answer) {
        auto knn_ind = DeviceArray<int>(VERTEX_NUM);
        auto dist = DeviceArray<float>(VERTEX_NUM);
        m_smpl_handler->count_dist_to_smpl(live_vertex, knn_ind, dist, 0);
        device::copy_right<<<grid, blk, 0, stream>>>(
                live_vertex.RawPtr(), knn_ind,
                d_body, d_left_arm, d_right_arm,
                m_knn_body.ptr(), m_knn_left_arm.ptr(), m_knn_right_arm.ptr(),
                m_right_vert.ptr()
        );
    }
}

void surfelwarp::VolumetricOptimization::ComputeJacobian(
        DeviceArrayView<float4> &live_vertex,
        DeviceArray<int> &knn_ind,
        float *r,
        float *j,
        cudaStream_t stream
) {
    auto vertJac = DeviceArray<float3>(72 * VERTEX_NUM);
    m_smpl_handler->countPoseJac(vertJac, stream);

    auto residual = DeviceArray<float>(6891);
    auto gradient = DeviceArray<float>(6891 * 72);
    cudaMemset(residual.ptr(), 0, sizeof(float) * 6891);
    cudaMemset(gradient.ptr(), 0, sizeof(float) * 6891 * 72);

    dim3 blk(128);
    dim3 grid(divUp(knn_ind.size(), blk.x));
    device::computeJacobianKernel2<<<grid, blk, 0, stream>>>(
            live_vertex.RawPtr(),
            m_smpl_handler->GetVertices().RawPtr(),
            vertJac.ptr(),
            knn_ind,
            //The output
            gradient.ptr(),
            residual.ptr()
    );
    residual.download(r);
    gradient.download(j);
}