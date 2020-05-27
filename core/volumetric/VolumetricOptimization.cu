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
        __global__ void computeResidualKernel(
                const float4 *live_vertex,
                const float3 *smpl_vertices,
                const PtrSz<const int> knn_ind,
                //Output
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
            }
        }

        __global__ void computeJacobianKernel(
                const float4 *live_vertex,
                const float3 *smpl_vertices,
                const float3 *smpl_vertices_jac,
                const PtrSz<const int> knn_ind,
                //Output
                float *gradient
        ) {
            const auto idx = threadIdx.x + blockDim.x * blockIdx.x;
            if (idx >= knn_ind.size)
                return;

            if (knn_ind[idx] != -1) {
                const auto smpl = smpl_vertices[idx];
                const auto lv4 = live_vertex[knn_ind[idx]];
                const auto lv = make_float3(lv4.x, lv4.y, lv4.z);

                for (int i = 0; i < 72; i++)
                    gradient[i * 6891 + idx] = (smpl.x - lv.x) * smpl_vertices_jac[idx * 72 + i].x +
                                               (smpl.y - lv.y) * smpl_vertices_jac[idx * 72 + i].y +
                                               (smpl.z - lv.z) * smpl_vertices_jac[idx * 72 + i].z;
            }
        }

        __global__ void copy_right(
                const float4 *live_vertex,
                PtrSz<int> knn,
                const PtrSz<const float> dist,
                float4 *right
        ) {
            const auto idx = threadIdx.x + blockDim.x * blockIdx.x;
            if (idx > knn.size)
                return;
            right[idx] = live_vertex[knn[idx]];
            if (dist[idx] > 0.03) {
                knn[idx] = -1;
            }
        }

        __global__ void check_right(
                const float4 *live_vertex,
                const PtrSz<const int> knn,
                const float4 *right,
                bool *ans
        ) {
            const auto idx = threadIdx.x + blockDim.x * blockIdx.x;
            if (idx > knn.size)
                return;

            if (knn[idx] != -1 && (
                   abs(right[idx].x - live_vertex[knn[idx]].x) > 0.03 ||
                   abs(right[idx].y - live_vertex[knn[idx]].y) > 0.03 ||
                   abs(right[idx].z - live_vertex[knn[idx]].z) > 0.03
                    ))
                *ans = true;
        }
    }
}

void surfelwarp::VolumetricOptimization::Initialize(
        SMPL::Ptr smpl_handler,
        DeviceArrayView<float4> &live_vertex,
        cudaStream_t stream
) {
    m_smpl_handler = smpl_handler;

    m_knn = DeviceArray<int>(VERTEX_NUM);
    auto dist = DeviceArray<float>(VERTEX_NUM);
    m_smpl_handler->count_dist_to_smpl(live_vertex, m_knn, dist, 0);
    m_right_vert = DeviceArray<float4>(VERTEX_NUM);

    dim3 blk(128);
    dim3 grid(divUp(m_knn.size(), blk.x));
    device::copy_right<<<grid, blk, 0, stream>>>(
            live_vertex.RawPtr(),
            m_knn,
            dist,
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
            m_knn,
            m_right_vert.ptr(),
            d_answer
    );
    cudaMemcpy(&h_answer, d_answer, sizeof(bool), cudaMemcpyDeviceToHost);
    cudaFree(d_answer);
    if (h_answer) {
        m_knn = DeviceArray<int>(VERTEX_NUM);
        auto dist = DeviceArray<float>(VERTEX_NUM);
        m_smpl_handler->count_dist_to_smpl(live_vertex, m_knn, dist, 0);
        device::copy_right<<<grid, blk, 0, stream>>>(
                live_vertex.RawPtr(), m_knn, dist,
                m_right_vert.ptr()
        );
    }
}

void surfelwarp::VolumetricOptimization::ComputeJacobian(
        DeviceArrayView<float4> &live_vertex,
        float *j,
        cudaStream_t stream
) {
    auto vertJac = DeviceArray<float3>(72 * VERTEX_NUM);
    m_smpl_handler->countPoseJac(vertJac, stream);
    auto gradient = DeviceArray<float>(6891 * 72);
    cudaMemset(gradient.ptr(), 0, sizeof(float) * 6891 * 72);

    dim3 blk(128);
    dim3 grid(divUp(m_knn.size(), blk.x));
    device::computeJacobianKernel<<<grid, blk, 0, stream>>>(
            live_vertex.RawPtr(),
            m_smpl_handler->GetVertices().RawPtr(),
            vertJac.ptr(),
            m_knn,
            //The output
            gradient.ptr()
    );
    gradient.download(j);
}

void surfelwarp::VolumetricOptimization::ComputeResidual(
        DeviceArrayView<float4> &live_vertex,
        float *r,
        cudaStream_t stream
) {
    auto residual = DeviceArray<float>(6891);
    cudaMemset(residual.ptr(), 0, sizeof(float) * 6891);
    dim3 blk(128);
    dim3 grid(divUp(m_knn.size(), blk.x));
    device::computeResidualKernel<<<grid, blk, 0, stream>>>(
            live_vertex.RawPtr(),
            m_smpl_handler->GetVertices().RawPtr(),
            m_knn,
            //The output
            residual.ptr()
    );
    residual.download(r);
}