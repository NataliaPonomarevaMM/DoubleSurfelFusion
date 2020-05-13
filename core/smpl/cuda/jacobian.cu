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

namespace surfelwarp {
    namespace device {
        __global__ void computeJacobianKernel1(
                const float *beta0,
                const float *beta,
                const float *theta0,
                const float *theta,
                //Output
                float *gradient,
                float *residual
        ) {
            // for theta
            residual[6891] = 0;
            for (int i = 10; i < 82; i++)
                residual[6891] += (theta[i] - theta0[i]) * (theta[i] - theta0[i]);
            residual[6891] = sqrt(residual[6891]);
            for (int i = 0; i < 10; i++)
                gradient[6891 * 82 + i] = 0;
            for (int i = 10; i < 82; i++)
                gradient[6891 * 82 + i] = 1;

            // for beta
            residual[6890] = 0;
            for (int i = 0; i < 10; i++)
                residual[6890] += (beta[i] - beta0[i]) * (beta[i] - beta0[i]);
            residual[6890] = sqrt(residual[6890]);
            for (int i = 0; i < 10; i++)
                gradient[6890 * 82 + i] = 1;
            for (int i = 10; i < 82; i++)
                gradient[6890 * 82 + i] = 0;
        }

        __global__ void computeJacobianKernel2(
                const float4 *live_vertex,
                const float4 *live_normal,
                const float3 *smpl_vertices,
                const float3 *smpl_vertices_jac,
                DeviceArrayView<ushort2> pairs,
                //Output
                float *gradient,
                float *residual
        ) {
            const auto idx = threadIdx.x + blockDim.x * blockIdx.x;
            if (idx >= pairs.Size())
                return;
            const auto cur_pair = pairs[idx];
            const auto smpl = smpl_vertices[cur_pair.x];
            const auto lv4 = live_vertex[cur_pair.y];
            const auto ln4 = live_normal[cur_pair.y];
            const auto lv = make_float3(lv4.x, lv4.y, lv4.z);
            const auto ln = make_float3(ln4.x, ln4.y, ln4.z);
            const auto smpl_jac = smpl_vertices_jac[cur_pair.x];

            residual[cur_pair.x] = dot(ln, smpl - lv);
            // beta
            for (int i = 0; i < 10; i++)
                gradient[cur_pair.x * 82 + i] = 0;
            // theta
            for (int i = 10; i < 82; i++)
                gradient[cur_pair.x * 82 + i] = dot(ln, smpl_jac);
        }

        __global__ void addBetaTheta(
                const DeviceArrayView<float> to_add,
                float *beta,
                float *theta
        ) {
            const auto idx = threadIdx.x + blockDim.x * blockIdx.x;
            if (idx > to_add.Size())
                return;
            if (idx < 10)
                beta[idx] += to_add[idx];
            else
                theta[idx - 10] += to_add[idx];
        }
    }
}

void surfelwarp::SMPL::ComputeJacobian(
        DeviceArrayView<float4> &live_vertex,
        DeviceArrayView<float4> &live_normal,
        DeviceArrayView<float> &beta0,
        DeviceArrayView<float> &theta0,
        float *r,
        float *j,
        cudaStream_t stream
) {
    auto vertJac = DeviceArray<float3>(72 * VERTEX_NUM);
    countPoseJac(vertJac, stream);
    //CountKnn(live_vertex, stream);

    SynchronizeArray<ushort2> pairs;
    pairs.AllocateBuffer(6890);
    m_pair_sorting->PerformSorting(m_knn.ArrayView(), m_dist.ArrayView(), pairs);

//    std::ofstream f("/home/nponomareva/surfelwarp/pairs.obj");
//
//    std::vector<float3> q1;
//    m_smpl_vertices.download(q1);
//    std::vector<float4> q2;
//    live_vertex.Download(q2);
//    for (auto i = 0; i < q1.size(); i++) {
//        f << 'v' << ' '
//                    << q1[i].x << ' '
//                    << q1[i].y << ' '
//                    << q1[i].z << std::endl;
//        f << 'v' << ' '
//          << q1[i].x + 0.0001<< ' '
//          << q1[i].y + 0.0001 << ' '
//          << q1[i].z + 0.0001 << std::endl;
//    }
//    for (auto i = 0; i < q2.size(); i++) {
//        f << 'v' << ' '
//          << q2[i].x << ' '
//          << q2[i].y << ' '
//          << q2[i].z << std::endl;
//    }
//    auto q = pairs.HostArray();
//    for (auto i = 0; i < q.size(); i++) {
//        f << "f " << q[i].x * 2 << " " << q[i].x * 2 + 1 << " " << q[i].y + 6890 * 2  << "\n";
//    }
//    f.close();
//    std::cout << "pairs jac\n";

    auto residual = DeviceArray<float>(6892);
    auto gradient = DeviceArray<float>(6892 * 82);

    cudaMemset(residual.ptr(), 0, sizeof(float) * 6892);
    cudaMemset(gradient.ptr(), 0, sizeof(float) * 6892 * 82);
    cudaStreamSynchronize(stream);

    dim3 blk(128);
    dim3 grid(divUp(pairs.DeviceArraySize(), blk.x));
    device::computeJacobianKernel2<<<grid, blk, 0, stream>>>(
            live_vertex.RawPtr(),
            live_normal.RawPtr(),
            m_smpl_vertices.ptr(),
            vertJac.ptr(),
            pairs.DeviceArrayReadOnly(),
            //The output
            gradient.ptr(),
            residual.ptr()
    );
    device::computeJacobianKernel1<<<1, 1, 0, stream>>>(
            m__beta.ptr(),
            beta0.RawPtr(),
            m__theta.ptr(),
            theta0.RawPtr(),
            //The output
            gradient.ptr(),
            residual.ptr()
    );
    cudaStreamSynchronize(stream);
    auto error = cudaGetLastError();
    if(error != cudaSuccess)
    {
        // print the CUDA error message and exit
        printf("2)CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }
    residual.download(r);
    gradient.download(j);
}

void surfelwarp::SMPL::AddBetaTheta(
        DeviceArrayView<float> &to_add,
        cudaStream_t stream
) {
    device::addBetaTheta<<<1, 82, 0, stream>>>(to_add, m__beta.ptr(), m__theta.ptr());
}