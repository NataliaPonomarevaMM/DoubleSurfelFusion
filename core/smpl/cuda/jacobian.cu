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

            residual[cur_pair.x] = dot(ln, smpl - lv);
            // beta
            for (int i = 0; i < 10; i++)
                gradient[cur_pair.x * 82 + i] = 0;
            // theta
            for (int i = 10; i < 82; i++)
                gradient[cur_pair.x * 82 + i] = dot(ln, smpl);
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
    LbsModel(stream);
    CountKnn(live_vertex, stream);

    SynchronizeArray<ushort2> pairs;
    pairs.AllocateBuffer(6890);
    m_pair_sorting->PerformSorting(m_knn.ArrayView(), m_dist.ArrayView(), pairs);

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