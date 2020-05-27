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

namespace surfelwarp {
    namespace device {
        __global__ void addTheta(
                const PtrSz<const float> to_add,
                float *theta
        ) {
            const auto idx = threadIdx.x + blockDim.x * blockIdx.x;
            if (idx > to_add.size)
                return;
            theta[idx] += to_add[idx];
        }

        __global__ void minTheta(
                const PtrSz<const float> to_add,
                float *theta
        ) {
            const auto idx = threadIdx.x + blockDim.x * blockIdx.x;
            if (idx > to_add.size)
                return;
            theta[idx] -= to_add[idx];
        }
    }
}

void surfelwarp::SMPL::AddTheta(
        std::vector<float> &to_add,
        cudaStream_t stream
) {
    DeviceArray<float> new_theta;
    new_theta.upload(to_add);
    device::addTheta<<<1, 72, 0, stream>>>(new_theta, m__theta.ptr());
    cudaStreamSynchronize(stream);
}

void surfelwarp::SMPL::SubTheta(
        std::vector<float> &to_sub,
        cudaStream_t stream
) {
    DeviceArray<float> new_theta;
    new_theta.upload(to_sub);
    device::minTheta<<<1, 72, 0, stream>>>(new_theta, m__theta.ptr());
    cudaStreamSynchronize(stream);
}