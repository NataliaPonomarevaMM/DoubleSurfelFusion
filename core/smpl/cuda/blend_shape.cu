#include <cmath>
#include "core/smpl/def.h"
#include "core/smpl/smpl.h"
#include "common/common_types.h"
#include <device_launch_parameters.h>
#include "math/vector_ops.hpp"

namespace surfelwarp {
    namespace device {
        __global__ void PoseBlend1(
                const PtrSz<const float> theta,
                PtrSz<float> poseRotation,
                PtrSz<float> restPoseRotation
        ) {
            const auto idx = threadIdx.x + blockDim.x * blockIdx.x;
            int ind = idx * 3;

            if (ind + 2 >= theta.size || ind * 3 + 8 >= poseRotation.size)
                return;
            float3 t = make_float3(theta[ind], theta[ind + 1], theta[ind + 2]) + 1e-8;
            float n = norm(t);
            float sin = sinf(n);
            float cos = cosf(n);
            t = t / n;

            float skew[9];
            skew[0] = 0;
            skew[1] = -1 * t.z;
            skew[2] = t.y;
            skew[3] = t.z;
            skew[4] = 0;
            skew[5] = -1 * t.x;
            skew[6] = -1 * t.y;
            skew[7] = t.x;
            skew[8] = 0;

            float outer[9];
            outer[0] = t.x * t.x;
            outer[1] = t.x * t.y;
            outer[2] = t.x * t.z;
            outer[3] = t.y * t.x;
            outer[4] = t.y * t.y;
            outer[5] = t.y * t.z;
            outer[6] = t.z * t.x;
            outer[7] = t.z * t.y;
            outer[8] = t.z * t.z;

            ind = ind * 3;
            for (int p = 0; p < 9; p++)
                poseRotation[ind + p] = 0;
            poseRotation[ind] = cos;
            poseRotation[ind + 4] = cos;
            poseRotation[ind + 8] = cos;
            for (int k = 0; k < 9; k++)
                poseRotation[ind + k] += (1 - cos) * outer[k] + sin * skew[k];// (N, 24, 3, 3)
            for (int k = 0; k < 9; k++)
                restPoseRotation[ind + k] = 0;
            restPoseRotation[ind] = 1;
            restPoseRotation[ind + 4] = 1;
            restPoseRotation[ind + 8] = 1;
        }

        __global__ void PoseBlend2(
                const PtrSz<const float> poseRotation,
                const PtrSz<const float> poseBlendBasis,
                const PtrSz<const float> restPoseRotation,
                PtrSz<float> poseBlendShape
        ) {
            const auto ind = threadIdx.x + blockDim.x * blockIdx.x;
            if (ind >= poseBlendShape.size)
                return;

            poseBlendShape[ind] = 0;
            for (int l = 0; l < 207; l++)
                poseBlendShape[ind] += (poseRotation[l + 9] - restPoseRotation[l + 9]) *
                      poseBlendBasis[ind * 207 + l];
        }

        __global__ void ShapeBlend(
                const PtrSz<const float> beta,
                const PtrSz<const float> shapeBlendBasis,
                const int shapebasisdim,
                PtrSz<float> shapeBlendShape
        ) {
            const auto ind = threadIdx.x + blockDim.x * blockIdx.x;
            if (ind >= shapeBlendShape.size)
                return;
            shapeBlendShape[ind] = 0;
            for (int l = 0; l < shapebasisdim; l++)
                shapeBlendShape[ind] += beta[l] * shapeBlendBasis[l * 6890 * 3 + ind];// (6890, 3)
        }
    }

    void SMPL::countPoseBlendShape(
            DeviceArray<float> &poseRotation,
            DeviceArray<float> &restPoseRotation,
            DeviceArray<float> &poseBlendShape,
            cudaStream_t stream) {
        device::PoseBlend1<<<1,JOINT_NUM,0,stream>>>(m__theta, poseRotation, restPoseRotation);
        dim3 blk(128);
        dim3 grid(divUp(VERTEX_NUM * 3, blk.x));
	    device::PoseBlend2<<<grid,blk,0,stream>>>(poseRotation, m__poseBlendBasis, restPoseRotation, poseBlendShape);
    }

    void SMPL::countShapeBlendShape(
            DeviceArray<float> &shapeBlendShape,
            cudaStream_t stream) {
        dim3 blk(128);
        dim3 grid(divUp(VERTEX_NUM * 3, blk.x));
        device::ShapeBlend<<<grid,blk,0,stream>>>(m__beta, m__shapeBlendBasis, SHAPE_BASIS_DIM, shapeBlendShape);
    }
}
