#include <cmath>
#include "core/smpl/def.cuh"
#include "core/smpl/smpl.h"
#include "common/common_types.h"
#include <device_launch_parameters.h>

namespace surfelwarp {
    namespace device {
        __global__ void RegressJoints1(
                const PtrSz<const float> shapeBlendShape,
                const PtrSz<const float> poseBlendShape,
                PtrSz<float> restShape
        ) {
            const auto ind = threadIdx.x + blockDim.x * blockIdx.x;
            if (ind >= restShape.size)
                return;
            restShape[ind] = m__templateRestShape[ind] + shapeBlendShape[ind] + poseBlendShape[ind];
        }

        __global__ void RegressJoints2(
                const PtrSz<const float> shapeBlendShape,
                const int vertexnum,
                PtrSz<float> joints
        ) {
            const auto ind = threadIdx.x + blockDim.x * blockIdx.x;
            if (ind >= joints.size)
                return;

		 int j = blockIdx.x;
             int l = threadIdx.x;
            joints[ind] = 0;
            for (int k = 0; k < vertexnum; k++)
                joints[ind] += (m__templateRestShape[k * 3 + l] +
                        shapeBlendShape[k * 3 + l]) * m__jointRegressor[j * vertexnum + k];
        }
    }

    void SMPL::regressJoints(
            const DeviceArray<float> &shapeBlendShape,
            const DeviceArray<float> &poseBlendShape,
            DeviceArray<float> &restShape,
            DeviceArray<float> &joints,
            cudaStream_t stream
    ) {
        device::RegressJoints1<<<VERTEX_NUM,3,0,stream>>>(shapeBlendShape, poseBlendShape, restShape);
        device::RegressJoints2<<<JOINT_NUM,3,0,stream>>>(shapeBlendShape, VERTEX_NUM, joints);
    }
}
