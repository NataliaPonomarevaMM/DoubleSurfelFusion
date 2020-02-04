#include <cmath>
#include "core/smpl/def.h"
#include "core/smpl/smpl.h"
#include "common/common_types.h"
#include <device_launch_parameters.h>

namespace surfelwarp {
    namespace device {
        __global__ void RegressJoints1(
                const PtrSz<const float> templateRestShape,
                const PtrSz<const float> shapeBlendShape,
                const PtrSz<const float> poseBlendShape,
                PtrSz<float> restShape
        ) {
            const auto ind = threadIdx.x + blockDim.x * blockIdx.x;
            if (ind >= restShape.size)
                return;
            restShape[ind] = templateRestShape[ind] + shapeBlendShape[ind] + poseBlendShape[ind];
        }

        __global__ void RegressJoints2(
                const PtrSz<const float> templateRestShape,
                const PtrSz<const float> shapeBlendShape,
                const PtrSz<const float> jointRegressor,
                const int vertexnum,
                PtrSz<float> joints
        ) {
            const auto ind = threadIdx.x + blockDim.x * blockIdx.x;
            if (ind >= restShape.size)
                return;

            int l = threadIdx.x;
            joints[ind] = 0;
            for (int k = 0; k < vertexnum; k++)
                joints[ind] += (templateRestShape[k * 3 + l] +
                        shapeBlendShape[k * 3 + l]) * jointRegressor[j * vertexnum + k];
        }
    }

    void SMPL::regressJoints(
            const DeviceArray<float> &d_shapeBlendShape,
            const DeviceArray<float> &d_poseBlendShape,
            DeviceArray<float> &d_restShape,
            DeviceArray<float> &d_joints,
            cudaStream_t stream
    ) {
        device::RegressJoints1<<<VERTEX_NUM,3,0,stream>>>(d_templateRestShape,
                d_shapeBlendShape, d_poseBlendShape, d_restShape);
        device::RegressJoints2<<<JOINT_NUM,3,0,stream>>>(d_templateRestShape,
                d_shapeBlendShape, d_jointRegressor, VERTEX_NUM, d_joints);
    }
}
