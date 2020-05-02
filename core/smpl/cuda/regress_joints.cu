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
            if (ind >= joints.size)
                return;

		    const auto j = ind / 3;
		    const auto l = ind % 3;
            joints[ind] = 0;
            for (int k = 0; k < vertexnum; k++)
                joints[ind] += (templateRestShape[k * 3 + l] +
                        shapeBlendShape[k * 3 + l]) * jointRegressor[j * vertexnum + k];
        }
    }

    void SMPL::regressJoints(
            const DeviceArray<float> &shapeBlendShape,
            const DeviceArray<float> &poseBlendShape,
            DeviceArray<float> &joints,
            cudaStream_t stream
    ) {
        dim3 blk(128);
        dim3 grid(divUp(VERTEX_NUM * 3, blk.x));
        device::RegressJoints1<<<grid, blk,0,stream>>>(m__templateRestShape,
                shapeBlendShape, poseBlendShape, m_restShape);
        device::RegressJoints2<<<grid, blk,0,stream>>>(m__templateRestShape,
                shapeBlendShape, m__jointRegressor, VERTEX_NUM, joints);
    }
}
