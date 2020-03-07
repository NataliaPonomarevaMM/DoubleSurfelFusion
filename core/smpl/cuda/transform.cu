#include <cmath>
#include "core/smpl/def.cuh"
#include "core/smpl/smpl.h"
#include "common/common_types.h"
#include <device_launch_parameters.h>

namespace surfelwarp {
    namespace device {
        __global__ void LocalTransform(
                const PtrSz<const float> joints,
                const PtrSz<const float>poseRotation,
                PtrSz<float>localTransformations
        ) {
            int i = threadIdx.x;
            if (i * 16 + 16 >= localTransformations.size)
                return;
            //copy data from poseRotation
            for (int k = 0; k < 3; k++)
                for (int l = 0; l < 3; l++)
                    localTransformations[i * 16 + k * 4 + l] = poseRotation[i * 9 + k * 3 + l];
            for (int l = 0; l < 3; l++)
                localTransformations[i * 16 + 3 * 4 + l] = 0;
            // data from joints
            int ancestor = m__kinematicTree[i];
            for (int k = 0; k < 3; k++)
                localTransformations[i * 16 + k * 4 + 3] = i != 0 ? joints[i * 3 + k] - joints[ancestor * 3 + k] : joints[k];
            localTransformations[i * 16 + 3 * 4 + 3] = 1;
        }


        __global__ void GlobalTransform(
                const PtrSz<const float> localTransformations,
                const int jointnum,
                PtrSz<float>globalTransformations
        ) {
            for (int k = 0; k < 4; k++)
                for (int l = 0; l < 4; l++)
                    globalTransformations[k * 4 + l] = localTransformations[k * 4 + l];

            for (int j = 1; j < jointnum; j++) {
                int anc = m__kinematicTree[j];
                for (int k = 0; k < 4; k++)
                    for (int l = 0; l < 4; l++) {
                        globalTransformations[j * 16 + k * 4 + l] = 0;
                        for (int t = 0; t < 4; t++)
                            globalTransformations[j * 16 + k * 4 + l] +=
                                    globalTransformations[anc * 16 + k * 4 + t] *
                                    localTransformations[j * 16 + t * 4 + l];
                    }
            }
        }

        __global__ void Transform(
                const PtrSz<const float> joints,
                PtrSz<float> globalTransformations
        ) {
            int j = threadIdx.x;
            if (j * 16 + 16 >= globalTransformations.size)
                return;

            float elim[3];
            for (int k = 0; k < 3; k++) {
                elim[k] = 0;
                for (int t = 0; t < 3; t++)
                    elim[k] += globalTransformations[j * 16 + k * 4 + t] * joints[j * 3 + t];
            }
            for (int k = 0; k < 3; k++)
                globalTransformations[j * 16 + k * 4 + 3] -= elim[k];
        }
    }

    void SMPL::transform(
            const DeviceArray<float> &poseRotation,
            const DeviceArray<float> &joints,
            DeviceArray<float> &globalTransformations,
            cudaStream_t stream
    ) {
        auto localTransformations = DeviceArray<float>(JOINT_NUM * 16);

        device::LocalTransform<<<1,JOINT_NUM,0,stream>>>(joints, poseRotation, localTransformations);
        device::GlobalTransform<<<1,1,0,stream>>>(localTransformations, JOINT_NUM, globalTransformations);
        device::Transform<<<1,JOINT_NUM,0,stream>>>(joints, globalTransformations);
    }
}
