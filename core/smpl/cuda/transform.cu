#include <cmath>
#include "core/smpl/def.h"
#include "core/smpl/smpl.h"
#include "common/common_types.h"
#include <device_launch_parameters.h>

namespace surfelwarp {
    namespace device {
        __global__ void LocalTransform(
                const PtrSz<const float> joints,
                const PtrSz<const int64_t> kinematicTree,
                const PtrSz<const float>poseRotation,
                PtrSz<float>localTransformations
        ) {
            int i = threadIdx.x;
            //copy data from poseRotation
            for (int k = 0; k < 3; k++)
                for (int l = 0; l < 3; l++)
                    localTransformations[i * 16 + k * 4 + l] = poseRotation[i * 9 + k * 3 + l];
            for (int l = 0; l < 3; l++)
                localTransformations[i * 16 + 3 * 4 + l] = 0;
            // data from joints
            int ancestor = kinematicTree[i];
            for (int k = 0; k < 3; k++)
                localTransformations[i * 16 + k * 4 + 3] = i != 0 ? joints[i * 3 + k] - joints[ancestor * 3 + k] : joints[k];
            localTransformations[i * 16 + 3 * 4 + 3] = 1;
        }


        __global__ void GlobalTransform(
                const PtrSz<const float> localTransformations,
                const PtrSz<const int64_t>kinematicTree,
                const int jointnum,
                PtrSz<float>globalTransformations
        ) {
            for (int k = 0; k < 4; k++)
                for (int l = 0; l < 4; l++)
                    globalTransformations[k * 4 + l] = localTransformations[k * 4 + l];

            for (int j = 1; j < jointnum; j++) {
                int anc = kinematicTree[j];
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
            const DeviceArray<float> &d_poseRotation,
            const DeviceArray<float> &d_joints,
            DeviceArray<float> &d_globalTransformations
    ) {
        DeviceArray<float> d_localTransformations = DeviceArray<float>(JOINT_NUM * 16);

        device::LocalTransform<<<1,JOINT_NUM>>>(d_joints, d_kinematicTree, d_poseRotation, d_localTransformations);
        device::GlobalTransform<<<1,1>>>(d_localTransformations, d_kinematicTree, JOINT_NUM, d_globalTransformations);
        device::Transform<<<1,JOINT_NUM>>>(d_joints, d_globalTransformations);
    }
}
