#include <cmath>
#include "core/smpl/def.h"
#include "core/smpl/smpl.h"
#include "common/common_types.h"
#include <device_launch_parameters.h>
#include "math/vector_ops.hpp"

namespace surfelwarp {
    namespace device {
        __global__ void countPoseRotJacobian(
                const PtrSz<const float> theta,
                PtrSz<float> poseRotationJac
        ) {
            const auto idx = threadIdx.x + blockDim.x * blockIdx.x; // 0..72
            if (idx >= theta.size)
                return;
            const int ind = idx / 3;
            float3 t = make_float3(theta[ind * 3], theta[ind * 3 + 1], theta[ind * 3 + 2]) + 1e-8;
            float n = norm(t);
            float sin = sinf(n);
            float cos = cosf(n);
            t = t / n;

            const auto cur_t = idx % 3 == 0 ? t.x : (idx % 3 == 1 ? t.y : t.z);
            const auto id1 = idx % 3 == 0 ? 5 : (idx % 3 == 1 ? 6 : 1);
            const auto id2 = idx % 3 == 0 ? 7 : (idx % 3 == 1 ? 2 : 3);

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

            float skew_jac[9];
            for (int i = 0; i < 9; i ++)
                skew_jac[i] = skew[i] * (-1) * cur_t / n;
            skew_jac[id1] = -1 * (1 - cur_t * cur_t) / n;
            skew_jac[id2] = (1 - cur_t * cur_t) / n;

            float outer_jac[9];
            for (int i = 0; i < 9; i ++)
                outer_jac[i] = outer[i] * (-2) * cur_t / n;
            outer_jac[(idx % 3) * 3 + (idx % 3)] += 2 * cur_t / n;
            const auto id3 = idx % 3 == 0 ? 1 : 0;
            const auto id4 = idx % 3 == 0 ? 2 : (idx % 3 == 1 ? 2 : 1);
            const auto t3 = idx % 3 == 0 ? t.y : t.x;
            const auto t4 = idx % 3 == 0 ? t.z : (idx % 3 == 1 ? t.z : t.y);
            outer_jac[(idx % 3) * 3 + id3] += t3 / n;
            outer_jac[id3 * 3 + (idx % 3)] += t3 / n;
            outer_jac[(idx % 3) * 3 + id4] += t4 / n;
            outer_jac[id4 * 3 + (idx % 3)] += t4 / n;

            if (idx == 0) {
                for (int i = 0; i < 9; i++)
                    printf("%f ", outer[i]);
                printf("\n");
                for (int i = 0; i < 9; i++)
                    printf("%f ", outer_jac[i]);
                printf("\n");
                for (int i = 0; i < 9; i++)
                    printf("%f ", skew[i]);
                printf("\n");
                for (int i = 0; i < 9; i++)
                    printf("%f ", skew_jac[i]);
                printf("\n");
            }

            for (int k = 0; k < 9; k++)
                poseRotationJac[idx * 9 + k] = sin * cur_t * outer[k] + (1 - cos) * outer_jac[k]
                        + cos * cur_t * skew[k] + sin * skew_jac[k];
        }

        __global__ void LocalTransformJac(
                const PtrSz<const float> joints,
                const PtrSz<const int64_t> kinematicTree,
                const PtrSz<const float> poseRotationJac,
                PtrSz<float>localTransformations
        ) {
            const int idx = threadIdx.x; //0..72
            const int i = idx / 3;
            if (idx >= 72)
                return;
            //copy data from poseRotation
            for (int k = 0; k < 3; k++)
                for (int l = 0; l < 3; l++)
                    localTransformations[idx * 16 + k * 4 + l] = poseRotationJac[idx * 9 + k * 3 + l];
            for (int l = 0; l < 3; l++)
                localTransformations[idx * 16 + 3 * 4 + l] = 0;
            // data from joints
            int ancestor = kinematicTree[i];
            for (int k = 0; k < 3; k++)
                localTransformations[idx * 16 + k * 4 + 3] = i != 0 ? joints[i * 3 + k] - joints[ancestor * 3 + k] : joints[k];
            localTransformations[idx * 16 + 3 * 4 + 3] = 1;
        }


        __global__ void GlobalTransformJac(
                const PtrSz<const float> localTransformations,
                const PtrSz<const float> localTransformationsJac,
                const PtrSz<const int64_t> kinematicTree,
                const PtrSz<const float> joints,
                const int jointnum,
                PtrSz<float> globalTransformations
        ) {
            const int idx = threadIdx.x; //0..72
            const int i = idx / 3;

            bool not_null[24];

            not_null[0] = (i == 0);
            for (int k = 0; k < 4; k++)
                for (int l = 0; l < 4; l++)
                    globalTransformations[idx * 384 + k * 4 + l] =
                            i == 0 ? localTransformationsJac[idx * 16 + k * 4 + l] :
                            localTransformations[k * 4 + l];

            for (int j = 1; j < jointnum; j++) {
                int anc = kinematicTree[j];
                not_null[j] = (not_null[anc] || i == j);
                for (int k = 0; k < 4; k++)
                    for (int l = 0; l < 4; l++) {
                        globalTransformations[idx * 384 + j * 16 + k * 4 + l] = 0;
                        for (int t = 0; t < 4; t++)
                            globalTransformations[idx * 384 + j * 16 + k * 4 + l] +=
                                    globalTransformations[idx * 384 + anc * 16 + k * 4 + t] *
                                        (i == j ? localTransformationsJac[idx * 16 + t * 4 + l] :
                                        localTransformations[j * 16 + t * 4 + l]);
                    }
            }
            for (int j = 0; j < jointnum; j++) {
                int anc = kinematicTree[j];
                if (!not_null[j]) {
                    for (int k = 0; k < 3; k++)
                        for (int l = 0; l < 3; l++)
                            globalTransformations[idx * 384 + j * 16 + k * 4 + l] = 0;
                    for (int k = 0; k < 3; k++)
                        globalTransformations[idx * 384 + j * 16 + k * 4 + 3] =
                                j != 0 ? joints[j * 3 + k] - joints[anc * 3 + k] : joints[k];
                }
            }
        }

        __global__ void TransformJac(
                const PtrSz<const float> joints,
                PtrSz<float> globalTransformations
        ) {
            int j = blockIdx.x; //0..24
            int i = threadIdx.x; // 0..72
            if (i >= 72 || j * 16 + 15 >= globalTransformations.size)
                return;

            float elim[3];
            for (int k = 0; k < 3; k++) {
                elim[k] = 0;
                for (int t = 0; t < 3; t++)
                    elim[k] += globalTransformations[i * 384 + j * 16 + k * 4 + t] * joints[j * 3 + t];
            }
            for (int k = 0; k < 3; k++)
                globalTransformations[i * 384 + j * 16 + k * 4 + 3] -= elim[k];
        }

        __global__ void SkinningJac(
                const PtrSz<const float> templateRestShape,
                const PtrSz<const float> shapeBlendShape,
                const PtrSz<const float> transformationJac,
                const PtrSz<const float> weights,
                const int jointnum,
                PtrSz<float3> verticesJac
        ) {
            const auto j = blockIdx.x;
            const auto i = threadIdx.x;
            if (i >= 72 || j >= verticesJac.size)
                return;

            float coeffs[16] = {0};
            for (int k = 0; k < 16; k++)
                coeffs[k] = 0;

            for (int k = 0; k < 4; k++)
                for (int l = 0; l < 4; l++)
                    for (int t = 0; t < jointnum; t++)
                        coeffs[k * 4 + l] += weights[j * jointnum + t] *
                                transformationJac[i * 384 + t * 16 + k * 4 + l];

            float homoW = coeffs[15];
            for (int t = 0; t < 3; t++)
                homoW += coeffs[12 + t] * (templateRestShape[j * 3 + t] + shapeBlendShape[j * 3 + t]);

            float vert[3];
            for (int k = 0; k < 3; k++) {
                vert[k] = coeffs[k * 4 + 3];
                for (int t = 0; t < 3; t++)
                    vert[k] += coeffs[k * 4 + t] * (templateRestShape[j * 3 + t] + shapeBlendShape[j * 3 + t]);
                vert[k] /= homoW;
            }
            verticesJac[j * 72 + i] = make_float3(vert[0], vert[1], vert[2]);
        }
    }

    void SMPL::countPoseJac(
            DeviceArray<float3> &vertJac,
            cudaStream_t stream) {
        auto poseRotation = DeviceArray<float>(JOINT_NUM * 9);
        auto restPoseRotation = DeviceArray<float>(JOINT_NUM * 9);
        auto poseBlendShape = DeviceArray<float>(VERTEX_NUM * 3);
        auto shapeBlendShape = DeviceArray<float>(VERTEX_NUM * 3);
        auto joints = DeviceArray<float>(JOINT_NUM * 3);
        auto globalTransformations = DeviceArray<float>(JOINT_NUM * 16);
        auto localTransformations = DeviceArray<float>(JOINT_NUM * 16);

        m_restShape = DeviceArray<float>(VERTEX_NUM * 3);
        m_smpl_vertices = DeviceArray<float3>(VERTEX_NUM);

        countPoseBlendShape(poseRotation, restPoseRotation, poseBlendShape, stream);
        countShapeBlendShape(shapeBlendShape, stream);
        countRestShape(shapeBlendShape, poseBlendShape, stream);
        regressJoints(shapeBlendShape, joints, stream);
        transform(poseRotation, joints, globalTransformations, localTransformations, stream);
        skinning(globalTransformations, stream);
        countNormals(stream);
        transform(stream);

        auto poseRotJac = DeviceArray<float>(72 * 9);
        device::countPoseRotJacobian<<<1,72,0,stream>>>(m__theta, poseRotJac);
        auto localTransformationsJac = DeviceArray<float>(72 * 16);
        device::LocalTransformJac<<<1,72,0,stream>>>(joints, m__kinematicTree,
                poseRotJac, localTransformationsJac);
        auto globalTransformationsJac = DeviceArray<float>(72 * JOINT_NUM * 16);
        device::GlobalTransformJac<<<1,72,0,stream>>>(localTransformations, localTransformationsJac,
                m__kinematicTree, joints, JOINT_NUM, globalTransformationsJac);
        device::TransformJac<<<JOINT_NUM,72,0,stream>>>(joints, globalTransformationsJac);
        device::SkinningJac<<<VERTEX_NUM,72,0,stream>>>(m__templateRestShape, shapeBlendShape,
                globalTransformationsJac,  m__weights, JOINT_NUM, vertJac);
        cudaStreamSynchronize(stream);
        auto error = cudaGetLastError();
        if(error != cudaSuccess)
        {
            printf("CUDA error in jac: %s\n", cudaGetErrorString(error));
            exit(-1);
        }

//        std::ofstream f("/home/nponomareva/surfelwarp/posejac.txt");
//        std::vector<float> q;
//        globalTransformationsJac.download(q);
//        for (int i = 0; i < q.size(); i++) {
//            f << q[i] << "\n";
//        }
////        std::vector<float3> q;
////        vertJac.download(q);
////        for (int i = 0; i < q.size(); i++) {
////            f << q[i].x << " " << q[i].y << " " << q[i].z << "\n";
////        }
//        std::cout << "saved jac\n";
    }
}
