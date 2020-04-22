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

namespace surfelwarp {
    namespace device {
        __global__ void find_closest_live_vertex(
                const DeviceArrayView<float4> live_vertex,
                const PtrSz<const float3> smpl_vertices,
                PtrSz<ushort2> pairs
        ) {
            int i = blockIdx.x;
            if (i >= smpl_vertices.size)
                return;

            auto first = make_float3(live_vertex[0].x, live_vertex[0].y, live_vertex[0].z);
            float min_dist = norm(first - smpl_vertices[0]);
            ushort ind = 0;
            for (ushort j = 1; j < live_vertex.Size(); j++) {
                float3 refv = make_float3(live_vertex[j].x, live_vertex[j].y, live_vertex[j].z);
                float cur_dist = norm(refv - smpl_vertices[i]);
                if (cur_dist < min_dist) {
                    min_dist = cur_dist;
                    ind = j;
                }
            }
            pairs[i] = make_ushort2(i, ind);
        }

        __global__ void computeJacobianKernel(
                const float4 *live_vertex,
                const float4 *live_normal,
                const float3 *smpl_vertices,
                DeviceArrayView<ushort2> pairs,
                const float *beta0,
                const float *beta,
                const float *theta0,
                const float *theta,
                const float *restshape,
                const float *dbeta,
                //Output
                float *gradient,
                float *residual
        ) {
            const auto idx = threadIdx.x + blockDim.x * blockIdx.x;
            if (idx >= 6892)
                return;
            if (idx == 6891) {
                residual[idx] = 0;
                for (int i = 10; i < 82; i++)
                    residual[idx] += (theta[i] - theta0[i]) * (theta[i] - theta0[i]);
                residual[idx] = sqrt(residual[idx]);
                for (int i = 0; i < 10; i++)
                    gradient[idx * 82 + i] = 0;
                for (int i = 10; i < 82; i++)
                    gradient[idx * 82 + i] = 1;
                return;
            }
            if (idx == 6890) {
                residual[idx] = 0;
                for (int i = 0; i < 10; i++)
                    residual[idx] += (beta[i] - beta0[i]) * (beta[i] - beta0[i]);
                residual[idx] = sqrt(residual[idx]);
                for (int i = 0; i < 10; i++)
                    gradient[idx * 82 + i] = 1;
                for (int i = 10; i < 82; i++)
                    gradient[idx * 82 + i] = 0;
                return;
            }

            const auto cur_pair = pairs[idx];
            const auto smpl = smpl_vertices[cur_pair.x];
            const auto lv = live_vertex[cur_pair.y];
            const auto ln = live_normal[cur_pair.y];

            residual[idx] = dot(ln, smpl - lv);
            // beta
            for (int i = 0; i < 10; i++)
                gradient[idx * 82 + i] = 0;
            // theta
            for (int i = 10; i < 82; i++)
                gradient[idx * 82 + i] = dot(ln, smpl) * restshape[];
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
        mat34 world2camera,
        cudaStream_t stream
) {
    auto poseRotation = DeviceArray<float>(JOINT_NUM * 9);
    auto restPoseRotation = DeviceArray<float>(JOINT_NUM * 9);
    auto poseBlendShape = DeviceArray<float>(VERTEX_NUM * 3);
    auto shapeBlendShape = DeviceArray<float>(VERTEX_NUM * 3);
    auto joints = DeviceArray<float>(JOINT_NUM * 3);
    auto globalTransformations = DeviceArray<float>(JOINT_NUM * 16);
    auto dbeta = DeviceArray<float>(VERTEX_NUM * SHAPE_BASIS_DIM);

    m_restShape = DeviceArray<float>(VERTEX_NUM * 3);
    m_smpl_vertices = DeviceArray<float3>(VERTEX_NUM);

    countPoseBlendShape(poseRotation, restPoseRotation, poseBlendShape, stream);
    countShapeBlendShape(shapeBlendShape, stream);
    regressJoints(shapeBlendShape, poseBlendShape, joints, stream);
    transform(poseRotation, joints, globalTransformations, stream);
    skinning(globalTransformations, stream);
    jacobian_beta(globalTransformations, dbeta, stream);
    CameraTransform(world2camera);

    auto residual = DeviceArray<float>(6892);
    auto gradient = DeviceArray<float>(6892 * 82);

    auto pairs = DeviceArray<ushort2>(m_smpl_vertices.size());
    device::find_closest_live_vertex<<<m_smpl_vertices.size(),1,0,stream>>>(
            live_vertex, m_smpl_vertices, pairs);

    dim3 blk(128);
    dim3 grid(divUp(6892, blk.x));
    device::computeJacobianKernel<<<grid, blk, 0, stream>>>(
            live_vertex.RawPtr(),
            live_normal.RawPtr(),
            m_smpl_vertices.RawPtr(),
            m_pairs,
            m__beta.ptr(),
            beta0.RawPtr(),
            m__theta.ptr(),
            theta0.RawPtr(),
            m_restShape.ptr(),
            dbeta.ptr(),
            //The output
            gradient.ptr(),
            residual.ptr()
    );
    residual.download(r);
    gradient.download(j);
}