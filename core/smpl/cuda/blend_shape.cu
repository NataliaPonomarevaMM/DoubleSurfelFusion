#include <cmath>
#include "core/smpl/def.h"
#include "core/smpl/smpl.h"
#include "common/common_types.h"
#include <device_launch_parameters.h>

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
            float norm = std::sqrt(
                    theta[ind] * theta[ind] + theta[ind + 1] * theta[ind + 1] + theta[ind + 2] * theta[ind + 2]);
            float sin = std::sin(norm);
            float cos = std::cos(norm);
            float t0 = theta[ind] / norm;
            float t1 = theta[ind + 1] / norm;
            float t2 = theta[ind + 2] / norm; // axes

            float skew[9];
            skew[0] = 0;
            skew[1] = -1 * t2;
            skew[2] = t1;
            skew[3] = t2;
            skew[4] = 0;
            skew[5] = -1 * t0;
            skew[6] = -1 * t1;
            skew[7] = t0;
            skew[8] = 0;

            ind = ind * 3;
            for (int p = 0; p < 0; p++)
                poseRotation[ind + p] = 0;
            poseRotation[ind] = 1;
            poseRotation[ind + 4] = 1;
            poseRotation[ind + 8] = 1;
            for (int k1 = 0; k1 < 3; k1++)
                for (int k2 = 0; k2 < 3; k2++) {
                    int k = k1 * 3 + k2;
                    poseRotation[ind + k] += skew[k] * sin;
                    float num = 0;
                    for (int l = 0; l < 3; l++)
                        num += skew[k1 * 3 + l] * skew[l * 3 + k2];
                    poseRotation[ind + k] += (1 - cos) * num;// (N, 24, 3, 3)
                }

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
	    if (ind >= .size) 
    		return;
            shapeBlendShape[ind] = 0;
            for (int l = 0; l < shapebasisdim; l++)
                shapeBlendShape[ind] += beta[l] * shapeBlendBasis[ind * shapebasisdim + l];// (6890, 3)
        }
    }

    void SMPL::poseBlendShape(
            const DeviceArray<float> &theta,
            DeviceArray<float> &d_poseRotation,
            DeviceArray<float> &d_restPoseRotation,
            DeviceArray<float> &d_poseBlendShape,
            cudaStream_t stream) {
	dim3 blk(1);
	dim3 grid(divUp(JOINT_NUM, blk.x));

        device::PoseBlend1<<<grid, blk,0,stream>>>(theta, d_poseRotation, d_restPoseRotation);
        
	dim3 blk2(3);
        dim3 grid2(divUp(VERTEX_NUM, blk2.x));
	device::PoseBlend2<<<grid2,blk2,0,stream>>>(d_poseRotation, d_poseBlendBasis, d_restPoseRotation, d_poseBlendShape);
    }

    void SMPL::shapeBlendShape(
            const DeviceArray<float> &beta,
            DeviceArray<float> &d_shapeBlendShape,
            cudaStream_t stream) {
	dim3 blk2(3);
        dim3 grid2(divUp(VERTEX_NUM, blk2.x));
        device::ShapeBlend<<<grid2,blk2,0,stream>>>(beta, d_shapeBlendBasis, SHAPE_BASIS_DIM, d_shapeBlendShape);
    }
}
