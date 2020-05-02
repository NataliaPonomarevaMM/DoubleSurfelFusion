#include <cmath>
#include "core/smpl/def.h"
#include "core/smpl/smpl.h"
#include "common/common_types.h"
#include <device_launch_parameters.h>

namespace surfelwarp {
    namespace device {
        __global__ void Skinning(
                const PtrSz<const float> restShape,
                const PtrSz<const float> transformation,
                const PtrSz<const float> weights,
                const int jointnum,
                PtrSz<float3> vertices
        ) {
            const auto j = threadIdx.x + blockDim.x * blockIdx.x;

            if (j >= vertices.size)
                return;

            float coeffs[16] = {0};
            for (int i = 0; i < 16; i++)
                coeffs[i] = 0;

            for (int k = 0; k < 4; k++)
                for (int l = 0; l < 4; l++)
                    for (int t = 0; t < jointnum; t++)
                        coeffs[k * 4 + l] += weights[j * jointnum + t] * transformation[t * 16 + k * 4 + l];

            float homoW = coeffs[15];
            for (int t = 0; t < 3; t++)
                homoW += coeffs[12 + t] * restShape[j * 3 + t];

            float vert[3];
            for (int k = 0; k < 3; k++) {
                vert[k] = coeffs[k * 4 + 3];
                for (int t = 0; t < 3; t++)
                    vert[k] += coeffs[k * 4 + t] * restShape[j * 3 + t];
                vert[k] /= homoW;
            }
            vertices[j] = make_float3(vert[0], vert[1], vert[2]);
        }
    }

    void SMPL::skinning(
            const DeviceArray<float> &transformation,
            cudaStream_t stream
    ) {
        dim3 blk(128);
        dim3 grid(divUp(VERTEX_NUM, blk.x));
        device::Skinning<<<grid,blk,0,stream>>>(m_restShape, transformation,  m__weights,
               JOINT_NUM, m_smpl_vertices);
    }
}
