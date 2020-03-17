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
                PtrSz<float> vertices
        ) {
            int j = blockIdx.x;

            if (j * 3 + 3 >= vertices.size)
                return;

            float coeffs[16] = {0};
            for (int k = 0; k < 4; k++)
                for (int l = 0; l < 4; l++)
                    for (int t = 0; t < jointnum; t++)
                        coeffs[k * 4 + l] += weights[j * jointnum + t] * transformation[t * 16 + k * 4 + l];

            float homoW = coeffs[15];
            for (int t = 0; t < 3; t++)
                homoW += coeffs[12 + t] * restShape[j * 3 + t];
            for (int k = 0; k < 3; k++) {
                vertices[j * 3 + k] = coeffs[k * 4 + 3];
                for (int t = 0; t < 3; t++)
                    vertices[j * 3 + k] += coeffs[k * 4 + t] * restShape[j * 3 + t];
                vertices[j * 3 + k] /= homoW;
            }
        }
    }

    void SMPL::skinning(
            const DeviceArray<float> &transformation,
            cudaStream_t stream
    ) {
        device::Skinning<<<VERTEX_NUM,1,0,stream>>>(m_restShape, transformation,  m__weights,
               JOINT_NUM, m_smpl_vertices);
    }
}
