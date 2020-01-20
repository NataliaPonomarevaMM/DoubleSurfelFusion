#include <cmath>
#include "core/smpl/def.h"
#include "core/smpl/smpl.h"

namespace surfelwarp {
    namespace device {
        __global__ void Skinning(
                const PtrSz<const float> restShape,
                const PtrSz<const float> transformation,
                const PtrSz<const float> weights,
                const int vertexnum,
                const int jointnum,
                PtrSz<float> vertices
        ) {
            int j = threadIdx.x;

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
            const DeviceArray<float> &d_transformation,
            const DeviceArray<float> &d_custom_weights,
            const DeviceArray<float> &d_vertices,
            DeviceArray<float> &d_result_vertices
    ) {
        device::Skinning<<<1,VERTEX_NUM>>>(d_vertices, d_transformation, d_custom_weights,
                d_vertices.size(), JOINT_NUM, d_result_vertices);
    }
}
