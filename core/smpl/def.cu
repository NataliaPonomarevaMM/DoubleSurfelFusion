#include "core/smpl/def.cuh"

namespace surfelwarp {
    namespace device {
        __device__ float m__poseBlendBasis[6890 * 3 * 207]; // Basis of the pose-dependent shape space, (6890, 3, 207).
        __device__ float m__shapeBlendBasis[6890 * 3 * 10]; // Basis of the shape-dependent shape space, (6890, 3, 10).
        __device__ float m__templateRestShape[6890 * 3]; // Template shape in rest pose, (6890, 3).
        __device__ float m__jointRegressor[24 * 6890]; // Joint coefficients of each vertices for regressing them to joint locations, (24, 6890).
        __device__ int64_t m__kinematicTree[2 * 24]; // Hierarchy relation between joints, the root is at the belly button, (2, 24).
    }

    int64_t vertex_num = 6890;// 6890
    const int64_t joint_num = 24;// 24
    const int64_t shape_basis_dim = 10;// 10
    const int64_t pose_basis_dim = 207;// 207
    const int64_t face_index_num = 13776;// 13776
} // namespace smpl
