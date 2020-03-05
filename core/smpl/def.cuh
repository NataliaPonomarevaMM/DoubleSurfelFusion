#ifndef DEF_H
#define DEF_H

#ifndef VERTEX_NUM
#define VERTEX_NUM surfelwarp::vertex_num
#endif // VERTEX_NUM

#ifndef JOINT_NUM
#define JOINT_NUM surfelwarp::joint_num
#endif // JOINT_NUM

#ifndef SHAPE_BASIS_DIM
#define SHAPE_BASIS_DIM surfelwarp::shape_basis_dim
#endif // SHAPE_BASIS_DIM

#ifndef POSE_BASIS_DIM
#define POSE_BASIS_DIM surfelwarp::pose_basis_dim
#endif // POSE_BASIS_DIM

#ifndef FACE_INDEX_NUM
#define FACE_INDEX_NUM surfelwarp::face_index_num
#endif // FACE_INDEX_NUM

#include <cstdlib>

namespace surfelwarp {
    namespace device {
        extern __device__ float m__poseBlendBasis[6890 * 3 * 207]; // Basis of the pose-dependent shape space, (6890, 3, 207).
        extern __device__ float m__shapeBlendBasis[6890 * 3 * 10]; // Basis of the shape-dependent shape space, (6890, 3, 10).
        extern __device__ float m__templateRestShape[6890 * 3]; // Template shape in rest pose, (6890, 3).
        extern __device__ float m__jointRegressor[24 * 6890]; // Joint coefficients of each vertices for regressing them to joint locations, (24, 6890).
        extern __device__ int64_t m__kinematicTree[2 * 24]; // Hierarchy relation between joints, the root is at the belly button, (2, 24).
    }

    extern int64_t vertex_num;// 6890
    extern const int64_t joint_num;// 24
    extern const int64_t shape_basis_dim;// 10
    extern const int64_t pose_basis_dim;// 207
    extern const int64_t face_index_num;// 13776
} // namespace smpl
#endif // DEF_H