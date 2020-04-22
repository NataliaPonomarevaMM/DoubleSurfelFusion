#ifndef SMPL_H
#define SMPL_H

#include <string>
#include <tuple>
#include <memory>
#include "core/smpl/def.h"
#include "common/common_types.h"
#include "imgproc/ImageProcessor.h"
#include "imgproc/frameio/VolumeDeformFileFetch.h"
#include "core/SurfelGeometry.h"

namespace surfelwarp {
    class SMPL {
    private:
        DeviceArray<float> m__beta;
        DeviceArray<float> m__theta;

        ///constant
        DeviceArray<float> m__poseBlendBasis; // Basis of the pose-dependent shape space, (6890, 3, 207).
        DeviceArray<float> m__shapeBlendBasis; // Basis of the shape-dependent shape space, (6890, 3, 10).
        DeviceArray<float> m__templateRestShape; // Template shape in rest pose, (6890, 3).
        DeviceArray<float> m__jointRegressor; // Joint coefficients of each vertices for regressing them to joint locations, (24, 6890).
        DeviceArray<float> m__weights; // Weights for linear blend skinning, (6890, 24).
        DeviceArray<int64_t> m__kinematicTree; // Hierarchy relation between joints, the root is at the belly button, (2, 24).
        DeviceArray<int> m__faceIndices; // (13776, 3)

        ///useful information to store
        int m_vert_frame = -1;
        DeviceArray<float> m_restShape;
        DeviceArray<float3> m_smpl_vertices;
        DeviceArray<float3> m_smpl_normals;
        DeviceArray<float> m_dist;
        int m_num_marked;
        DeviceArray<bool> m_marked_vertices;
        int m_knn_frame = -1;
        DeviceArray<ushort4> m_knn;
        DeviceArray<float4> m_knn_weight;
        DeviceArray<int> m_onbody;

        void countPoseBlendShape(
                DeviceArray<float> &poseRotation,
                DeviceArray<float> &restPoseRotation,
                DeviceArray<float> &poseBlendShape,
                cudaStream_t stream);
        void countShapeBlendShape(
                DeviceArray<float> &shapeBlendShape,
                cudaStream_t stream);
        void regressJoints(
                const DeviceArray<float> &shapeBlendShape,
                const DeviceArray<float> &poseBlendShape,
                DeviceArray<float> &joints,
                cudaStream_t stream);
        void transform(
                const DeviceArray<float> &poseRotation,
                const DeviceArray<float> &joints,
                DeviceArray<float> &globalTransformations,
                cudaStream_t stream);
        void skinning(
                const DeviceArray<float> &transformation,
                cudaStream_t stream);
        void jacobian_beta(
                const DeviceArray<float> &transformation,
                DeviceArray<float> &dtheta,
                cudaStream_t stream);
        void countNormals(cudaStream_t stream);
        void CameraTransform(mat34 world2camera);
        void lbsModel(mat34 world2camera, cudaStream_t stream);
        void markVertices(
                const DeviceArrayView<float4>& live_vertex,
                cudaStream_t stream);
        void countKnn(
                const DeviceArrayView<float4>& live_vertex,
                const int frame_idx,
                mat34 world2camera,
                cudaStream_t stream);
    public:
        using Ptr = std::shared_ptr<SMPL>;
        SMPL();
        SURFELWARP_NO_COPY_ASSIGN_MOVE(SMPL);

        void Split(
                const DeviceArrayView<float4>& live_vertex,
                const DeviceArrayView<float4>& reference_vertex,
                const int frame_idx,
                DeviceArray<float4>& onbody_points,
                DeviceArray<float4>& farbody_points,
                mat34 world2camera,
                cudaStream_t stream = 0);

        struct SolverInput {
            DeviceArrayView<float3> smpl_vertices;
            DeviceArrayView<float3> smpl_normals;
            DeviceArrayView<ushort4> knn;
            DeviceArrayView<float4> knn_weight;
            DeviceArrayView<int> onbody;
        };
        SolverInput SolverAccess(
                const DeviceArrayView<float4>& live_vertex,
                const int frame_idx,
                mat34 world2camera,
                cudaStream_t stream = 0);
        DeviceArray<float3> GetVertices() const { return m_smpl_vertices; };
        DeviceArray<int> GetFaceIndices() const { return m__faceIndices; };

        DeviceArrayView<float> GetBeta() const {
            return DeviceArrayView<float>(m__beta);
        }

        DeviceArrayView<float> GetTheta() const {
            return DeviceArrayView<float>(m__theta);
        }

        void ComputeJacobian(
                DeviceArrayView<float4> &live_vertex,
                DeviceArrayView<float4> &live_normal,
                DeviceArrayView<float> &beta0,
                DeviceArrayView<float> &theta0,
                float *r,
                float *j,
                mat34 world2camera,
                cudaStream_t stream
        );
    };
} // namespace smpl
#endif // SMPL_H
