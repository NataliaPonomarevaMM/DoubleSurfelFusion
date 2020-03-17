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

        ///useful information to store
        DeviceArray<float> m_restShape;
        DeviceArray<float> m_smpl_vertices;
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
    public:
        using Ptr = std::shared_ptr<SMPL>;
        SMPL();
        ~SMPL();
        SURFELWARP_NO_COPY_ASSIGN_MOVE(SMPL);

        void LbsModel(cudaStream_t stream = 0);
        void Split(
                const DeviceArrayView<float4>& reference_vertex,
                DeviceArray<float4>& onbody_points,
                DeviceArray<float4>& farbody_points,
                cudaStream_t stream = 0);
        void CountKnn(
                const DeviceArrayView<float4>& reference_vertex,
                cudaStream_t stream = 0);

        struct SolverInput {
            DeviceArrayView<float> smpl_vertices;
            DeviceArrayView<ushort4> knn;
            DeviceArrayView<float4> knn_weight;
            DeviceArrayView<int> onbody;
        };
        SolverInput SolverAccess() const;
    };
} // namespace smpl
#endif // SMPL_H
