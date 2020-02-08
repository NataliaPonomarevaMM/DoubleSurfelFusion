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
        ///GPU
        DeviceArray<float> d_poseBlendBasis; // Basis of the pose-dependent shape space, (6890, 3, 207).
        DeviceArray<float> d_shapeBlendBasis; // Basis of the shape-dependent shape space, (6890, 3, 10).
        DeviceArray<float> d_templateRestShape; // Template shape in rest pose, (6890, 3).
        DeviceArray<float> d_jointRegressor; // Joint coefficients of each vertices for regressing them to joint locations, (24, 6890).
        DeviceArray<float> d_weights; // Hierarchy relation between joints, the root is at the belly button, (2, 24).
        DeviceArray<int64_t> d_kinematicTree; // Weights for linear blend skinning, (6890, 24).

        void poseBlendShape(
                const DeviceArray<float> &theta,
                DeviceArray<float> &d_poseRotation,
                DeviceArray<float> &d_restPoseRotation,
                DeviceArray<float> &d_poseBlendShape,
                cudaStream_t stream);
        void shapeBlendShape(
                const DeviceArray<float> &beta,
                DeviceArray<float> &d_shapeBlendShape,
                cudaStream_t stream);
        void regressJoints(
                const DeviceArray<float> &d_shapeBlendShape,
                const DeviceArray<float> &d_poseBlendShape,
                DeviceArray<float> &d_restShape,
                DeviceArray<float> &d_joints,
                cudaStream_t stream);
        void transform(
                const DeviceArray<float> &d_poseRotation,
                const DeviceArray<float> &d_joints,
                DeviceArray<float> &d_globalTransformations,
                cudaStream_t stream);
        void skinning(
                const DeviceArray<float> &d_transformation,
                const DeviceArray<float> &d_custom_weights,
                const DeviceArray<float> &d_vertices,
                DeviceArray<float> &d_result_vertices,
                cudaStream_t stream);

        void run(
                const DeviceArray<float> &beta,
                const DeviceArray<float> &theta,
                const DeviceArray<float> &d_custom_weights,
                DeviceArray<float> &d_result_vertices,
                cudaStream_t stream,
                const DeviceArray<float> &d_vertices = DeviceArray<float>());
    public:
        using Ptr = std::shared_ptr<SMPL>;
        SMPL(std::string &modelPath);
        ~SMPL();
        SURFELWARP_NO_COPY_ASSIGN_MOVE(SMPL);
        // Run the model with a specific group of beta, theta.
        void lbs_for_model(
                const DeviceArray<float> &beta,
                const DeviceArray<float> &theta,
		DeviceArray<float> &result_vertices,
                cudaStream_t stream = 0);
        DeviceArray<float> lbs_for_custom_vertices(
            const DeviceArray<float> &beta,
            const DeviceArray<float> &theta,
            const DeviceArray<float> &d_vertices,
            cudaStream_t stream = 0);
    };
} // namespace smpl
#endif // SMPL_H
