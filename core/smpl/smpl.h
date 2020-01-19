#ifndef SMPL_H
#define SMPL_H

#include <string>
#include <tuple>
#include "core/smpl/def.h"
#include "common/common_types.h"

namespace smpl {
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
                DeviceArray<float> &d_poseBlendShape);
        void shapeBlendShape(
                const DeviceArray<float> &beta,
                DeviceArray<float> &d_shapeBlendShape);
        void regressJoints(
                const DeviceArray<float> &d_shapeBlendShape,
                const DeviceArray<float> &d_poseBlendShape,
                DeviceArray<float> &d_restShape,
                DeviceArray<float> &d_joints);
        void transform(
                const DeviceArray<float> &d_poseRotation,
                const DeviceArray<float> &d_joints,
                DeviceArray<float> &d_globalTransformations);
        void skinning(
                const DeviceArray<float> &d_transformation,
                const DeviceArray<float> &d_custom_weights,
                const DeviceArray<float> &d_vertices,
                DeviceArray<float> &d_result_vertices);

        void SMPL::run(
                const DeviceArray<float> &beta,
                const DeviceArray<float> &theta,
                const DeviceArray<float> &d_custom_weights,
                DeviceArray<float> &d_result_vertices,
                const DeviceArray<float> &d_vertices = nullptr
        );
    public:
        // Constructor and Destructor
        SMPL(std::string &modelPath);
        ~SMPL();
        // Run the model with a specific group of beta, theta.
        DeviceArray<float> SMPL::lbs_for_model(const DeviceArray<float> &beta, const DeviceArray<float> &theta)
        float *lbs_for_custom_vertices(float *beta, float *theta, float *vertices, int vertnum);
    };
} // namespace smpl
#endif // SMPL_H
