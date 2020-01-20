#include <fstream>
// #include <experimental/filesystem>
#include <nlohmann/json.hpp>
#include <xtensor/xarray.hpp>
#include <xtensor/xjson.hpp>
#include "core/smpl/smpl.h"
#include "core/smpl/def.h"

namespace smpl {
    SMPL::SMPL(std::string &modelPath) {
        nlohmann::json model; // JSON object represents.
        std::ifstream file(modelPath);
        file >> model;

        auto shapeBlendBasis = model["shape_blend_shapes"].get<std::vector<float>>();
        auto poseBlendBasis = model["pose_blend_shapes"].get<std::vector<float>>();
        auto templateRestShape = model["vertices_template"].get<std::vector<float>>();
        auto jointRegressor = model["joint_regressor"].get<std::vector<float>>();
        auto kinematicTree = model["kinematic_tree"].get<std::vector<int64_t>>();
        auto weights = model["weights"].get<std::vector<float>>();

        d_poseBlendBasis = DeviceArray<float>(poseBlendBasis.data(), VERTEX_NUM * 3 * POSE_BASIS_DIM);
        d_shapeBlendBasis = DeviceArray<float>(shapeBlendBasis.data(), VERTEX_NUM * 3 * SHAPE_BASIS_DIM);
        d_templateRestShape = DeviceArray<float>(templateRestShape.data(), VERTEX_NUM * 3);
        d_jointRegressor = DeviceArray<float>(jointRegressor.data(), JOINT_NUM * VERTEX_NUM);
        d_weights = DeviceArray<float>(weights.data(), VERTEX_NUM * JOINT_NUM);
        d_kinematicTree = DeviceArray<int64_t>(kinematicTree.data(), 2 * JOINT_NUM);
    }

    SMPL::~SMPL() {
        d_poseBlendBasis.release();
        d_shapeBlendBasis.release();
        d_templateRestShape.release();
        d_jointRegressor.release();
        d_weights.release();
        d_kinematicTree.release();
    }

    void SMPL::run(
            const DeviceArray<float> &beta,
            const DeviceArray<float> &theta,
            const DeviceArray<float> &d_custom_weights,
            DeviceArray<float> &d_result_vertices,
            const DeviceArray<float> &d_vertices
    ) {
        DeviceArray<float> d_poseRotation(DeviceArray<float>(JOINT_NUM * 9));
        DeviceArray<float> d_restPoseRotation(DeviceArray<float>(theta, JOINT_NUM * 9));
        DeviceArray<float> d_poseBlendShape(DeviceArray<float>(theta, VERTEX_NUM * 3));
        DeviceArray<float> d_shapeBlendShape(DeviceArray<float>(VERTEX_NUM * 3));
        DeviceArray<float> d_restShape(DeviceArray<float>(VERTEX_NUM * 3));
        DeviceArray<float> d_joints(DeviceArray<float>(JOINT_NUM * 3));
        DeviceArray<float> d_globalTransformations(DeviceArray<float>(JOINT_NUM * 16));

        if (d_vertices.size() == 0)
            d_vertices = d_restShape;

        poseBlendShape(theta, d_poseRotation, d_restPoseRotation, d_poseBlendShape);
        shapeBlendShape(beta, d_shapeBlendShape);
        regressJoints(d_shapeBlendShape, d_poseBlendShape, d_restShape, d_joints);
        transform(d_poseRotation, d_joints, d_globalTransformations);
        skinning(d_globalTransformations, d_custom_weights, d_vertices, d_result_vertices);

        d_shapeBlendShape.release();
        d_poseBlendShape.release();
        d_poseRotation.release();
        d_joints.release();
        d_restShape.release();
        d_globalTransformations.release();
    }

    DeviceArray<float> SMPL::lbs_for_model(const DeviceArray<float> &beta, const DeviceArray<float> &theta) {
        DeviceArray<float> d_result_vertices(DeviceArray<float>(VERTEX_NUM * 3));
        run(beta, theta, d_weights, d_result_vertices);
        return d_result_vertices;
    }
} // namespace smpl
