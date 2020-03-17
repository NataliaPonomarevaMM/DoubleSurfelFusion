#include <fstream>
// #include <experimental/filesystem>
#include <nlohmann/json.hpp>
#include "core/smpl/smpl.h"
#include "core/smpl/def.h"

namespace surfelwarp {
    SMPL::SMPL() {
        std::string modelPath = "/home/nponomareva/DoubleFusion/data/smpl_female2.json";
        std::string hmrPath = "/home/nponomareva/data/hmr_results/hmr_data.json";

        nlohmann::json model; // JSON object represents.
        std::ifstream file(modelPath);
        file >> model;
        auto shapeBlendBasis = model["shape_blend_shapes"].get<std::vector<float>>();
        auto poseBlendBasis = model["pose_blend_shapes"].get<std::vector<float>>();
        auto templateRestShape = model["vertices_template"].get<std::vector<float>>();
        auto jointRegressor = model["joint_regressor"].get<std::vector<float>>();
        auto kinematicTree = model["kinematic_tree"].get<std::vector<int64_t>>();
        auto modelweights = model["weights"].get<std::vector<float>>();

        m__poseBlendBasis.upload(poseBlendBasis.data(), VERTEX_NUM * 3 * POSE_BASIS_DIM);
        m__shapeBlendBasis.upload(shapeBlendBasis.data(), VERTEX_NUM * 3 * SHAPE_BASIS_DIM);
        m__templateRestShape.upload(templateRestShape.data(), VERTEX_NUM * 3);
        m__jointRegressor.upload(jointRegressor.data(), JOINT_NUM * VERTEX_NUM);
        m__kinematicTree.upload(kinematicTree.data(), 2 * JOINT_NUM);
        m__weights.upload(modelweights.data(), VERTEX_NUM * JOINT_NUM);

        nlohmann::json tb_data; // JSON object represents.
        std::ifstream file2(hmrPath);
        file2 >> tb_data;
        float *data_arr = tb_data["arr"].get<std::vector<std::vector<float>>>()[0].data();
        m__theta.upload(data_arr + 3, 72);
        m__beta.upload(data_arr + 75, 10);

        m_restShape = DeviceArray<float>(VERTEX_NUM * 3);
        m_smpl_vertices = DeviceArray<float>(VERTEX_NUM * 3);
    }

    void SMPL::LbsModel(cudaStream_t stream) {
	    auto poseRotation = DeviceArray<float>(JOINT_NUM * 9);
        auto restPoseRotation = DeviceArray<float>(JOINT_NUM * 9);
        auto poseBlendShape = DeviceArray<float>(VERTEX_NUM * 3);
        auto shapeBlendShape = DeviceArray<float>(VERTEX_NUM * 3);
        auto joints = DeviceArray<float>(JOINT_NUM * 3);
        auto globalTransformations = DeviceArray<float>(JOINT_NUM * 16);

        countPoseBlendShape(poseRotation, restPoseRotation, poseBlendShape, stream);
        countShapeBlendShape(shapeBlendShape, stream);
        regressJoints(shapeBlendShape, poseBlendShape, joints, stream);
        transform(poseRotation, joints, globalTransformations, stream);
	    skinning(globalTransformations, stream);
    }

    SMPL::SolverInput SMPL::SolverAccess() const {
        SolverInput solver_input;
        solver_input.smpl_vertices = m_smpl_vertices;
        solver_input.knn = m_knn;
        solver_input.knn_weight = m_knn_weight;
        solver_input.onbody = m_onbody;
        return solver_input;
    }
} // namespace smpl
