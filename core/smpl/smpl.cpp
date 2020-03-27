#include <fstream>
// #include <experimental/filesystem>
#include <nlohmann/json.hpp>
#include "core/smpl/smpl.h"
#include "core/smpl/def.h"

namespace surfelwarp {
    void moveModel() {
        nlohmann::json model; // JSON object represents.
        std::ifstream file("/home/nponomareva/DoubleFusion/data/smpl_female.json");
        file >> model;

        auto shapeBlendBasis = model["shape_blend_shapes"].get<std::vector<std::vector<std::vector<float>>>>();
        auto poseBlendBasis = model["pose_blend_shapes"].get<std::vector<std::vector<std::vector<float>>>>();
        auto templateRestShape = model["vertices_template"].get<std::vector<std::vector<float>>>();
        auto jointRegressor = model["joint_regressor"].get<std::vector<std::vector<float>>>();
        auto kinematicTree = model["kinematic_tree"].get<std::vector<std::vector<int64_t>>>();
        auto weights = model["weights"].get<std::vector<std::vector<float>>>();
        auto faceind = model["face_indices"].get<std::vector<std::vector<float>>>();

        std::vector<float> pose = std::vector<float>(VERTEX_NUM * 3 * POSE_BASIS_DIM);
        std::vector<float> shape = std::vector<float>(VERTEX_NUM * 3 * SHAPE_BASIS_DIM);
        std::vector<float> templ = std::vector<float>(VERTEX_NUM * 3);
        std::vector<float> joint = std::vector<float>(JOINT_NUM * VERTEX_NUM);
        std::vector<float> wei = std::vector<float>(VERTEX_NUM * JOINT_NUM);
        std::vector<int64_t> kinematic = std::vector<int64_t>(2 * JOINT_NUM);
        std::vector<int> face_ind = std::vector<int>(13776 * 3);

        for (int i = 0; i < VERTEX_NUM; i++)
            for (int j = 0; j < 3; j++)
                for (int k = 0; k < POSE_BASIS_DIM; k++)
                    pose[i * 3 * POSE_BASIS_DIM + j * POSE_BASIS_DIM + k] = poseBlendBasis[i][j][k];
        for (int i = 0; i < VERTEX_NUM; i++)
            for (int j = 0; j < 3; j++)
                for (int k = 0; k < SHAPE_BASIS_DIM; k++)
                    shape[i * 3 * SHAPE_BASIS_DIM + j * SHAPE_BASIS_DIM + k] = shapeBlendBasis[i][j][k];
        for (int i = 0; i < VERTEX_NUM; i++)
            for (int j = 0; j < 3; j++)
                templ[i * 3 + j] = templateRestShape[i][j];
        for (int i = 0; i < JOINT_NUM; i++)
            for (int j = 0; j < VERTEX_NUM; j++)
                joint[i * VERTEX_NUM + j] = jointRegressor[i][j];
        for (int i = 0; i < VERTEX_NUM; i++)
            for (int j = 0; j < JOINT_NUM; j++)
                wei[i * JOINT_NUM + j] = weights[i][j];
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < JOINT_NUM; j++)
                kinematic[i * JOINT_NUM + j] = kinematicTree[i][j];
        for (int i = 0; i < 13776; i++)
            for (int j = 0; j < 3; j++)
                face_ind[i * 3 + j] = faceind[i][j];

        nlohmann::json model2; // JSON object represents.
        std::ofstream file2("/home/nponomareva/DoubleFusion/data/smpl_female2.json");
        model2["shape_blend_shapes"] = shape;
        model2["pose_blend_shapes"] = pose;
        model2["vertices_template"] = templ;
        model2["joint_regressor"] = joint;
        model2["kinematic_tree"] = kinematic;
        model2["weights"] = wei;
        model2["face_indices"] = face_ind;
        file2 << model2;
    }

    SMPL::SMPL() {
//        moveModel();

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
        auto face_indices = model["face_indices"].get<std::vector<int>>();

        m__poseBlendBasis.upload(poseBlendBasis.data(), VERTEX_NUM * 3 * POSE_BASIS_DIM);
        m__shapeBlendBasis.upload(shapeBlendBasis.data(), VERTEX_NUM * 3 * SHAPE_BASIS_DIM);
        m__templateRestShape.upload(templateRestShape.data(), VERTEX_NUM * 3);
        m__jointRegressor.upload(jointRegressor.data(), JOINT_NUM * VERTEX_NUM);
        m__kinematicTree.upload(kinematicTree.data(), 2 * JOINT_NUM);
        m__weights.upload(modelweights.data(), VERTEX_NUM * JOINT_NUM);
        m__faceIndices.upload(face_indices.data(), 13776 * 3);

        nlohmann::json tb_data; // JSON object represents.
        std::ifstream file2(hmrPath);
        file2 >> tb_data;
        float *data_arr = tb_data["arr"].get<std::vector<std::vector<float>>>()[0].data();
        m__theta.upload(data_arr + 3, 72);
        m__beta.upload(data_arr + 75, 10);
    }

    void SMPL::lbsModel(cudaStream_t stream) {
	    auto poseRotation = DeviceArray<float>(JOINT_NUM * 9);
        auto restPoseRotation = DeviceArray<float>(JOINT_NUM * 9);
        auto poseBlendShape = DeviceArray<float>(VERTEX_NUM * 3);
        auto shapeBlendShape = DeviceArray<float>(VERTEX_NUM * 3);
        auto joints = DeviceArray<float>(JOINT_NUM * 3);
        auto globalTransformations = DeviceArray<float>(JOINT_NUM * 16);

        m_restShape = DeviceArray<float>(VERTEX_NUM * 3);
        m_smpl_vertices = DeviceArray<float3>(VERTEX_NUM);

        countPoseBlendShape(poseRotation, restPoseRotation, poseBlendShape, stream);
        countShapeBlendShape(shapeBlendShape, stream);
        regressJoints(shapeBlendShape, poseBlendShape, joints, stream);
        transform(poseRotation, joints, globalTransformations, stream);
	    skinning(globalTransformations, stream);

        countNormals(stream);
    }

    SMPL::SolverInput SMPL::SolverAccess(
            const DeviceArrayView<float4>& live_vertex,
            const int frame_idx,
            cudaStream_t stream)
    {
        if (m_knn_frame != frame_idx) {
            countKnn(live_vertex, frame_idx, stream);
            m_knn_frame = frame_idx;
        }

        SolverInput solver_input;
        solver_input.smpl_vertices = m_smpl_vertices;
        solver_input.smpl_normals = m_smpl_normals;
        solver_input.knn = m_knn;
        solver_input.knn_weight = m_knn_weight;
        solver_input.onbody = m_onbody;
        return solver_input;
    }
    DeviceArray<float3> SMPL::GetVertices() const {
        return m_smpl_vertices;
    }
} // namespace smpl
