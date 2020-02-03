#include <fstream>
// #include <experimental/filesystem>
#include <nlohmann/json.hpp>
#include "core/smpl/smpl.h"
#include "core/smpl/def.h"

namespace surfelwarp {
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

    void moveModel(std::string &modelPath) {
        nlohmann::json model; // JSON object represents.
        std::ifstream file(modelPath);
        file >> model;

        auto shapeBlendBasis = model["shape_blend_shapes"].get<std::vector<std::vector<std::vector<float>>>>();
        auto poseBlendBasis = model["pose_blend_shapes"].get<std::vector<std::vector<std::vector<float>>>>();
        auto templateRestShape = model["vertices_template"].get<std::vector<std::vector<float>>>();
        auto jointRegressor = model["joint_regressor"].get<std::vector<std::vector<float>>>();
        auto kinematicTree = model["kinematic_tree"].get<std::vector<std::vector<int64_t>>>();
        auto weights = model["weights"].get<std::vector<std::vector<float>>>();

        std::vector<float> pose = std::vector<float>(VERTEX_NUM * 3 * POSE_BASIS_DIM);
        std::vector<float> shape = std::vector<float>(VERTEX_NUM * 3 * SHAPE_BASIS_DIM);
        std::vector<float> templ = std::vector<float>(VERTEX_NUM * 3);
        std::vector<float> joint = std::vector<float>(JOINT_NUM * VERTEX_NUM);
        std::vector<float> wei = std::vector<float>(VERTEX_NUM * JOINT_NUM);
        std::vector<float> kinematic = std::vector<float>(2 * JOINT_NUM);

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

        nlohmann::json model2; // JSON object represents.
        std::ofstream file2("/home/nponomareva/DoubleFusion/data/smpl_female2.json");
        model2["shape_blend_shapes"] = shape;
        model2["pose_blend_shapes"] = pose;
        model2["vertices_template"] = templ;
        model2["joint_regressor"] = joint;
        model2["kinematic_tree"] = kinematic;
        model2["weights"] = wei;
        file2 << model2;
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
            cudaStream_t stream,
            const DeviceArray<float> &d_vertices
    ) {
        DeviceArray<float> d_poseRotation(DeviceArray<float>(JOINT_NUM * 9));
        DeviceArray<float> d_restPoseRotation(DeviceArray<float>(JOINT_NUM * 9));
        DeviceArray<float> d_poseBlendShape(DeviceArray<float>(VERTEX_NUM * 3));
        DeviceArray<float> d_shapeBlendShape(DeviceArray<float>(VERTEX_NUM * 3));
        DeviceArray<float> d_restShape(DeviceArray<float>(VERTEX_NUM * 3));
        DeviceArray<float> d_joints(DeviceArray<float>(JOINT_NUM * 3));
        DeviceArray<float> d_globalTransformations(DeviceArray<float>(JOINT_NUM * 16));

        poseBlendShape(theta, d_poseRotation, d_restPoseRotation, d_poseBlendShape, stream);
	    std::cout << "poseblend\n";
        shapeBlendShape(beta, d_shapeBlendShape, stream);
	    std::cout << "shapevlend\n";
        regressJoints(d_shapeBlendShape, d_poseBlendShape, d_restShape, d_joints, stream);
        std::cout << "regress\n";
	    transform(d_poseRotation, d_joints, d_globalTransformations, stream);
	    std::cout << "transform\n";

        if (d_vertices.size() == 0)
            skinning(d_globalTransformations, d_custom_weights, d_restShape, d_result_vertices, stream);
        else
            skinning(d_globalTransformations, d_custom_weights, d_vertices, d_result_vertices, stream);

	    std::cout << "done\n";
    }

    DeviceArray<float> SMPL::lbs_for_model(
            const DeviceArray<float> &beta,
            const DeviceArray<float> &theta,
            cudaStream_t stream
    ) {
        DeviceArray<float> d_result_vertices(DeviceArray<float>(VERTEX_NUM * 3));
        run(beta, theta, d_weights, d_result_vertices, stream);
        return d_result_vertices;
    }
} // namespace smpl
