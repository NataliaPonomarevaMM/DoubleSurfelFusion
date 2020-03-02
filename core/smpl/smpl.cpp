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

        d_poseBlendBasis.upload(poseBlendBasis.data(), VERTEX_NUM * 3 * POSE_BASIS_DIM);
        d_shapeBlendBasis.upload(shapeBlendBasis.data(), VERTEX_NUM * 3 * SHAPE_BASIS_DIM);
        d_templateRestShape.upload(templateRestShape.data(), VERTEX_NUM * 3);
        d_jointRegressor.upload(jointRegressor.data(), JOINT_NUM * VERTEX_NUM);
        d_kinematicTree.upload(kinematicTree.data(), 2 * JOINT_NUM);
        d_weights.upload(modelweights.data(), VERTEX_NUM * JOINT_NUM);

        nlohmann::json tb_data; // JSON object represents.
        std::ifstream file(hmrPath);
        file >> tb_data;
        float *data_arr = tb_data["arr"].get<std::vector<std::vector<float>>>()[0].data();
        m__theta.upload(data_arr + 3, 72);
        m__beta.upload(data_arr + 75, 10);

        cudaSafeCall(cudaDeviceSynchronize());
        cudaSafeCall(cudaGetLastError());
    }

    SMPL::~SMPL() {
        d_poseBlendBasis.release();
        d_shapeBlendBasis.release();
        d_templateRestShape.release();
        d_jointRegressor.release();
        d_kinematicTree.release();
        d_weights.release();
    }

    void SMPL::lbs_for_model(
            DeviceArray<float> &result_vertices,
            cudaStream_t stream
    ) {
	    auto poseRotation = DeviceArray<float>(JOINT_NUM * 9);
        auto restPoseRotation = DeviceArray<float>(JOINT_NUM * 9);
        auto poseBlendShape = DeviceArray<float>(VERTEX_NUM * 3);
        auto shapeBlendShape = DeviceArray<float>(VERTEX_NUM * 3);
        auto restShape = DeviceArray<float>(VERTEX_NUM * 3);
        auto joints = DeviceArray<float>(JOINT_NUM * 3);
        auto globalTransformations = DeviceArray<float>(JOINT_NUM * 16);

        poseBlendShape(poseRotation, restPoseRotation, poseBlendShape, stream);
        shapeBlendShape(shapeBlendShape, stream);
        regressJoints(shapeBlendShape, poseBlendShape, restShape, joints, stream);
        transform(poseRotation, joints, globalTransformations, stream);
	    skinning(globalTransformations, m__weights, restShape, result_vertices, stream);

	    cudaSafeCall(cudaDeviceSynchronize(stream));
        cudaSafeCall(cudaGetLastError(stream));
        std::cout << "done\n";
    }
} // namespace smpl
