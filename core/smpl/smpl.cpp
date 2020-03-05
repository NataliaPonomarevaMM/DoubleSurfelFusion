#include <fstream>
#include <nlohmann/json.hpp>
#include "core/smpl/smpl.h"
#include "core/smpl/def.cuh"

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

        cudaMemcpy(m__poseBlendBasis, poseBlendBasis.data(),
                sizeof(float) * VERTEX_NUM * 3 * POSE_BASIS_DIM, cudaMemcpyHostToDevice);
        cudaMemcpy(m__shapeBlendBasis, shapeBlendBasis.data(),
                sizeof(float) * VERTEX_NUM * 3 * SHAPE_BASIS_DIM, cudaMemcpyHostToDevice);
        cudaMemcpy(m__templateRestShape, templateRestShape.data(),
                sizeof(float) * VERTEX_NUM * 3, cudaMemcpyHostToDevice);
        cudaMemcpy(m__jointRegressor, jointRegressor.data(),
                sizeof(float) * JOINT_NUM * VERTEX_NUM, cudaMemcpyHostToDevice);
        cudaMemcpy(m__kinematicTree, kinematicTree.data(),
                sizeof(int64_t) * 2 * JOINT_NUM, cudaMemcpyHostToDevice);

        m__weights.upload(modelweights.data(), VERTEX_NUM * JOINT_NUM);

        nlohmann::json tb_data; // JSON object represents.
        std::ifstream file2(hmrPath);
        file2 >> tb_data;
        float *data_arr = tb_data["arr"].get<std::vector<std::vector<float>>>()[0].data();
        m__theta.upload(data_arr + 3, 72);
        m__beta.upload(data_arr + 75, 10);

        cudaSafeCall(cudaDeviceSynchronize());
        cudaSafeCall(cudaGetLastError());
    }

    void SMPL::LbsModel(
            DeviceArray<float> &result_vertices,
            cudaStream_t stream
    ) {
	    auto poseRotation = DeviceArray<float>(JOINT_NUM * 9);
        auto restPoseRotation = DeviceArray<float>(JOINT_NUM * 9);
        auto pposeBlendShape = DeviceArray<float>(VERTEX_NUM * 3);
        auto sshapeBlendShape = DeviceArray<float>(VERTEX_NUM * 3);
        auto restShape = DeviceArray<float>(VERTEX_NUM * 3);
        auto joints = DeviceArray<float>(JOINT_NUM * 3);
        auto globalTransformations = DeviceArray<float>(JOINT_NUM * 16);

        poseBlendShape(poseRotation, restPoseRotation, pposeBlendShape, stream);
        shapeBlendShape(sshapeBlendShape, stream);
        regressJoints(sshapeBlendShape, pposeBlendShape, restShape, joints, stream);
        transform(poseRotation, joints, globalTransformations, stream);
	    skinning(globalTransformations, m__weights, restShape, result_vertices, stream);

	    cudaSafeCall(cudaDeviceSynchronize());
	    cudaSafeCall(cudaGetLastError());
    }
} // namespace smpl
