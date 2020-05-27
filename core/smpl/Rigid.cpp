#include "core/smpl/smpl.h"
#include <cilantro/point_cloud.hpp>
#include <cilantro/icp_common_instances.hpp>
#include <vector>
#include <Eigen/Eigen>

namespace surfelwarp {
    void SMPL::Transform(
            const surfelwarp::DeviceArrayView<float4> &live_vertex,
            cudaStream_t stream
    ) {
        std::vector<float3> smpl_host;
        m_smpl_vertices.download(smpl_host);
        std::vector<float3> smpl_norm_host;
        m_smpl_normals.download(smpl_norm_host);

        int num = 0;
        for (auto i = 0; i < smpl_host.size(); i++)
            if (smpl_host[i].z <= 0)
                num++;
        cilantro::PointCloud3f smpl_cloud;
        smpl_cloud.points.resize(cilantro::PointCloud3f::Dimension, num);
        smpl_cloud.normals.resize(cilantro::PointCloud3f::Dimension, num);
        num = 0;
        for (auto i = 0; i < smpl_host.size(); i++) {
            if (smpl_host[i].z <= 0) {
                smpl_cloud.points.col(num) = Eigen::Vector3f(smpl_host[i].x, smpl_host[i].y, smpl_host[i].z);
                smpl_cloud.normals.col(num++) = Eigen::Vector3f(smpl_norm_host[i].x, smpl_norm_host[i].y,
                                                                smpl_norm_host[i].z);
            }
        }

        std::vector<float4> live_host;
        live_vertex.Download(live_host);
        cilantro::PointCloud3f live_cloud;
        live_cloud.points.resize(cilantro::PointCloud3f::Dimension, live_host.size());
        for (auto i = 0; i < live_host.size(); i++)
            live_cloud.points.col(i) = Eigen::Vector3f(live_host[i].x, live_host[i].y, live_host[i].z);

        cilantro::SimpleCombinedMetricRigidICP3f icp(smpl_cloud.points, smpl_cloud.normals, live_cloud.points);

        // Parameter setting
        icp.setMaxNumberOfOptimizationStepIterations(10).setPointToPointMetricWeight(1.0f).setPointToPlaneMetricWeight(
                0.0f);
        icp.correspondenceSearchEngine().setMaxDistance(3.0f);
        icp.setConvergenceTolerance(0.0001f).setMaxNumberOfIterations(500);

        cilantro::RigidTransform3f tf_est = icp.estimate().getTransform().inverse();
        init_mat = mat34(tf_est.matrix());
        transform(stream);
        cudaStreamSynchronize(stream);
    }
}