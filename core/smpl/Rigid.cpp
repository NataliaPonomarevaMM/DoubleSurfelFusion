#include "core/smpl/smpl.h"
#include <cilantro/registration/icp_common_instances.hpp>
#include <cilantro/utilities/point_cloud.hpp>
#include <cilantro/visualization.hpp>
#include <cilantro/utilities/timer.hpp>
#include <vector>
#include <Eigen/Eigen>

void surfelwarp::Transform(
        const surfelwarp::DeviceArrayView<float3> &smpl_vertices,
        const surfelwarp::DeviceArrayView<float3> &smpl_normals,
        const surfelwarp::DeviceArrayView<float4> &live_vertex
) {
    std::vector<float4> smpl_host;
    smpl_vertices.Download(smpl_host);
    std::vector<float4> smpl_norm_host;
    smpl_normals.Download(smpl_norm_host);

    cilantro::PointCloud3f smpl_cloud(new cilantro::PointCloud3f());
    smpl_cloud->points.resize(PointCloud3f::Dimension, smpl_host.size());
    smpl_cloud->normals.resize(PointCloudNormal::Dimension, smpl_host.size());
    for (auto i = 0; i < smpl_host.size(); i++) {
        smpl_cloud->points.col(i) = Eigen::Vector3f(smpl_host[i].x, smpl_host[i].y, smpl_host[i].z);;
        smpl_cloud->normals.col(i) = Eigen::Vector3f(smpl_norm_host[i].x, smpl_norm_host[i].y, smpl_norm_host[i].z);
    }

    std::vector<float4> live_host;
    live_vertex.Download(live_host);

    cilantro::PointCloud3f live_cloud(new cilantro::PointCloud3f());
    live_cloud->points.resize(PointCloud3f::Dimension, live_host.size());
    for (auto i = 0; i < live_host.size(); i++) {
        live_cloud->points.col(i) = Eigen::Vector3f(live_host[i].x, live_host[i].y, live_host[i].z);;
    }

    cilantro::SimpleCombinedMetricRigidICP3f icp(smpl_cloud.points, smpl_cloud.normals, live_cloud.points);

    // Parameter setting
    icp.setMaxNumberOfOptimizationStepIterations(1).setPointToPointMetricWeight(0.0f).setPointToPlaneMetricWeight(1.0f);
    icp.correspondenceSearchEngine().setMaxDistance(0.1f*0.1f);
    icp.setConvergenceTolerance(1e-4f).setMaxNumberOfIterations(30);

    cilantro::RigidTransform3f tf_est = icp.estimate().getTransform();
    std::cout << "Iterations performed: " << icp.getNumberOfPerformedIterations() << std::endl;
    std::cout << "Has converged: " << icp.hasConverged() << std::endl;;
    std::cout << "ESTIMATED transformation:" << std::endl << tf_est.matrix() << std::endl
}