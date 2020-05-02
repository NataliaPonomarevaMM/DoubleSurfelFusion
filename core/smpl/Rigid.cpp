#include "core/smpl/smpl.h"
#include <cilantro/point_cloud.hpp>
#include <cilantro/icp_common_instances.hpp>
#include <vector>
#include <Eigen/Eigen>

void surfelwarp::Transform(
        const surfelwarp::DeviceArrayView<float3> &smpl_vertices,
        const surfelwarp::DeviceArrayView<float3> &smpl_normals,
        const surfelwarp::DeviceArrayView<float4> &live_vertex,
        const surfelwarp::DeviceArrayView<int> &face_ind
) {
    std::vector<float3> smpl_host;
    smpl_vertices.Download(smpl_host);
    std::vector<float3> smpl_norm_host;
    smpl_normals.Download(smpl_norm_host);

    cilantro::PointCloud3f smpl_cloud;
    smpl_cloud.points.resize(cilantro::PointCloud3f::Dimension, smpl_host.size());
    smpl_cloud.normals.resize(cilantro::PointCloud3f::Dimension, smpl_host.size());
    std::ofstream file_output;
    file_output.open("/home/nponomareva/surfelwarp/rigid/smpl.obj");
    for (auto i = 0; i < smpl_host.size(); i++) {
        smpl_cloud.points.col(i) = Eigen::Vector3f(smpl_host[i].x, smpl_host[i].y, smpl_host[i].z);
        smpl_cloud.normals.col(i) = Eigen::Vector3f(smpl_norm_host[i].x, smpl_norm_host[i].y, smpl_norm_host[i].z);
        file_output << 'v' << ' '
                    << smpl_host[i].x << ' '
                    << smpl_host[i].y << ' '
                    << smpl_host[i].z << std::endl;
    }
    std::vector<int> h_face;
    face_ind.Download(h_face);
    for (auto i = 0; i < h_face.size() / 3; i++) {
        file_output << 'f' << ' '
                    << h_face[i * 3] << ' '
                    << h_face[i * 3 + 1] << ' '
                    << h_face[i * 3 + 2] << std::endl;
    }
    file_output.close();

    std::vector<float4> live_host;
    live_vertex.Download(live_host);

    cilantro::PointCloud3f live_cloud;
    live_cloud.points.resize(cilantro::PointCloud3f::Dimension, live_host.size());
    file_output.open("/home/nponomareva/surfelwarp/rigid/live.obj");
    for (auto i = 0; i < live_host.size(); i++) {
        live_cloud.points.col(i) = Eigen::Vector3f(live_host[i].x, live_host[i].y, live_host[i].z);
        file_output << 'v' << ' '
                    << live_host[i].x << ' '
                    << live_host[i].y << ' '
                    << live_host[i].z << std::endl;
    }
    file_output.close();

    cilantro::SimpleCombinedMetricRigidICP3f icp(smpl_cloud.points, smpl_cloud.normals, live_cloud.points);

    // Parameter setting
    icp.setMaxNumberOfOptimizationStepIterations(10).setPointToPointMetricWeight(1.0f).setPointToPlaneMetricWeight(0.0f);
    icp.correspondenceSearchEngine().setMaxDistance(10.0f);
    icp.setConvergenceTolerance(1e-4f).setMaxNumberOfIterations(100);

    cilantro::RigidTransform3f tf_est = icp.estimate().getTransform();
    std::cout << "Iterations performed: " << icp.getNumberOfPerformedIterations() << std::endl;
    std::cout << "Has converged: " << icp.hasConverged() << std::endl;;
    std::cout << "ESTIMATED transformation:" << std::endl << tf_est.matrix() << std::endl;
    std::cout << "ESTIMATED linear:" << std::endl << tf_est.linear() << std::endl;
    std::cout << "ESTIMATED trans:" << std::endl << tf_est.translation() << std::endl;

    auto m = tf_est.matrix();
    std::cout << m(0,0) << " " << m(0,1) <<"\n";

    file_output.open("/home/nponomareva/surfelwarp/rigid/smpl_estimated.obj");
    for (auto i = 0; i < smpl_host.size(); i++) {
        float x = m(0,0) * smpl_host[i].x + m(0,1) * smpl_host[i].y +
                m(0,2) * smpl_host[i].z + m(0,3);
        float y = m(1,0) * smpl_host[i].x + m(1,1) * smpl_host[i].y +
                  m(1,2) * smpl_host[i].z + m(1,3);;
        float z = m(2,0) * smpl_host[i].x + m(2,1) * smpl_host[i].y +
                  m(2,2) * smpl_host[i].z + m(2,3);;
        file_output << 'v' << ' '
                    << x << ' '
                    << y << ' '
                    << z << std::endl;
    }
    for (auto i = 0; i < h_face.size() / 3; i++) {
        file_output << 'f' << ' '
                    << h_face[i * 3] << ' '
                    << h_face[i * 3 + 1] << ' '
                    << h_face[i * 3 + 2] << std::endl;
    }
    file_output.close();

    file_output.open("/home/nponomareva/surfelwarp/rigid/live_estimated.obj");
    for (auto i = 0; i < live_host.size(); i++) {
        float x = m(0,0) * live_host[i].x + m(0,1) * live_host[i].y +
                  m(0,2) * live_host[i].z + m(0,3);
        float y = m(1,0) * live_host[i].x + m(1,1) * live_host[i].y +
                  m(1,2) * live_host[i].z + m(1,3);;
        float z = m(2,0) * live_host[i].x + m(2,1) * live_host[i].y +
                  m(2,2) * live_host[i].z + m(2,3);;
        file_output << 'v' << ' '
                    << x << ' '
                    << y << ' '
                    << z << std::endl;
    }
    file_output.close();
}