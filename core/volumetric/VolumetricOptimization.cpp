#include <iostream>
#include <Eigen/Dense>

#include "core/volumetric/VolumetricOptimization.h"

float count_dist(float *r) {
    float dist = 0;
    for (int i = 0; i < 6891; i++)
        dist += r[i] * r[i];
    return sqrt(dist);
}

void surfelwarp::VolumetricOptimization::Solve(
        DeviceArrayView<float4> &live_vertex,
        cudaStream_t stream
) {
    CheckPoints(live_vertex, stream);
    for (auto i = 0; i < Constants::kNumGaussNewtonIterations; i++) {
        float r[6891], j[6891 * 72];
        ComputeJacobian(live_vertex, j, stream);
        ComputeResidual(live_vertex, r, stream);

        Eigen::MatrixXf J = Eigen::Map<Eigen::Matrix<float, 6891, 72>>(j);
        Eigen::VectorXf R(Eigen::Map<Eigen::VectorXf>(r, 6891));
        Eigen::MatrixXf Jt = J.transpose();
        Eigen::MatrixXf JtJ = Jt * J; // A
        Eigen::VectorXf b = -1 * Jt * R;

        Eigen::LeastSquaresConjugateGradient<Eigen::MatrixXf> lscg;
        lscg.compute(JtJ);
        auto x = lscg.solve(b);

        std::vector<float> v;
        for (int p = 0; p < 72; p++)
            v.push_back(0.5 * x(p));
        m_smpl_handler->AddTheta(v);

        ComputeResidual(live_vertex, r, stream);
        float dist1 = count_dist(r);

        m_smpl_handler->SubTheta(v);
        m_smpl_handler->SubTheta(v);

        ComputeResidual(live_vertex, r, stream);
        float dist2 = count_dist(r);
        if (dist2 > dist1) {
            m_smpl_handler->AddTheta(v);
            m_smpl_handler->AddTheta(v);
        }
    }
}