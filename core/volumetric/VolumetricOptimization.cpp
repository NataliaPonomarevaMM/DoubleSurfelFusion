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
        for (auto k = 0; k < 3; k++) {
            auto cur_knn = k == 0 ? m_knn_body : (k == 1 ? m_knn_left_arm : m_knn_right_arm);

            float r[6891], j[6891 * 72];
            ComputeJacobian(live_vertex, cur_knn, r, j, stream);
            float distt = count_dist(r);

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
            for (int p = 0; p < 12 * 3; p++)
                v[p] = 0;
            for (int p = 0; p < 3; p++) {
                v[20 * 3 + p] = 0;
                v[21 * 3 + p] = 0;
                v[22 * 3 + p] = 0;
                v[23 * 3 + p] = 0;
            }
            if (k != 0)
                for (int p = 12 * 3; p < 16 * 3; p++)
                    v[p] = 0;
            if (k != 1)
                for (int p = 0; p < 3; p++) {
                    v[16 * 3 + p] = 0;
                    v[18 * 3 + p] = 0;
                    v[20 * 3 + p] = 0;
                }
            if (k != 2)
                for (int p = 0; p < 3; p++) {
                    v[17 * 3 + p] = 0;
                    v[19 * 3 + p] = 0;
                    v[21 * 3 + p] = 0;
                }
            m_smpl_handler->AddTheta(v);

            ComputeJacobian(live_vertex, cur_knn, r, j, stream);
            float dist1 = count_dist(r);

            m_smpl_handler->SubTheta(v);
            m_smpl_handler->SubTheta(v);

            ComputeJacobian(live_vertex, cur_knn, r, j, stream);
            float dist2 = count_dist(r);
            if (dist2 > dist1) {
                m_smpl_handler->AddTheta(v);
                m_smpl_handler->AddTheta(v);
            }
        }
    }
}