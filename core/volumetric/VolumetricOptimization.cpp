#include <iostream>
#include <Eigen/Dense>

#include "core/volumetric/VolumetricOptimization.h"

void surfelwarp::VolumetricOptimization::Solve(
        SMPL::Ptr smpl_handler,
        DeviceArrayView<float4> &live_vertex,
        DeviceArrayView<float4> &live_normal
) {
    std::cout << "started\n";
    auto beta0 = smpl_handler->GetBeta0();
    auto theta0 = smpl_handler->GetTheta0();

    //for (auto i = 0; i < Constants::kNumGaussNewtonIterations; i++) {
    for (auto i = 0; i < 1; i++) {
        float r[6892], j[6892 * 82];
        smpl_handler->ComputeJacobian(live_vertex, live_normal, beta0, theta0, r, j);

        std::cout << "jacobian computed\n";

        Eigen::MatrixXf J = Eigen::Map<Eigen::Matrix<float,6892,82>>(j);
        Eigen::VectorXf R(Eigen::Map<Eigen::VectorXf>(r, 6892));

        Eigen::MatrixXf Jt = J.transpose();
        Eigen::MatrixXf JtJ = Jt * J; // A
        Eigen::VectorXf b = -1 * Jt * R;

        Eigen::LeastSquaresConjugateGradient<Eigen::MatrixXf> lscg;
        lscg.compute(JtJ);

        std::cout << "eigen computed\n";

        auto x = lscg.solve(b);
        std::cout << "#iterations:     " << lscg.iterations() << std::endl;
        std::cout << "estimated error: " << lscg.error()      << std::endl;
        //from the eigen vector to the std vector
        std::vector<float> v;
        for (int i = 0; i < 82; i++)
            v.push_back(x(i));
        auto new_beta_theta = DeviceArray<float>(82);
        new_beta_theta.upload(v);
        auto bt = DeviceArrayView<float>(new_beta_theta);
        smpl_handler->AddBetaTheta(bt);
    }

    std::vector<float> old_beta, new_beta;
    beta0.Download(old_beta);
    smpl_handler->GetBeta().Download(new_beta);
    std::cout << "old beta:";
    for (int i = 0; i < 10; i++)
        std::cout << old_beta[i] << " ";
    std::cout << "\n";
    std::cout << "new beta:";
    for (int i = 0; i < 10; i++)
        std::cout << new_beta[i] << " ";
    std::cout << "\n";

    std::vector<float> old_theta, new_theta;
    theta0.Download(old_theta);
    smpl_handler->GetTheta().Download(new_theta);
    std::cout << "old theta:";
    for (int i = 0; i < 72; i++)
        std::cout << old_theta[i] << " ";
    std::cout << "\n";
    std::cout << "new theta:";
    for (int i = 0; i < 72; i++)
        std::cout << new_theta[i] << " ";
    std::cout << "\n";
}