#include <iostream>
#include <Eigen/Dense>

#include "core/volumetric/VolumetricOptimization.h"

void surfelwarp::VolumetricOptimization::Solve(
        SMPL::Ptr smpl_handler,
        DeviceArrayView<float4> &live_vertex,
        DeviceArrayView<float4> &live_normal,
        mat34 world2camera
) {
    auto beta0 = smpl_handler->GetBeta();
    auto theta0 = smpl_handler->GetTheta();

    for (auto i = 0; i < Constants::kNumGaussNewtonIterations; i++) {
        float r[6892], j[6892 * 82];
        smpl_handler->ComputeJacobian(live_vertex, live_normal, beta0, theta0, r, j, world2camera);

        Eigen::MatrixXf J = Eigen::Map<Eigen::Matrix<float,6892,82>>(j);
        Eigen::VectorXf R = Eigen::Map<Eigen::Matrix<float,6892,1>>(r);;

        Eigen::MatrixXf Jt = J.transpose();
        Eigen::MatrixXf JtJ = Jt * J; // A
        Eigen::VectorXf b = -1 * Jt * R;

        Eigen::LeastSquaresConjugateGradient<Eigen::Matrix<float, 82, 82>> lscg;
        lscg.compute(JtJ);
        auto x = lscg.solve(b);
        std::cout << "#iterations:     " << lscg.iterations() << std::endl;
        std::cout << "estimated error: " << lscg.error()      << std::endl;
        std::cout << "x:" << x << std::endl;
    }
}