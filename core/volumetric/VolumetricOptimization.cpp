#include "VolumetricOptimization.h"
#include "common/ConfigParser.h"

#include <iostream>
#include <Eigen/Dense>

void surfelwarp::VolumetricOptimization::Solve(
        SMPL::Ptr smpl_handler,
        DeviceArrayView<float4> &live_vertex,
        DeviceArrayView<float4> &live_normal,
        mat34 world2camera
) {
    auto beta0 = smpl_handler->GetBeta();
    auto theta = smpl_handler->GetTheta();

    for (auto i = 0; i < Constants::kNumGaussNewtonIterations; i++) {
        float r[6892], j[6892 * 82];
        smpl_handler->ComputeJacobian(live_vertex, live_normal, beta0, theta0, r, j, world2camera);

        Eigen::MatrixXd J(j, 6892, 82); //6890 smpl vert + 2 reg, 10 beta + 72 theta
        Eigen::VectorXd r(r, 6892);

        Eigen::MatrixXd Jt = J.transpose();
        Eigen::MatrixXd JtJ = Jt * J; // A
        Eigen::VectorXd b = -1 * Jt * r;

        LeastSquaresConjugateGradient<Matrix<double>> lscg;
        lscg.compute(JtJ);
        auto x = lscg.solve(b);
        std::cout << "#iterations:     " << lscg.iterations() << std::endl;
        std::cout << "estimated error: " << lscg.error()      << std::endl;
        std::cout << x << std::endl;
    }
}