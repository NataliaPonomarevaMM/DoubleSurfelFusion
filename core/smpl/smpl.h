#ifndef SMPL_H
#define SMPL_H

#include <string>
#include <tuple>
#include <memory>
#include "core/smpl/def.h"
#include "common/common_types.h"
#include "imgproc/ImageProcessor.h"
#include "imgproc/frameio/VolumeDeformFileFetch.h"
#include "core/SurfelGeometry.h"
#include "core/volumetric/PairsSorting.h"

namespace surfelwarp {
    class SMPL {
    private:
        DeviceArray<float> m__beta;
        DeviceArray<float> m__theta;
        DeviceArray<float> m__beta0;
        DeviceArray<float> m__theta0;

        ///constant
        DeviceArray<float> m__poseBlendBasis; // Basis of the pose-dependent shape space, (6890, 3, 207).
        DeviceArray<float> m__shapeBlendBasis; // Basis of the shape-dependent shape space, (6890, 3, 10).
        DeviceArray<float> m__templateRestShape; // Template shape in rest pose, (6890, 3).
        DeviceArray<float> m__jointRegressor; // Joint coefficients of each vertices for regressing them to joint locations, (24, 6890).
        DeviceArray<float> m__weights; // Weights for linear blend skinning, (6890, 24).
        DeviceArray<int64_t> m__kinematicTree; // Hierarchy relation between joints, the root is at the belly button, (2, 24).
        DeviceArray<int> m__faceIndices; // (13776, 3)

        ///smpl information
        DeviceArray<float> m_restShape;
        DeviceArray<float3> m_smpl_vertices;
        DeviceArray<float3> m_smpl_normals;
        /// surfel-based information
        DeviceBufferArray<float> m_dist; // (live_vertex.size,6890)
        unsigned m_num_marked = 0; // number of onbody live vertices
        DeviceBufferArray<ushort4> m_knn; // smpl knn for live_vertices (m_num_marked)
        DeviceBufferArray<float4> m_knn_weight; // smpl knn weights for live vertices (m_num_marked)
        DeviceBufferArray<int> m_onbody; // is live vertex on body (live_vertex.size)

        PairsSorting::Ptr m_pair_sorting;
        mat34 init_mat;

        void countPoseBlendShape(
                DeviceArray<float> &poseRotation,
                DeviceArray<float> &restPoseRotation,
                DeviceArray<float> &poseBlendShape,
                cudaStream_t stream);
        void countShapeBlendShape(
                DeviceArray<float> &shapeBlendShape,
                cudaStream_t stream);
        void countRestShape(
                const DeviceArray<float> &shapeBlendShape,
                const DeviceArray<float> &poseBlendShape,
                cudaStream_t stream);
        void regressJoints(
                const DeviceArray<float> &shapeBlendShape,
                DeviceArray<float> &joints,
                cudaStream_t stream);
        void transform(
                const DeviceArray<float> &poseRotation,
                const DeviceArray<float> &joints,
                DeviceArray<float> &globalTransformations,
                DeviceArray<float> &localTransformations,
                cudaStream_t stream);
        void skinning(
                const DeviceArray<float> &transformation,
                cudaStream_t stream);
        void countNormals(cudaStream_t stream);
        unsigned count_dist(
                const DeviceArrayView<float4>& live_vertex,
                DeviceArray<unsigned> &marked,
                DeviceArray<float> &dist,
                DeviceArray<int> &knn_ind,
                cudaStream_t stream
        );
        void count_knn(
                const DeviceArray<float> &dist,
                const DeviceArray<unsigned> &marked,
                const DeviceArray<int> &knn_ind,
                const unsigned num_marked,
                DeviceArray<int> &onbody,
                DeviceArray<ushort4> &knn,
                DeviceArray<float4> &knn_weight,
                cudaStream_t stream
        );
        void transform(cudaStream_t stream);
        void countPoseJac(DeviceArray<float3> &vertJac,cudaStream_t stream);
    public:
        using Ptr = std::shared_ptr<SMPL>;
        SMPL();
        ~SMPL();
        SURFELWARP_NO_COPY_ASSIGN_MOVE(SMPL);

        void LbsModel(cudaStream_t stream = 0);
        void CountKnn(const DeviceArrayView<float4>& live_vertex, cudaStream_t stream = 0);
        void CountAppendedKnn(
                const DeviceArrayView<float4>& appended_live_vertex,
                int num_remaining_surfel,
                int num_appended_surfel,
                cudaStream_t stream = 0);
        void CameraTransform(mat34 world2camera, cudaStream_t stream = 0);

        void SplitReferenceVertices(
                const DeviceArrayView<float4>& live_vertex,
                const DeviceArrayView<float4>& reference_vertex,
                DeviceArray<float4>& onbody_points,
                DeviceArray<float4>& farbody_points,
                cudaStream_t stream = 0);

        struct SolverInput {
            DeviceArrayView<float3> smpl_vertices;
            DeviceArrayView<float3> smpl_normals;
            DeviceArrayView<ushort4> knn;
            DeviceArrayView<float4> knn_weight;
            DeviceArrayView<int> onbody;
        };
        SolverInput SolverAccess(cudaStream_t stream = 0) const;
        DeviceArrayView<float3> GetVertices() const { return  DeviceArrayView<float3>(m_smpl_vertices); }
        DeviceArrayView<float3> GetNormals() const { return  DeviceArrayView<float3>(m_smpl_normals); }
        DeviceArrayView<int> GetFaceIndices() const { return  DeviceArrayView<int>(m__faceIndices); }
        DeviceArrayView<float> GetBeta() const {return DeviceArrayView<float>(m__beta); }
        DeviceArrayView<float> GetTheta() const {return DeviceArrayView<float>(m__theta);}
        DeviceArrayView<float> GetBeta0() {
            m__beta.copyTo(m__beta0);
            return DeviceArrayView<float>(m__beta0);
        }
        DeviceArrayView<float> GetTheta0() {
            m__theta.copyTo(m__theta0);
            return DeviceArrayView<float>(m__theta0);
        }

        bool ShouldDoVolumetricOptimization() const {return m_num_marked > 0;}
        void ComputeJacobian(
                DeviceArrayView<float4> &live_vertex,
                DeviceArrayView<float4> &live_normal,
                DeviceArrayView<float> &beta0,
                DeviceArrayView<float> &theta0,
                float *r,
                float *j,
                cudaStream_t stream = 0
        );
        void AddBetaTheta(
                DeviceArrayView<float> &to_add,
                cudaStream_t stream = 0
        );

        void Transform(const surfelwarp::DeviceArrayView<float4> &live_vertex,cudaStream_t stream = 0);
    };
} // namespace smpl


bool knn_cuda_texture(const float * ref,
                      int           ref_nb,
                      const float * query,
                      int           query_nb,
                      int           dim,
                      int           k,
                      float *       knn_dist,
                      int *         knn_index);

#endif // SMPL_H
