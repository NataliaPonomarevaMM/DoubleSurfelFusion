#include <vector>
#include <cmath>
#include "core/smpl/def.h"
#include "core/smpl/smpl.h"
#include "common/common_types.h"
#include "common/Constants.h"

namespace surfelwarp {
    namespace device {
        __global__ void FindKNN1(
                const PtrSz<const float> templateRestShape,
                const PtrSz<const float> shapeBlendShape,
                const int vertexnum,
                const PtrSz<const float> curvertices,
                PtrSz<float> dist
        ) {
            int i = blockIdx.x;
            int j = threadIdx.x;
            int ind = i * vertexnum + j;
            dist[ind] = 0;
            for (int k = 0; k < 3; k++) {
                float restShape = templateRestShape[j * 3 + k] + shapeBlendShape[j * 3 + k];
                dist[ind] += (curvertices[i * 3 + k] - restShape) * (curvertices[i * 3 + k] - restShape);
            }
        }

        __global__ void FindKNN2(
                const PtrSz<const float>dist,
                const int vertexnum,
                PtrSz<int> ind
        ) {
            int i = threadIdx.x;

            ind[i * 4 + 0] = 0;
            ind[i * 4 + 1] = 1;
            ind[i * 4 + 2] = 2;
            ind[i * 4 + 3] = 3;

            for (int l = 0; l < 4; l++)
                for (int p = 0; p < 3 - l; p++)
                    if (dist[i * vertexnum + ind[i * 4 + p]] > dist[i * vertexnum + ind[i * 4 + p + 1]]) {
                        int tmp = ind[p];
                        ind[p] = ind[p + 1];
                        ind[p + 1] = tmp;
                    }

            //find first 4 minimum distances
            for (int k = 4; k < vertexnum; k++)
                for (int t = 0; t < 4; t++)
                    if (dist[i * vertexnum + k] < dist[i * vertexnum + ind[i * 4 + t]]) {
                        for (int l = 3; l > t; l--)
                            ind[i * 4 + l] = ind[i * 4 + l - 1];
                        ind[i * 4 + t] = k;
                        continue;
                    }
        }

        __global__ void CalculateWeights(
                const PtrSz<const float> dist,
                const PtrSz<const float> weights,
                const PtrSz<const int> ind,
                const int jointnum,
                const int vertexnum,
                PtrSz<float> new_weights
        ) {
            int j = threadIdx.x; // num of weight
            int i = blockIdx.x; // num of vertex

            new_weights[i * jointnum + j] = 0;
            float weight = 0;
            for (int k = 0; k < 4; k++) {
                weight += dist[i * vertexnum + ind[i * 4 + k]];
                new_weights[i * jointnum + j] += dist[i * vertexnum + ind[i * 4 + k]] *
                        weights[ind[i * 4 + k] * jointnum + j];
            }
            new_weights[i * jointnum + j] /= weight;
        }


        __global__ void mark_body_nodes(
                const DeviceArrayView<float4> reference_vertex,
                const PtrSz<const float> templateRestShape,
                const PtrSz<const float> shapeBlendShape,
                const int vertex_num,
                const int max_dist,
                PtrSz<bool> on_body
        ) {
            int i = blockIdx.x; // reference vertex size
            int j = threadIdx.x; // smpl size
            if (3 * (j + 1) >= shapeBlendShape.size || i >= reference_vertex.Size())
                return;

            float dist = 0;
            const float cur[3] = {reference_vertex[i].x, reference_vertex[i].y, reference_vertex[i].z};
            for (int k = 0; k < 3; k++) {
                float restShape = templateRestShape[j * 3 + k] + shapeBlendShape[j * 3 + k];
                dist += (cur[k] - restShape) * (cur[k] - restShape);
            }
            if (dist <= max_dist)
                on_body[i] = true;
        }

        __global__ void copy_body_nodes(
                const DeviceArrayView<float4> reference_vertex,
                const PtrSz<const bool> on_body,
                PtrSz<float4> onbody_points,
                PtrSz<float4> farbody_points
        ) {
            int on = 0, far = 0;
            for (int i = 0; i < reference_vertex.Size(); i++) {
                if (on_body[i])
                    onbody_points[on++] = reference_vertex[i];
                else
                    farbody_points[far++] = reference_vertex[i];
            }
        }
    }

    DeviceArray<float> SMPL::lbs_for_custom_vertices(
            const DeviceArray<float> &beta,
            const DeviceArray<float> &theta,
            const DeviceArray<float> &d_vertices,
            cudaStream_t stream
    ) {
        DeviceArray<float> d_shapeBlendShape(DeviceArray<float>(VERTEX_NUM * 3));
        DeviceArray<float> d_dist(DeviceArray<float>(d_vertices.size() * VERTEX_NUM));
        DeviceArray<int> d_ind(DeviceArray<int>(d_vertices.size() * 4));
        DeviceArray<float> d_cur_weights(DeviceArray<float>(d_vertices.size() * JOINT_NUM));
        DeviceArray<float> d_result_vertices(DeviceArray<float>(d_vertices.size() * 3));

        shapeBlendShape(beta, d_shapeBlendShape, stream);
        // find k nearest neigbours
        device::FindKNN1<<<d_vertices.size(),VERTEX_NUM>>>(d_templateRestShape, d_shapeBlendShape, VERTEX_NUM, d_vertices, d_dist);
        device::FindKNN2<<<1,d_vertices.size()>>>(d_dist, VERTEX_NUM, d_ind);
        // calculate weights
        device::CalculateWeights<<<d_vertices.size(),JOINT_NUM>>>(d_dist, d_weights, d_ind,  JOINT_NUM, VERTEX_NUM, d_cur_weights);
        run(beta, theta, d_cur_weights, d_result_vertices, stream, d_vertices);

        return d_result_vertices;
    }

    void SMPL::SplitOnBodyVertices(
            const DeviceArrayView<float4>& reference_vertex,
            const DeviceArray<float> &beta,
            DeviceArray<float4>& onbody_points,
           DeviceArray<float4>& farbody_points,
       	cudaStream_t stream
    ) {
	std::cout << "start split\n";

        DeviceArray<float> d_shapeBlendShape = DeviceArray<float>(VERTEX_NUM * 3);
        DeviceArray<bool> marked_vertices = DeviceArray<bool>(reference_vertex.Size());

        shapeBlendShape(beta, d_shapeBlendShape, stream);

        device::mark_body_nodes<<<reference_vertex.Size(),VERTEX_NUM>>>(
                reference_vertex, d_templateRestShape,
                d_shapeBlendShape, VERTEX_NUM, 2.8f * Constants::kNodeRadius, marked_vertices);

        bool *host_array = (bool *)malloc(sizeof(bool) * marked_vertices.size());

	marked_vertices.download(host_array);
        int num = 0;
        for (int i = 0; i < marked_vertices.size(); i++)
            if (host_array[i])
                num++;

        onbody_points = DeviceArray<float4>(num);
        farbody_points = DeviceArray<float4>(reference_vertex.Size() - num);

        device::copy_body_nodes<<<1,1>>>(reference_vertex, marked_vertices, onbody_points, farbody_points);
cudaSafeCall(cudaStreamSynchronize(stream));
	std::cout << "end split\n";
    }
}
