#include <vector>
#include <cmath>
#include "core/smpl/def.h"
#include "core/smpl/smpl.h"
#include "common/common_types.h"
#include "common/Constants.h"

namespace surfelwarp {
    namespace device {
        __global__ void count_distance(
                const DeviceArrayView<float4> reference_vertex,
                const PtrSz<const float> restShape,
                const int max_dist,
                const int vertexnum,
                PtrSz<float> dist,
                PtrSz<bool> marked
        ) {
            int i = blockIdx.x; // reference vertex size
            if (i >= reference_vertex.Size())
                return;
            const float cur[3] = {reference_vertex[i].x, reference_vertex[i].y, reference_vertex[i].z};

            for (int j = 0; j < vertexnum; j++) {
                dist[i * vertexnum + j] = 0;
                for (int k = 0; k < 3; k++) {
                    float r = cur[k] - restShape[j * 3 + k];
                    dist[i * vertexnum + j] += r * r;
                }
                if (dist[i * vertexnum + j] <= max_dist)
                    marked[i] = true;
            }
        }

        __global__ void copy_body_nodes(
                const DeviceArrayView<float4> reference_vertex,
                const PtrSz<const bool> on_body,
                PtrSz<float4> onbody_points,
                PtrSz<float4> farbody_points,
        ) {
            int on = 0, far = 0;
            for (int i = 0; i < reference_vertex.Size(); i++) {
                if (on_body[i])
                    onbody_points[on] = reference_vertex[i];
                else
                    farbody_points[far++] = reference_vertex[i];
            }
        }

        __global__ void fill_index(
                const DeviceArrayView<float4> reference_vertex,
                const PtrSz<const bool> on_body,
                PtrSz<int> knn_ind,
                PtrSz<int> reverse_ind
        ) {
            int on = 0;
            for (int i = 0; i < reference_vertex.Size(); i++) {
                if (on_body[i]) {
                    onbody_ind[on] = i;
                    reverse_ind[i] = on++;
                } else
                    reverse_ind[i] = -1;
            }
        }

        __global__ void findKNN(
                const PtrSz<const float> dist,
                const PtrSz<const int> knn_ind,
                const int vertexnum,
                PtrSz<ushort4> knn
        ) {
            int i = blockIdx.x; // num
            if (i >= onbody_ind.size)
                return;

            int dist_ind = onbody_ind[i] * vertexnum;

            int ind[4];
            for (int l = 0; l < 4; l++)
                ind[l] = l;
            for (int l = 0; l < 4; l++)
                for (int p = 0; p < 3 - l; p++)
                    if (dist[dist_ind + ind[p]] > dist[dist_ind + ind[p + 1]]) {
                        int tmp = ind[p];
                        ind[p] = ind[p + 1];
                        ind[p + 1] = tmp;
                    }

            //find first 4 minimum distances
            for (int k = 4; k < vertexnum; k++)
                for (int t = 0; t < 4; t++)
                    if (dist[dist_ind + k] < dist[dist_ind + ind[t]]) {
                        for (int l = 3; l > t; l--)
                            ind[l] = ind[l - 1];
                        ind[t] = k;
                        continue;
                    }
            knn[i] = make_ushort4(ind[0], ind[1], ind[2], ind[3]);
        }

        __global__ void calculate_weights(
                const PtrSz<const float> dist,
                const PtrSz<const ushort4> knn,
                const PtrSz<const int> knn_ind,
                const int vertexnum,
                PtrSz<float4> weights
        ) {
            int i = blockIdx.x; // num of reference vertex
            if (i >= on_body.size || !on_body[i])
                return;

            int dist_ind = onbody_ind[i] * vertexnum;
            float4 weight;
            weight.x = __expf(-dist[dist_ind + knn[i].x]/ (2 * d_node_radius_square));
            weight.y = __expf(-dist[dist_ind + knn[i].y] / (2 * d_node_radius_square));
            weight.z = __expf(-dist[dist_ind + knn[i].z] / (2 * d_node_radius_square));
            weight.w = __expf(-dist[dist_ind + knn[i].w] / (2 * d_node_radius_square));
            weights[i] = weight;
        }
    }

    void SMPL::Split(
            const DeviceArrayView<float4>& reference_vertex,
            DeviceArray<float4>& onbody_points,
            DeviceArray<float4>& farbody_points,
            cudaStream_t stream
    ) {
        auto dist = DeviceArray<float>(vertices.size() * VERTEX_NUM);
        auto marked_vertices = DeviceArray<bool>(reference_vertex.Size());

        device::count_distance<<<vertices.size(),1,0,stream>>>(reference_vertex, m_restShape,
                2.8f * Constants::kNodeRadius, VERTEX_NUM, dist, marked_vertices);

        //split on ondoby and farbody
        bool *host_array = (bool *)malloc(sizeof(bool) * marked_vertices.size());
        marked_vertices.download(host_array);
        int num = 0;
        for (int i = 0; i < marked_vertices.size(); i++)
            if (host_array[i])
                num++;

        onbody_points = DeviceArray<float4>(num);
        farbody_points = DeviceArray<float4>(reference_vertex.Size() - num);
        device::copy_body_nodes<<<1,1,0,stream>>>(reference_vertex, marked_vertices,
                onbody_points, farbody_points);
    }

    void SMPL::CountKnn(
            const DeviceArrayView<float4>& reference_vertex,
            cudaStream_t stream
    ) {
        auto dist = DeviceArray<float>(vertices.size() * VERTEX_NUM);
        auto marked_vertices = DeviceArray<bool>(reference_vertex.Size());

        device::count_distance<<<vertices.size(),1,0,stream>>>(reference_vertex, m_restShape,
                2.8f * Constants::kNodeRadius, VERTEX_NUM, dist, marked_vertices);

        //split on ondoby and farbody
        bool *host_array = (bool *)malloc(sizeof(bool) * marked_vertices.size());
        marked_vertices.download(host_array);
        int num = 0;
        for (int i = 0; i < marked_vertices.size(); i++)
            if (host_array[i])
                num++;

        auto knn_ind = DeviceArray<int>(num);
        m_onbody = DeviceArray<int>(reference_vertex.Size());
        device::fill_index<<<1,1,0,stream>>>(reference_vertex, marked_vertices, knn_ind, m_onbody);

        m_knn = DeviceArray<ushort4>(num);
        m_knn_weight = DeviceArray<float4>(num);

        //find 4 nearest neighbours
        device::findKNN<<<vertices.size(),1,0,stream>>>(dist, knn_ind, VERTEX_NUM, m_knn);
        device::calculate_weights<<<vertices.size(),1,0,stream>>>(dist, m_knn, knn_ind,
                VERTEX_NUM, m_knn_weight);
    }
}
