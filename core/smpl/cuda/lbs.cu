#include <vector>
#include <cmath>
#include "core/smpl/def.h"
#include "core/smpl/smpl.h"
#include "common/Constants.h"
#include "math/vector_ops.hpp"
#include "common/algorithm_types.h"
#include <device_launch_parameters.h>

namespace surfelwarp {
    namespace device {
        __global__ void count_distance(
                const DeviceArrayView<float4> reference_vertex,
                const PtrSz<const float3> smpl_vertices,
                const int max_dist,
                PtrSz<float> dist,
                PtrSz<unsigned> marked
        ) {
            int i = blockIdx.x; // reference vertex size
            int j = threadIdx.x;
            if (i >= reference_vertex.Size())
                return;

            int vertexnum = smpl_vertices.size;
            int count = 27; // 6890 / 256
            marked[i] = 0;

	        float3 refv = make_float3(reference_vertex[i].x, reference_vertex[i].y, reference_vertex[i].z);
            for (int k = 0; k < count; k++) {
                if (j * 256 + k < vertexnum) {
                    dist[i * vertexnum + j * 256 + k] = squared_norm(refv - smpl_vertices[j * 256 + k]);
                    if (dist[i * vertexnum + j * 256 + k] <= max_dist)
                        marked[i] = 1;
                }
            }
        }

        __global__ void copy_body_nodes(
                const DeviceArrayView<float4> reference_vertex,
                const PtrSz<const unsigned> on_body,
                PtrSz<float4> onbody_points,
                PtrSz<float4> farbody_points
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
                const PtrSz<const unsigned> on_body,
                PtrSz<int> knn_ind,
                PtrSz<int> reverse_ind
        ) {
            int on = 0;
            for (int i = 0; i < reference_vertex.Size(); i++) {
                if (on_body[i]) {
                    knn_ind[on] = i;
                    reverse_ind[i] = on++;
                } else
                    reverse_ind[i] = -1;
            }
        }

        __global__ void findKNN(
                const PtrSz<const float> dist,
                const PtrSz<const int> knn_ind,
                const int vertexnum,
                ushort4* knn,
                float4* weights
        ) {
            const auto i = threadIdx.x + blockDim.x * blockIdx.x;
            if (i >= knn_ind.size)
                return;

            int dist_ind = knn_ind[i] * vertexnum;

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

            float4 weight;
            weight.x = __expf(-dist[dist_ind + knn[i].x]/ (2 * d_node_radius_square));
            weight.y = __expf(-dist[dist_ind + knn[i].y] / (2 * d_node_radius_square));
            weight.z = __expf(-dist[dist_ind + knn[i].z] / (2 * d_node_radius_square));
            weight.w = __expf(-dist[dist_ind + knn[i].w] / (2 * d_node_radius_square));
            weights[i] = weight;
        }
    }

    void SMPL::markVertices(
            const DeviceArrayView<float4>& live_vertex,
            cudaStream_t stream
    ) {
        m_dist = DeviceArray<float>(live_vertex.Size() * VERTEX_NUM);
        m_marked_vertices = DeviceArray<unsigned>(live_vertex.Size());

        device::count_distance<<<live_vertex.Size(),256,0,stream>>>(live_vertex, m_smpl_vertices,
                2.8f * Constants::kNodeRadius, m_dist, m_marked_vertices);

        //Prefix sum
        PrefixSum pref_sum;
        pref_sum.InclusiveSum(m_marked_vertices, stream);
        const auto& prefixsum_label = pref_sum.valid_prefixsum_array;
        cudaSafeCall(cudaMemcpyAsync(
                &m_num_marked,
                prefixsum_label.ptr() + prefixsum_label.size() - 1,
                sizeof(unsigned),
                cudaMemcpyDeviceToHost,
                stream
        ));
        cudaStreamSynchronize(stream);
        cudaError_t error = cudaGetLastError();
        if(error != cudaSuccess)
        {
            // print the CUDA error message and exit
            printf("CUDA error: %s\n", cudaGetErrorString(error));
            exit(-1);
        }
    }

    void SMPL::Split(
            const DeviceArrayView<float4>& live_vertex,
            const DeviceArrayView<float4>& reference_vertex,
            const int frame_idx,
            DeviceArray<float4>& onbody_points,
            DeviceArray<float4>& farbody_points,
            mat34 world2camera,
            cudaStream_t stream
    ) {
        if (m_vert_frame != frame_idx) {
            lbsModel(world2camera, stream);
            markVertices(live_vertex, stream);
            m_vert_frame = frame_idx;
        }
        onbody_points = DeviceArray<float4>(m_num_marked);
        farbody_points = DeviceArray<float4>(reference_vertex.Size() - m_num_marked);
        device::copy_body_nodes<<<1,1,0,stream>>>(reference_vertex, m_marked_vertices,
                onbody_points, farbody_points);
    }

    void SMPL::countKnn(
            const DeviceArrayView<float4>& live_vertex,
            const int frame_idx,
            mat34 world2camera,
            cudaStream_t stream
    ) {
        if (m_vert_frame != frame_idx) {
            lbsModel(world2camera, stream);
            markVertices(live_vertex, stream);
            m_vert_frame = frame_idx;
        }
        auto knn_ind = DeviceArray<int>(m_num_marked);
        m_onbody = DeviceArray<int>(live_vertex.Size());
        device::fill_index<<<1,1,0,stream>>>(live_vertex, m_marked_vertices, knn_ind, m_onbody);

        m_knn = DeviceArray<ushort4>(m_num_marked);
        m_knn_weight = DeviceArray<float4>(m_num_marked);

        //find 4 nearest neighbours
        if (m_num_marked > 0) {
            dim3 blk(64);
            dim3 grid(divUp(m_num_marked, blk.x));
            device::findKNN<<<grid,blk,0,stream>>>(m_dist, knn_ind, VERTEX_NUM, m_knn.ptr(), m_knn_weight.ptr());
        }
    }
}