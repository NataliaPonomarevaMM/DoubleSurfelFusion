#include <vector>
#include <cmath>
#include "core/smpl/def.h"
#include "core/smpl/smpl.h"
#include "common/Constants.h"
#include "math/vector_ops.hpp"
#include "common/algorithm_types.h"
#include <device_launch_parameters.h>
#include <stdio.h>
#include <math.h>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <fstream>
#include <cstdlib>
#include <chrono>

using mcs = std::chrono::microseconds;
using ms = std::chrono::milliseconds;
using clk = std::chrono::system_clock;


namespace surfelwarp {
    namespace device {
        __global__ void count_distance(
                const DeviceArrayView<float4> reference_vertex,
                const PtrSz<const float3> smpl_vertices,
                const int max_dist,
                float* dist,
                PtrSz<unsigned> marked
        ) {
            int i = blockIdx.x; // reference vertex size
            int j = threadIdx.x;
            if (i >= reference_vertex.Size())
                return;

            int vertexnum = smpl_vertices.size;
            int count = 27; // 6890 / 256

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
                const int *on_body,
                PtrSz<float4> onbody_points,
                PtrSz<float4> farbody_points
        ) {
            int on = 0, far = 0;
            for (int i = 0; i < reference_vertex.Size(); i++) {
                if (on_body[i] != -1)
                    onbody_points[on] = reference_vertex[i];
                else
                    farbody_points[far++] = reference_vertex[i];
            }
        }

        __global__ void fill_index(
                const DeviceArrayView<float4> reference_vertex,
                const PtrSz<const unsigned> on_body,
                PtrSz<int> knn_ind,
                int* reverse_ind
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

        __global__ void copy_vert1(
                const DeviceArrayView<float4> live_vertex,
                float *new_vert
        ) {
            const auto i = threadIdx.x + blockDim.x * blockIdx.x;
            if (i >= live_vertex.Size())
                return;
            new_vert[i * 3] = live_vertex[i].x;
            new_vert[i * 3 + 1] = live_vertex[i].y;
            new_vert[i * 3 + 2] = live_vertex[i].z;
        }

        __global__ void copy_vert2(
                const PtrSz<const float3> smpl_vertex,
                float *new_vert
        ) {
            const auto i = threadIdx.x + blockDim.x * blockIdx.x;
            if (i >= smpl_vertex.size)
                return;
            new_vert[i * 3] = smpl_vertex[i].x;
            new_vert[i * 3 + 1] = smpl_vertex[i].y;
            new_vert[i * 3 + 2] = smpl_vertex[i].z;
        }
    }

    unsigned SMPL::count_dist(
            const DeviceArrayView<float4>& live_vertex,
            DeviceArray<unsigned> &marked,
            DeviceArray<float> &dist,
            cudaStream_t stream
    ) {
        cudaMemset(marked.ptr(), 0, sizeof(float) * live_vertex.Size());
        device::count_distance<<<live_vertex.Size(),256,0,stream>>>(live_vertex, m_smpl_vertices,
                2.8f * Constants::kNodeRadius, dist.ptr(), marked);

        //Prefix sum
        PrefixSum pref_sum;
        unsigned num = 0;
        pref_sum.InclusiveSum(marked, stream);
        const auto& prefixsum_label = pref_sum.valid_prefixsum_array;
        cudaSafeCall(cudaMemcpyAsync(
                &num,
                prefixsum_label.ptr() + prefixsum_label.size() - 1,
                sizeof(unsigned),
                cudaMemcpyDeviceToHost,
                stream
        ));
        return num;
    }

    void SMPL::count_knn(
            const DeviceArrayView<float4>& live_vertex,
            const DeviceArray<float> &dist,
            const DeviceArray<unsigned> &marked,
            const unsigned num_marked,
            DeviceArray<int> &onbody,
            DeviceArray<ushort4> &knn,
            DeviceArray<float4> &knn_weight,
            cudaStream_t stream
    ) {
        auto knn_ind = DeviceArray<int>(num_marked);
        device::fill_index<<<1,1,0,stream>>>(live_vertex, marked, knn_ind, onbody.ptr());

        //find 4 nearest neighbours
        if (num_marked > 0) {
            dim3 blk(64);
            dim3 grid(divUp(num_marked, blk.x));
            device::findKNN<<<grid,blk,0,stream>>>(dist, knn_ind, VERTEX_NUM, knn.ptr(), knn_weight.ptr());
        }
    }

    void SMPL::CountKnn(
            const surfelwarp::DeviceArrayView<float4> &live_vertex,
            cudaStream_t stream)
    {
        auto marked = DeviceArray<unsigned>(live_vertex.Size());
        m_dist.ResizeArrayOrException(live_vertex.Size() * VERTEX_NUM);
        m_onbody.ResizeArrayOrException(live_vertex.Size());

        auto cur_dist = m_dist.Array();
        m_num_marked = count_dist(live_vertex, marked, cur_dist, stream);
        auto cur_onbody = m_onbody.Array();

        m_knn.ResizeArrayOrException(m_num_marked);
        m_knn_weight.ResizeArrayOrException(m_num_marked);
        auto cur_knn = m_knn.Array();
        auto cur_knn_weight = m_knn_weight.Array();
        count_knn(live_vertex, cur_dist, marked, m_num_marked, cur_onbody,
                cur_knn, cur_knn_weight, stream);
    }

    void SMPL::CountAppendedKnn(
            const DeviceArrayView<float4>& appended_live_vertex,
            int num_remaining_surfel,
            int num_appended_surfel,
            cudaStream_t stream)
    {
        const int all_surfels = num_remaining_surfel + num_appended_surfel;
        m_dist.ResizeArrayOrException(all_surfels * VERTEX_NUM);
        m_onbody.ResizeArrayOrException(all_surfels);
        auto marked = DeviceArray<unsigned>(num_appended_surfel);

        float *dist_ptr = m_dist.Ptr() + num_remaining_surfel * VERTEX_NUM;
        auto cur_dist = DeviceArray<float>(dist_ptr, num_appended_surfel * VERTEX_NUM);

        auto cur_num_marked = count_dist(appended_live_vertex, marked, cur_dist, stream);
        m_num_marked += cur_num_marked;

        int *onbody_ptr = m_onbody.Ptr() + num_remaining_surfel;
        auto cur_onbody = DeviceArray<int>(onbody_ptr, num_appended_surfel);

        m_knn.ResizeArrayOrException(m_num_marked);
        m_knn_weight.ResizeArrayOrException(m_num_marked);

        ushort4 *knn_ptr = m_knn.Ptr() + cur_num_marked;
        auto cur_knn = DeviceArray<ushort4>(knn_ptr, num_appended_surfel);
        float4 *knn_weight_ptr = m_knn_weight.Ptr() + cur_num_marked;
        auto cur_knn_weight = DeviceArray<float4>(knn_weight_ptr, num_appended_surfel);
        count_knn(appended_live_vertex, cur_dist, marked, cur_num_marked, cur_onbody,
                  cur_knn, cur_knn_weight, stream);
    }

    void SMPL::SplitReferenceVertices(
            const DeviceArrayView<float4>& live_vertex,
            const DeviceArrayView<float4>& reference_vertex,
            DeviceArray<float4>& onbody_points,
            DeviceArray<float4>& farbody_points,
            cudaStream_t stream
    ) {
        onbody_points = DeviceArray<float4>(m_num_marked);
        farbody_points = DeviceArray<float4>(reference_vertex.Size() - m_num_marked);
        device::copy_body_nodes<<<1,1,0,stream>>>(reference_vertex, m_onbody.Ptr(),
                onbody_points, farbody_points);
    }
}


//        m_dist = DeviceArray<float>(live_vertex.Size() * VERTEX_NUM);
//        auto knn_ind = DeviceArray<int>(live_vertex.Size() * 4);
//
//        const int lsize = live_vertex.Size();
//        auto cur_smpl_vert = DeviceArray<float>(VERTEX_NUM * 3);
//        auto cur_live_vert = DeviceArray<float>(lsize * 3);
//        dim3 blk(256);
//        dim3 grid(divUp(lsize, blk.x));
//        device::copy_vert1<<<grid,blk,0,stream>>>(live_vertex, cur_live_vert.ptr());
//        dim3 blk2(256);
//        dim3 grid2(divUp(VERTEX_NUM, blk.x));
//        device::copy_vert2<<<grid2,blk2,0,stream>>>(m_smpl_vertices, cur_smpl_vert.ptr());
//        knn_cuda_texture(cur_smpl_vert, VERTEX_NUM, cur_live_vert, lsize, 3, 4, m_dist.ptr(), knn_ind.ptr());
