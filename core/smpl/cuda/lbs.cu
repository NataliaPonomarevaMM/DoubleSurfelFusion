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
        __global__ void mark(
                const float* dist,
                const int live_size,
                const float squared_max_dist,
                PtrSz<unsigned> marked
        ) {
            const auto dist_ind = threadIdx.x + blockDim.x * blockIdx.x;
            const int mark_ind = dist_ind % live_size;

            if (dist_ind >= live_size * 4)
                return;

            if (dist[dist_ind] <= squared_max_dist)
                marked[mark_ind] = 1;
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
                    onbody_points[on++] = reference_vertex[i];
                else
                    farbody_points[far++] = reference_vertex[i];
            }
        }

        __global__ void fill_index(
                const PtrSz<const unsigned> on_body,
                PtrSz<int> reverse_onbody_ind,
                int* onbody_ind
        ) {
            int on = 0;
            for (int i = 0; i < on_body.size; i++) {
                if (on_body[i]) {
                    reverse_onbody_ind[on] = i;
                    onbody_ind[i] = on++;
                } else {
                    onbody_ind[i] = -1;
                }
            }
        }

        __global__ void copyKNN(
                const PtrSz<const float> dist,
                const PtrSz<const int> reverse_onbody,
                const PtrSz<const int> knn_ind,
                ushort4* knn,
                float4* weights
        ) {
            const auto i = threadIdx.x + blockDim.x * blockIdx.x;
            if (i >= reverse_onbody.size)
                return;

            const int num = dist.size / 4;
            auto cur_ind = reverse_onbody[i];
            knn[i] = make_ushort4(knn_ind[0 * num + cur_ind], knn_ind[1 * num + cur_ind],
                    knn_ind[2 * num + cur_ind], knn_ind[3 * num + cur_ind]);

            float4 weight;
            weight.x = __expf(-dist[0 * num + cur_ind]/ (2 * d_node_radius_square));
            weight.y = __expf(-dist[1 * num + cur_ind] / (2 * d_node_radius_square));
            weight.z = __expf(-dist[2 * num + cur_ind] / (2 * d_node_radius_square));
            weight.w = __expf(-dist[3 * num + cur_ind] / (2 * d_node_radius_square));
            auto sum = weight.x + weight.y + weight.z + weight.w;
            weight.x /= sum;
            weight.y /= sum;
            weight.z /= sum;
            weight.w /= sum;
            weights[i] = weight;
        }

        __global__ void copy_live_vert(
                const DeviceArrayView<float4> live_vertex,
                float *new_vert
        ) {
            const auto i = threadIdx.x + blockDim.x * blockIdx.x;
            if (i >= live_vertex.Size())
                return;
            new_vert[0 * live_vertex.Size() + i] = live_vertex[i].x;
            new_vert[1 * live_vertex.Size() + i] = live_vertex[i].y;
            new_vert[2 * live_vertex.Size() + i] = live_vertex[i].z;
        }

        __global__ void copy_smpl_vert(
                const PtrSz<const float3> smpl_vertex,
                float *new_vert
        ) {
            const auto i = threadIdx.x + blockDim.x * blockIdx.x;
            if (i >= smpl_vertex.size)
                return;
            new_vert[0 * 6890 + i] = smpl_vertex[i].x;
            new_vert[1 * 6890 + i] = smpl_vertex[i].y;
            new_vert[2 * 6890 + i] = smpl_vertex[i].z;
        }
    }

    unsigned SMPL::count_dist(
            const DeviceArrayView<float4>& live_vertex,
            DeviceArray<unsigned> &marked,
            DeviceArray<float> &dist,
            DeviceArray<int> &knn_ind,
            cudaStream_t stream
    ) {
        auto cur_smpl_vert = DeviceArray<float>(VERTEX_NUM * 3);
        auto cur_live_vert = DeviceArray<float>(live_vertex.Size() * 3);
        dim3 blk(256);
        dim3 grid(divUp(live_vertex.Size(), blk.x));
        device::copy_live_vert<<<grid,blk,0,stream>>>(live_vertex, cur_live_vert.ptr());
        dim3 blk2(256);
        dim3 grid2(divUp(VERTEX_NUM, blk.x));
        device::copy_smpl_vert<<<grid2,blk2,0,stream>>>(m_smpl_vertices, cur_smpl_vert.ptr());
        knn_cuda_texture(cur_smpl_vert, VERTEX_NUM, cur_live_vert, live_vertex.Size(), 3, 4, dist.ptr(), knn_ind.ptr());

        cudaMemset(marked.ptr(), 0, sizeof(unsigned) * live_vertex.Size());
        dim3 blk3(256);
        dim3 grid3(divUp(4 * live_vertex.Size(), blk.x));
        device::mark<<<grid3,blk3,0,stream>>>(dist.ptr(), live_vertex.Size(),
                5.0f * Constants::kNodeRadius, marked);
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
            const DeviceArray<float> &dist,
            const DeviceArray<unsigned> &marked,
            const DeviceArray<int> &knn_ind,
            const unsigned num_marked,
            DeviceArray<int> &onbody,
            DeviceArray<ushort4> &knn,
            DeviceArray<float4> &knn_weight,
            cudaStream_t stream
    ) {
        auto reverse_onbody = DeviceArray<int>(num_marked);
        device::fill_index<<<1,1,0,stream>>>(marked, reverse_onbody, onbody.ptr());

        //find 4 nearest neighbours
        if (num_marked > 0) {
            dim3 blk(256);
            dim3 grid(divUp(num_marked, blk.x));
            device::copyKNN<<<grid,blk,0,stream>>>(dist, reverse_onbody, knn_ind, knn.ptr(), knn_weight.ptr());
        }
    }

    void SMPL::CountKnn(
            const surfelwarp::DeviceArrayView<float4> &live_vertex,
            cudaStream_t stream)
    {
        const int lsize = live_vertex.Size();
        m_dist.ResizeArrayOrException(lsize * 4);
        m_onbody.ResizeArrayOrException(lsize);
        auto knn_ind = DeviceArray<int>(lsize * 4);
        auto marked = DeviceArray<unsigned>(lsize);
        auto cur_dist = m_dist.Array();
        m_num_marked = count_dist(live_vertex, marked, cur_dist, knn_ind, stream);

        auto cur_onbody = m_onbody.Array();
        m_knn.ResizeArrayOrException(m_num_marked);
        m_knn_weight.ResizeArrayOrException(m_num_marked);
        auto cur_knn = m_knn.Array();
        auto cur_knn_weight = m_knn_weight.Array();
        count_knn(cur_dist, marked, knn_ind, m_num_marked, cur_onbody,
                cur_knn, cur_knn_weight, stream);
    }

    void SMPL::CountAppendedKnn(
            const DeviceArrayView<float4>& appended_live_vertex,
            int num_remaining_surfel,
            int num_appended_surfel,
            cudaStream_t stream)
    {
        if (appended_live_vertex.Size() == 0)
            return;

        const int all_surfels = num_remaining_surfel + num_appended_surfel;
        m_dist.ResizeArrayOrException(all_surfels * 4);
        m_onbody.ResizeArrayOrException(all_surfels);
        auto knn_ind = DeviceArray<int>(num_appended_surfel * 4);
        auto marked = DeviceArray<unsigned>(num_appended_surfel);
        float *dist_ptr = m_dist.Ptr() + num_remaining_surfel * 4;
        auto cur_dist = DeviceArray<float>(dist_ptr, num_appended_surfel * 4);
        auto cur_num_marked = count_dist(appended_live_vertex, marked, cur_dist, knn_ind, stream);
        m_num_marked += cur_num_marked;

        int *onbody_ptr = m_onbody.Ptr() + num_remaining_surfel;
        auto cur_onbody = DeviceArray<int>(onbody_ptr, num_appended_surfel);
        m_knn.ResizeArrayOrException(m_num_marked);
        m_knn_weight.ResizeArrayOrException(m_num_marked);
        ushort4 *knn_ptr = m_knn.Ptr() + cur_num_marked;
        auto cur_knn = DeviceArray<ushort4>(knn_ptr, num_appended_surfel);
        float4 *knn_weight_ptr = m_knn_weight.Ptr() + cur_num_marked;
        auto cur_knn_weight = DeviceArray<float4>(knn_weight_ptr, num_appended_surfel);
        count_knn(cur_dist, marked, knn_ind, cur_num_marked, cur_onbody,
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
