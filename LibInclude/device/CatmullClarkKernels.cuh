// Copyright (c) 2020  Daniel Mlakar daniel.mlakar@icg.tugraz.at
					// Martin Winter martin.winter@icg.tugraz.at
					// Pascal Stadlbauer pascal.stadlbauer@icg.tugraz.at
					// Hans-Peter Seidel hpseidel@mpi-inf.mpg.de
					// Markus Steinberger steinberger@icg.tugraz.at
					// Rhaleb Zayer rzayer@mpi-inf.mpg.de

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#include <cstdio>

#include <builtin_types.h>
#include <host_defines.h>

#include "math/vector.h"

namespace HelperKernels
{
	template<typename T>
	__global__ void setDeviceMem_d(void* ptr, T value, int num)
	{
		int tid = blockIdx.x * blockDim.x + threadIdx.x;
		if (tid >= num) return;

		reinterpret_cast<T*>(ptr)[tid] = value;
	}
	template<typename T>
	void setDeviceMem(void* ptr, T value, int num, int block_dim = -1)
	{
		block_dim = block_dim > 0 ? block_dim : 256;
		size_t grid_dim = (num + block_dim - 1) / block_dim;
		setDeviceMem_d<T> << <grid_dim, block_dim >> > (ptr, value, num);
	}


	struct smaller_than
	{
		template <typename T>
		__device__ static bool comp(T a, T b) { return a < b; }
	};

	struct greater_than
	{
		template <typename T>
		__device__ static bool comp(T a, T b) { return a > b; }
	};

	template<typename Tk, typename Tv, class Comperator>
	__device__ bool compareAndSwapPair(Tk* a, Tk* b, Tv* av, Tv* bv)
	{
		bool swap = Comperator::comp(*a, *b);
		if (swap)
		{
			Tk tmp = *a;
			*a = *b;
			*b = tmp;
			Tv tmpv = *av;
			*av = *bv;
			*bv = tmpv;
		}
		return swap;
	}

	template<typename Tk, typename Tv>
	__device__ void bubble_sort(Tk* start_k, Tv* start_v, unsigned nelements)
	{
		for (auto i = 0; i < nelements - 1; ++i)
		{
			bool done = true;
			for (auto j = 1; j < nelements - i; ++j)
			{
				done &= !compareAndSwapPair<Tk, Tv, greater_than>(&start_k[j - 1], &start_k[j], &start_v[j - 1], &start_v[j]);
			}
			if (done) { break; };
		}
	}

	template<typename INDEX_TYPE, typename KEY_TYPE, typename VALUE_TYPE>
	__global__ void sortKeyValuePairsSegmentedInPlace_d(const INDEX_TYPE* offsets, KEY_TYPE* keys, VALUE_TYPE* values, unsigned nsegments)
	{
		unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
		if (tid >= nsegments)
			return;

		auto start = offsets[tid];
		auto stop = offsets[tid + 1];
		auto nelements = stop - start;
		bubble_sort<KEY_TYPE, VALUE_TYPE>(&keys[start], &values[start], nelements);
	}

	template<typename INDEX_TYPE, typename KEY_TYPE, typename VALUE_TYPE>
	void sortKeyValuePairsSegmentedInPlace(const INDEX_TYPE* offsets, KEY_TYPE* keys, VALUE_TYPE* values, unsigned nsegments, int block_dim = -1)
	{
		block_dim = block_dim > 0 ? block_dim : 256;
		size_t grid_dim = (nsegments + block_dim - 1) / block_dim;
		sortKeyValuePairsSegmentedInPlace_d << <grid_dim, block_dim >> > (offsets, keys, values, nsegments);
	}
}

namespace CCKernels
{
	__device__ __forceinline__ math::float4 to_math_f4(float4 v) { return math::float4(v.x, v.y, v.z, v.w); }
	__device__ __forceinline__ float4 to_f4(math::float4 v) { return{ v.x, v.y, v.z, v.w }; }

	__device__ __forceinline__ void atomicAddVec(float* address, math::float4 val)
	{
		atomicAdd(address + 0, val.x);
		atomicAdd(address + 1, val.y);
		atomicAdd(address + 2, val.z);
		atomicAdd(address + 3, val.w);
	}

	__global__ void someTest()
	{
		const auto tid = threadIdx.x + blockIdx.x * blockDim.x;
		printf("Thread %d is here!\n", tid);
	}

	template<typename INDEX_TYPE>
	__global__ void evaluateValences(const INDEX_TYPE* __restrict M_row_ids, INDEX_TYPE* valences, INDEX_TYPE* valences_sub_offset, unsigned nnz)
	{
		unsigned int threadId = blockIdx.x * blockDim.x + threadIdx.x;
		if (threadId >= nnz)
			return;

		valences_sub_offset[threadId] = atomicAdd(&valences[M_row_ids[threadId]], 1);
	}

	template<typename INDEX_TYPE>
	__global__ void evaluateAdjacenciesQuad(const INDEX_TYPE* __restrict M_row_ids, INDEX_TYPE* adj_off, INDEX_TYPE* adj_sub_off, INDEX_TYPE* S_row_ids, INDEX_TYPE* S_epids_lt, unsigned nnz)
	{
		unsigned int threadId = blockIdx.x * blockDim.x + threadIdx.x;
		if (threadId >= nnz)
			return;

		auto v0 = M_row_ids[threadId];
		auto v1 = M_row_ids[(threadId / 4 * 4) + ((threadId + 1) % 4)];
		auto adj_id = adj_sub_off[threadId];

		S_row_ids[adj_off[v0] + adj_id] = v1;
		S_epids_lt[adj_off[v0] + adj_id] = v0 < v1;
	}

	template<typename INDEX_TYPE>
	__global__ void evaluateAdjacencies(const INDEX_TYPE* __restrict M_col_ptr, const INDEX_TYPE* __restrict M_row_ids, INDEX_TYPE* adj_off, INDEX_TYPE* adj_sub_off,
		INDEX_TYPE* S_row_ids, INDEX_TYPE* S_epids, INDEX_TYPE* face_orders, unsigned ncols)
	{
		unsigned int threadId = blockIdx.x * blockDim.x + threadIdx.x;
		if (threadId >= ncols)
			return;

		auto face_order = M_col_ptr[threadId + 1] - M_col_ptr[threadId];
		face_orders[threadId] = face_order;

		for (auto i = M_col_ptr[threadId]; i < M_col_ptr[threadId + 1]; ++i)
		{
			auto v0 = M_row_ids[i];
			auto v1 = 0;
			if (i == M_col_ptr[threadId + 1] - 1)
				v1 = M_row_ids[M_col_ptr[threadId]];
			else
				v1 = M_row_ids[i + 1];

			auto adj_id = adj_sub_off[i];

			S_row_ids[adj_off[v0] + adj_id] = v1;
			S_epids[adj_off[v0] + adj_id] = v0 < v1;
		}
	}

	template<typename INDEX_TYPE>
	__global__ void createThreadFaceMapping(const INDEX_TYPE* __restrict face_offsets, INDEX_TYPE* mapping, unsigned nfaces)
	{
		unsigned int threadId = blockIdx.x * blockDim.x + threadIdx.x;
		if (threadId >= nfaces)
			return;

		for (auto i = face_offsets[threadId]; i < face_offsets[threadId + 1]; ++i)
			mapping[i] = threadId;
	}

	template<typename INDEX_TYPE>
	__global__ void checkForBoundaryQuad(const INDEX_TYPE* __restrict M_row_ids,
		const INDEX_TYPE* __restrict S_col_ptr, const INDEX_TYPE* __restrict S_row_ids, INDEX_TYPE* S_epids_lt, INDEX_TYPE* nextern, unsigned nnz)
	{
		unsigned int threadId = blockIdx.x * blockDim.x + threadIdx.x;
		if (threadId >= nnz)
			return;

		INDEX_TYPE v0 = M_row_ids[threadId];
		INDEX_TYPE vout = M_row_ids[(threadId / 4 * 4) + ((threadId + 1) % 4)];

		INDEX_TYPE eid = 0;
		for (INDEX_TYPE i = S_col_ptr[v0]; i < S_col_ptr[v0 + 1]; ++i)
		{
			if (S_row_ids[i] == vout)
			{
				eid = i;
				break;
			}
		}

		bool is_extern = true;
		for (INDEX_TYPE i = S_col_ptr[vout]; i < S_col_ptr[vout + 1]; ++i)
		{
			if (S_row_ids[i] == v0)
			{
				is_extern = false;
				break;
			}
		}

		if (!is_extern)
			return;

		atomicAdd(nextern, 1);
		S_epids_lt[eid] = 1;
	}

	template<typename INDEX_TYPE>
	__global__ void checkForBoundary(const INDEX_TYPE* __restrict mapping, const INDEX_TYPE* __restrict M_row_ids, const INDEX_TYPE* __restrict face_offs,
		const INDEX_TYPE* __restrict S_col_ptr, const INDEX_TYPE* __restrict S_row_ids, INDEX_TYPE* S_epids_lt, INDEX_TYPE* nextern, unsigned nnz)
	{
		unsigned int threadId = blockIdx.x * blockDim.x + threadIdx.x;
		if (threadId >= nnz)
			return;

		auto v0 = M_row_ids[threadId];

		auto my_face = mapping[threadId];
		auto face_order = face_offs[my_face + 1] - face_offs[my_face];
		auto vout = M_row_ids[face_offs[my_face] + (threadId + 1 - face_offs[my_face]) % face_order];

		INDEX_TYPE eid = 0;
		for (auto i = S_col_ptr[v0]; i < S_col_ptr[v0 + 1]; ++i)
		{
			if (S_row_ids[i] == vout)
			{
				eid = i;
				break;
			}
		}

		bool is_extern = true;
		for (auto i = S_col_ptr[vout]; i < S_col_ptr[vout + 1]; ++i)
		{
			if (S_row_ids[i] == v0)
			{
				is_extern = false;
				break;
			}
		}

		if (!is_extern)
			return;

		atomicAdd(nextern, 1);
		S_epids_lt[eid] = 1;
	}


	template<typename INDEX_TYPE>
	__global__ void checkForBoundaryFillFaceAdjacency(const INDEX_TYPE* __restrict mapping, const INDEX_TYPE* __restrict M_row_ids, const INDEX_TYPE* __restrict face_offs,
		const INDEX_TYPE* __restrict S_col_ptr, const INDEX_TYPE* __restrict S_row_ids, INDEX_TYPE* S_epids_lt, INDEX_TYPE* S_face_adj, INDEX_TYPE* nextern, unsigned nnz)
	{
		unsigned int threadId = blockIdx.x * blockDim.x + threadIdx.x;
		if (threadId >= nnz)
			return;

		auto v0 = M_row_ids[threadId];

		auto my_face = mapping[threadId];
		auto face_order = face_offs[my_face + 1] - face_offs[my_face];
		auto vout = M_row_ids[face_offs[my_face] + (threadId + 1 - face_offs[my_face]) % face_order];

		INDEX_TYPE eid = 0;
		for (auto i = S_col_ptr[v0]; i < S_col_ptr[v0 + 1]; ++i)
		{
			if (S_row_ids[i] == vout)
			{
				eid = i;
				break;
			}
		}

		bool is_extern = true;
		for (auto i = S_col_ptr[vout]; i < S_col_ptr[vout + 1]; ++i)
		{
			if (S_row_ids[i] == v0)
			{
				is_extern = false;
				S_face_adj[i] = my_face;
				break;
			}
		}

		if (!is_extern)
			return;

		S_face_adj[eid] = -(my_face + 1);
		atomicAdd(nextern, 1);
		S_epids_lt[eid] = 1;
	}


	template<typename INDEX_TYPE>
	__global__ void markBoundaryQuad(const INDEX_TYPE* __restrict M_row_ids,
		const INDEX_TYPE* __restrict S_col_ptr, const INDEX_TYPE* __restrict S_row_ids, INDEX_TYPE* S_epids_lt, INDEX_TYPE* everts, INDEX_TYPE* nextern, unsigned nnz)
	{
		unsigned int threadId = blockIdx.x * blockDim.x + threadIdx.x;
		if (threadId >= nnz)
			return;

		const INDEX_TYPE v0 = M_row_ids[threadId];
		const INDEX_TYPE vout = M_row_ids[(threadId / 4 * 4) + ((threadId + 1) % 4)];

		INDEX_TYPE eid = 0;
		for (INDEX_TYPE i = S_col_ptr[v0]; i < S_col_ptr[v0 + 1]; ++i)
		{
			if (S_row_ids[i] == vout)
			{
				eid = i;
				break;
			}
		}

		bool is_extern = true;
		for (INDEX_TYPE i = S_col_ptr[vout]; i < S_col_ptr[vout + 1]; ++i)
		{
			if (S_row_ids[i] == v0)
			{
				is_extern = false;
				break;
			}
		}

		if (!is_extern)
			return;

		INDEX_TYPE loc = atomicAdd(nextern, -1) - 1;
		everts[loc] = v0;
		S_epids_lt[eid] = -(S_epids_lt[eid] + 1);
	}

	template<typename INDEX_TYPE>
	__global__ void markBoundary(const INDEX_TYPE* __restrict mapping, const INDEX_TYPE* __restrict M_row_ids, const INDEX_TYPE* __restrict face_offs,
		const INDEX_TYPE* __restrict S_col_ptr, const INDEX_TYPE* __restrict S_row_ids, INDEX_TYPE* S_epids_lt, INDEX_TYPE* everts, INDEX_TYPE* nextern, unsigned nnz)
	{
		unsigned int threadId = blockIdx.x * blockDim.x + threadIdx.x;
		if (threadId >= nnz)
			return;

		auto v0 = M_row_ids[threadId];

		auto my_face = mapping[threadId];
		auto face_order = face_offs[my_face + 1] - face_offs[my_face];
		auto vout = M_row_ids[face_offs[my_face] + (threadId + 1 - face_offs[my_face]) % face_order];

		INDEX_TYPE eid = 0;
		for (auto i = S_col_ptr[v0]; i < S_col_ptr[v0 + 1]; ++i)
		{
			if (S_row_ids[i] == vout)
			{
				eid = i;
				break;
			}
		}

		bool is_extern = true;
		for (auto i = S_col_ptr[vout]; i < S_col_ptr[vout + 1]; ++i)
		{
			if (S_row_ids[i] == v0)
			{
				is_extern = false;
				break;
			}
		}

		if (!is_extern)
			return;

		auto loc = atomicAdd(nextern, -1) - 1;
		everts[loc] = v0;
		S_epids_lt[eid] = -(S_epids_lt[eid] + 1);
	}

	template<typename INDEX_TYPE>
	__global__ void refineTopologyQuad(INDEX_TYPE const* const __restrict M_row_ids,
		INDEX_TYPE const* const __restrict S_col_ptr, INDEX_TYPE const* const __restrict S_row_ids, INDEX_TYPE const* const __restrict S_epids,
		INDEX_TYPE* const M_new_row_ids, int nnz, int nv, int nf)
	{
		int threadId = blockIdx.x * blockDim.x + threadIdx.x;
		if (threadId >= nnz)
			return;

		INDEX_TYPE v0 = M_row_ids[threadId];
		INDEX_TYPE vout = M_row_ids[(threadId / 4 * 4) + ((threadId + 1) % 4)];
		INDEX_TYPE epid = 0;
		for (INDEX_TYPE i = S_col_ptr[min(v0, vout)]; i < S_col_ptr[min(v0, vout) + 1]; ++i)
		{
			if (S_row_ids[i] == max(v0, vout))
			{
				epid = i;
				break;
			}
		}
		epid = S_epids[epid] >= 0 ? S_epids[epid] + nv + nf : -S_epids[epid] + nv + nf - 1;
		reinterpret_cast<int4*>(M_new_row_ids)[threadId] = { v0, epid, threadId / 4 + nv,  __shfl_sync(0xffffffff, epid, (threadId + 3), 4) };
	}

	template<typename INDEX_TYPE>
	__global__ void refineTopologyQuadExtern(INDEX_TYPE const* const __restrict M_row_ids,
		INDEX_TYPE const* const __restrict S_col_ptr, INDEX_TYPE const* const __restrict S_row_ids, INDEX_TYPE const* const __restrict S_epids,
		INDEX_TYPE* const M_new_row_ids, int nnz, int nv, int nf)
	{
		int threadId = blockIdx.x * blockDim.x + threadIdx.x;
		if (threadId >= nnz)
			return;

		INDEX_TYPE v0 = M_row_ids[threadId];
		INDEX_TYPE vout = M_row_ids[(threadId / 4 * 4) + ((threadId + 1) % 4)];
		INDEX_TYPE epid = 0;
		bool found = false;
		for (INDEX_TYPE i = S_col_ptr[min(v0, vout)]; i < S_col_ptr[min(v0, vout) + 1]; ++i)
		{
			if (S_row_ids[i] == max(v0, vout))
			{
				epid = i;
				found = true;
				break;
			}
		}
		if (!found)
		{
			for (INDEX_TYPE i = S_col_ptr[max(v0, vout)]; i < S_col_ptr[max(v0, vout) + 1]; ++i)
			{
				if (S_row_ids[i] == min(v0, vout))
				{
					epid = i;
					found = true;
					break;
				}
			}
		}
		epid = S_epids[epid] >= 0 ? S_epids[epid] + nv + nf : -S_epids[epid] + nv + nf - 1;
		reinterpret_cast<int4*>(M_new_row_ids)[threadId] = { v0, epid, threadId / 4 + nv,  __shfl_sync(0xffffffff, epid, (threadId + 3), 4) };
	}

	template<typename INDEX_TYPE>
	__global__ void refineTopology(const INDEX_TYPE* __restrict mapping, const INDEX_TYPE* __restrict M_row_ids, INDEX_TYPE* face_offs,
		const INDEX_TYPE* __restrict S_col_ptr, const INDEX_TYPE* __restrict S_row_ids, const INDEX_TYPE* __restrict S_epids,
		INDEX_TYPE* M_new_row_ids, int nf, int nv, int nnz)
	{
		unsigned int threadId = blockIdx.x * blockDim.x + threadIdx.x;
		if (threadId >= nnz)
			return;

		auto v0 = M_row_ids[threadId];

		auto my_face = mapping[threadId];
		auto face_order = face_offs[my_face + 1] - face_offs[my_face];
		auto vout = M_row_ids[face_offs[my_face] + (threadId + 1 - face_offs[my_face]) % face_order];
		auto vin = M_row_ids[face_offs[my_face] + (threadId + face_order - 1 - face_offs[my_face]) % face_order];

		INDEX_TYPE epoutid = 0;
		for (auto j = S_col_ptr[min(v0, vout)]; j < S_col_ptr[min(v0, vout) + 1]; ++j)
		{
			if (S_row_ids[j] == max(v0, vout))
			{
				epoutid = j;
				break;
			}
		}
		epoutid = S_epids[epoutid] + nv + nf;

		INDEX_TYPE epinid = 0;
		for (auto j = S_col_ptr[min(v0, vin)]; j < S_col_ptr[min(v0, vin) + 1]; ++j)
		{
			if (S_row_ids[j] == max(v0, vin))
			{
				epinid = j;
				break;
			}
		}
		epinid = S_epids[epinid] + nv + nf;

		reinterpret_cast<int4*>(M_new_row_ids)[threadId] = { v0, epoutid, my_face + nv, epinid };
	}

	template<typename INDEX_TYPE>
	__global__ void refineTopologyExtern(const INDEX_TYPE* __restrict mapping, const INDEX_TYPE* __restrict M_row_ids, INDEX_TYPE* face_offs,
		const INDEX_TYPE* __restrict S_col_ptr, const INDEX_TYPE* __restrict S_row_ids, const INDEX_TYPE* __restrict S_epids,
		INDEX_TYPE* M_new_row_ids, int nf, int nv, int nnz)
	{
		unsigned int threadId = blockIdx.x * blockDim.x + threadIdx.x;
		if (threadId >= nnz)
			return;

		auto v0 = M_row_ids[threadId];

		auto my_face = mapping[threadId];
		auto face_order = face_offs[my_face + 1] - face_offs[my_face];
		auto vout = M_row_ids[face_offs[my_face] + (threadId + 1 - face_offs[my_face]) % face_order];
		auto vin = M_row_ids[face_offs[my_face] + (threadId + face_order - 1 - face_offs[my_face]) % face_order];

		INDEX_TYPE epoutid = 0;
		bool found = false;
		for (auto i = S_col_ptr[min(v0, vout)]; i < S_col_ptr[min(v0, vout) + 1]; ++i)
		{
			if (S_row_ids[i] == max(v0, vout))
			{
				epoutid = i;
				found = true;
				break;
			}
		}
		if (!found)
		{
			for (auto i = S_col_ptr[max(v0, vout)]; i < S_col_ptr[max(v0, vout) + 1]; ++i)
			{
				if (S_row_ids[i] == min(v0, vout))
				{
					epoutid = i;
					found = true;
					break;
				}
			}
		}
		epoutid = S_epids[epoutid] >= 0 ? S_epids[epoutid] + nv + nf : -S_epids[epoutid] + nv + nf - 1;

		INDEX_TYPE epinid = 0;
		found = false;
		for (auto i = S_col_ptr[min(v0, vin)]; i < S_col_ptr[min(v0, vin) + 1]; ++i)
		{
			if (S_row_ids[i] == max(v0, vin))
			{
				epinid = i;
				found = true;
				break;
			}
		}
		if (!found)
		{
			for (auto i = S_col_ptr[max(v0, vin)]; i < S_col_ptr[max(v0, vin) + 1]; ++i)
			{
				if (S_row_ids[i] == min(v0, vin))
				{
					epinid = i;
					found = true;
					break;
				}
			}
		}
		epinid = S_epids[epinid] >= 0 ? S_epids[epinid] + nv + nf : -S_epids[epinid] + nv + nf - 1;

		reinterpret_cast<int4*>(M_new_row_ids)[threadId] = { v0, epoutid, my_face + nv, epinid };
	}

	template<typename INDEX_TYPE, typename VERTEX_TYPE>
	__global__ void calculateFaceEdgepointsQuad(const INDEX_TYPE* M_row_ids, const INDEX_TYPE* M_new_row_ids,
		VERTEX_TYPE* pos_in, VERTEX_TYPE* fps, VERTEX_TYPE* eps,
		unsigned nnz, unsigned nverts, unsigned ncomponents)
	{
		unsigned threadId = blockIdx.x * blockDim.x + threadIdx.x;
		if (threadId >= nnz * 4)
			return;

		unsigned int vid = M_row_ids[threadId / 4];
		const float v_comp = pos_in[4 * vid + threadId % 4];
		float fp_comp = v_comp;
		fp_comp += __shfl_down_sync(0xffffffff, fp_comp, 8, 16);
		fp_comp += __shfl_down_sync(0xffffffff, fp_comp, 4, 16);
		fp_comp = __shfl_up_sync(0xffffffff, fp_comp, 4, 16);
		fp_comp = __shfl_up_sync(0xffffffff, fp_comp, 8, 16);

		fp_comp *= (1.0f / 4.0f);
		fps[threadId / 16 * 4 + threadId % 4] = fp_comp;

		const unsigned fids_new = M_new_row_ids[threadId];
		const unsigned ep_off = (__shfl_sync(0xffffffff, fids_new, 1, 4) - nverts - nnz / 4) * 4 + threadId % 4;
		atomicAdd(&eps[ep_off], (fp_comp + v_comp) / 4.0f);
	}

	template<typename INDEX_TYPE, typename VERTEX_TYPE>
	__global__ void calculateFacepoints(const INDEX_TYPE* __restrict M_col_ptr, const INDEX_TYPE* __restrict M_row_ids, VERTEX_TYPE* pos_in, VERTEX_TYPE* pos_out, unsigned nfaces, unsigned nverts, unsigned ncomponents)
	{
		unsigned threadId = blockIdx.x * blockDim.x + threadIdx.x;
		if (threadId >= nfaces)
			return;

		VERTEX_TYPE face_order_rcp = static_cast<VERTEX_TYPE>(1) / static_cast<VERTEX_TYPE>(M_col_ptr[threadId + 1] - M_col_ptr[threadId]);
		math::float4 val = math::float4(0.0f);
		float4* pos_in_load = reinterpret_cast<float4*>(pos_in);
		for (INDEX_TYPE i = M_col_ptr[threadId]; i < M_col_ptr[threadId + 1]; ++i)
		{
			INDEX_TYPE row = M_row_ids[i];
			val += to_math_f4(pos_in_load[row]);
		}
		reinterpret_cast<float4*>(pos_out)[threadId] = to_f4(val * face_order_rcp);
	}

	template<typename INDEX_TYPE, typename VERTEX_TYPE>
	__global__ void calculateEdgepoints(const INDEX_TYPE* __restrict M_row_ids,
		const INDEX_TYPE* __restrict S_col_ptr, const INDEX_TYPE* __restrict S_row_ids, const INDEX_TYPE* __restrict S_epids,
		const VERTEX_TYPE* __restrict pos, const VERTEX_TYPE* __restrict facepoints, VERTEX_TYPE* edgepoints,
		unsigned nnz, unsigned nverts, unsigned nfaces, unsigned ncomponents)
	{
		unsigned threadId = blockIdx.x * blockDim.x + threadIdx.x;
		if (threadId >= nnz)
			return;

		uint4 f = reinterpret_cast<const uint4*>(M_row_ids)[threadId];
		atomicAddVec(&edgepoints[4 * (f.y - nverts - nfaces)], (to_math_f4(reinterpret_cast<const float4*>(pos)[f.x]) + to_math_f4(reinterpret_cast<const float4*>(facepoints)[f.z - nverts])) * 1.0f / 4.0f);
	}

	template<typename VERTEX_TYPE, typename INDEX_TYPE>
	__global__ void vertexSum(INDEX_TYPE const* __restrict d_col_offsets, INDEX_TYPE const* __restrict d_row_ids, VERTEX_TYPE const* __restrict x, INDEX_TYPE const* valences,
		VERTEX_TYPE* y, unsigned int cols_size_y, unsigned ncomponents)
	{
		unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
		if (tid < cols_size_y)
		{
			math::float4 val = math::float4(0.0f);
			for (INDEX_TYPE i = d_col_offsets[tid]; i < d_col_offsets[tid + 1]; ++i)
			{
				INDEX_TYPE rid = d_row_ids[i];
				val += to_math_f4(reinterpret_cast<const float4*>(x)[rid]);
			}

			math::float4 original = to_math_f4(reinterpret_cast<const float4*>(x)[tid])* static_cast<VERTEX_TYPE>(valences[tid] - 2) / static_cast<VERTEX_TYPE>(valences[tid]);
			reinterpret_cast<float4*>(y)[tid] = to_f4(val / static_cast<VERTEX_TYPE>(valences[tid] * valences[tid]) + original);

		}
	}

	template<typename VERTEX_TYPE, typename INDEX_TYPE>
	__global__ void facepointSumQuad(INDEX_TYPE const* d_row_ids, VERTEX_TYPE const* x, INDEX_TYPE const* __restrict valences, VERTEX_TYPE* y,
		unsigned cols_size_y, unsigned cols_size_x, unsigned ncomponents)
	{
		unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
		if (tid < ncomponents * cols_size_x)
		{
			VERTEX_TYPE xval = x[tid];
			uint4 rid = reinterpret_cast<const uint4*>(d_row_ids)[tid / 4];
			unsigned int component = threadIdx.x % 4;
			atomicAdd(y + 4 * rid.x + component, xval / static_cast<VERTEX_TYPE>(valences[rid.x] * valences[rid.x]));
			atomicAdd(y + 4 * rid.y + component, xval / static_cast<VERTEX_TYPE>(valences[rid.y] * valences[rid.y]));
			atomicAdd(y + 4 * rid.z + component, xval / static_cast<VERTEX_TYPE>(valences[rid.z] * valences[rid.z]));
			atomicAdd(y + 4 * rid.w + component, xval / static_cast<VERTEX_TYPE>(valences[rid.w] * valences[rid.w]));
		}
	}

	template<typename VERTEX_TYPE, typename INDEX_TYPE>
	__global__ void facepointSum(INDEX_TYPE const* __restrict d_col_offsets, INDEX_TYPE const* d_row_ids, VERTEX_TYPE const* x, INDEX_TYPE const* __restrict valences, VERTEX_TYPE* y,
		unsigned int cols_size_y, unsigned int cols_size_x, unsigned int ncomponents)
	{
		unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
		if (tid < cols_size_x)
		{
			float4 xval_load = reinterpret_cast<const float4*>(x)[tid];
			math::float4 xval = to_math_f4(xval_load);
			for (unsigned int i = d_col_offsets[tid]; i < d_col_offsets[tid + 1]; ++i)
			{
				unsigned int rid = d_row_ids[i];
				atomicAddVec(y + 4 * rid, xval / static_cast<VERTEX_TYPE>(valences[rid] * valences[rid]));
			}
		}
	}

	template<typename INDEX_TYPE, typename VERTEX_TYPE>
	__global__ void correctBoundaryS1(const INDEX_TYPE* __restrict M_row_ids,
		const INDEX_TYPE* __restrict S_col_ptr, const INDEX_TYPE* __restrict S_row_ids, INDEX_TYPE* S_epids_lt, INDEX_TYPE* everts,
		VERTEX_TYPE* coarseVertexData, VERTEX_TYPE* refinedVertexData, unsigned nextern, unsigned nverts, unsigned nfaces)
	{
		unsigned int threadId = blockIdx.x * blockDim.x + threadIdx.x;
		if (threadId >= nextern)
			return;

		float4* refined_store = reinterpret_cast<float4*>(refinedVertexData);
		float4* coarse_load = reinterpret_cast<float4*>(coarseVertexData);
		auto my_evert = everts[threadId];

		auto e0_loc = 0;
		for (auto i = S_col_ptr[my_evert]; i < S_col_ptr[my_evert + 1]; ++i)
		{
			if (S_epids_lt[i] < 0)
			{
				e0_loc = i;
				break;
			}
		}

		auto ep0_id = -S_epids_lt[e0_loc] + nverts + nfaces - 1;
		auto v0_id = S_row_ids[e0_loc];
		refined_store[ep0_id] = to_f4((to_math_f4(coarse_load[my_evert]) + to_math_f4(coarse_load[v0_id])) / 2.0f);
		refined_store[my_evert] = to_f4(to_math_f4(coarse_load[my_evert]) * (3.0f / 4.0f) + to_math_f4(coarse_load[v0_id]) * (1.0f / 8.0f));
	}

	template<typename INDEX_TYPE, typename VERTEX_TYPE>
	__global__ void correctBoundaryS2(const INDEX_TYPE* __restrict M_row_ids,
		const INDEX_TYPE* __restrict S_col_ptr, const INDEX_TYPE* __restrict S_row_ids, INDEX_TYPE* S_epids_lt, INDEX_TYPE* everts,
		VERTEX_TYPE* coarseVertexData, VERTEX_TYPE* refinedVertexData, unsigned nextern, unsigned nverts, unsigned nfaces)
	{
		unsigned int threadId = blockIdx.x * blockDim.x + threadIdx.x;
		if (threadId >= nextern)
			return;

		float4* refined_store = reinterpret_cast<float4*>(refinedVertexData);
		float4* coarse_load = reinterpret_cast<float4*>(coarseVertexData);
		auto my_evert = everts[threadId];

		auto e0_loc = 0;
		for (auto i = S_col_ptr[my_evert]; i < S_col_ptr[my_evert + 1]; ++i)
		{
			if (S_epids_lt[i] < 0)
			{
				e0_loc = i;
				break;
			}
		}
		refined_store[S_row_ids[e0_loc]] = to_f4(to_math_f4(refined_store[S_row_ids[e0_loc]]) + to_math_f4(coarse_load[my_evert]) * (1.0f / 8.0f));
	}
}
