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
#include "cub_wrappers.cuh"

namespace LAKHelperKernels
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

	template<typename LHS_TYPE, typename RHS_TYPE, int STRIDE>
	__global__ void addElementWise_d(const LHS_TYPE* lhs, const RHS_TYPE* rhs, LHS_TYPE* res, unsigned length)
	{
		int tid = blockIdx.x * blockDim.x + threadIdx.x;
		if (tid >= length)
			return;

		res[tid] = lhs[tid] + rhs[tid / STRIDE];
	}

	template<typename LHS_TYPE, typename RHS_TYPE, int STRIDE = 1>
	void addElementWise(const LHS_TYPE* lhs, const RHS_TYPE* rhs, LHS_TYPE* res, const size_t nelements, int block_dim = -1)
	{
		block_dim = block_dim > 0 ? block_dim : 256;
		size_t grid_dim = (nelements + block_dim - 1) / block_dim;
		addElementWise_d<LHS_TYPE, RHS_TYPE, STRIDE> << <grid_dim, block_dim >> > (lhs, rhs, res, nelements);
	}

	template<typename LHS_TYPE, typename RHS_TYPE, int STRIDE>
	__global__ void multElementWise_d(const LHS_TYPE* lhs, const RHS_TYPE* rhs, LHS_TYPE add, LHS_TYPE* res, unsigned length)
	{
		unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
		if (tid >= length)
			return;

		res[tid] = lhs[tid] * (rhs[tid / STRIDE] + add);
	}

	template<typename LHS_TYPE, typename RHS_TYPE, int STRIDE = 4>
	void multElementWise(const LHS_TYPE* lhs, const RHS_TYPE* rhs, const LHS_TYPE add, LHS_TYPE* res, const size_t nelements, int block_dim = -1)
	{
		block_dim = block_dim > 0 ? block_dim : 256;
		size_t grid_dim = (nelements + block_dim - 1) / block_dim;
		multElementWise_d<LHS_TYPE, RHS_TYPE, STRIDE> << <grid_dim, block_dim >> > (lhs, rhs, add, res, nelements);
	}

	template<typename DATA_TYPE>
	__global__ void divElementWise_d(const DATA_TYPE* in, const DATA_TYPE* div, DATA_TYPE add, DATA_TYPE* out, unsigned length)
	{
		unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
		if (tid >= length)
			return;
		float4 in_f4 = reinterpret_cast<const float4*>(in)[tid];
		float val = div[tid];
		reinterpret_cast<float4*>(out)[tid] = { in_f4.x / (val + add), in_f4.y / (val + add), in_f4.z / (val + add), in_f4.w / (val + add) };
	}

	template<typename NUM_TYPE, typename DENOM_TYPE, int STRIDE >
	__global__ void divElementWise_d(const NUM_TYPE* num, const DENOM_TYPE* denom, NUM_TYPE add, NUM_TYPE* out, unsigned length)
	{
		unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
		if (tid >= length)
			return;

		out[tid] = num[tid] / denom[tid / STRIDE] + add;
	}

	template<typename NUM_TYPE, typename DENOM_TYPE, int STRIDE = 4>
	void divElementWise(const NUM_TYPE* num, const DENOM_TYPE* denom, const NUM_TYPE add, NUM_TYPE* out, const size_t nelements, int block_dim = -1)
	{
		block_dim = block_dim > 0 ? block_dim : 256;
		size_t grid_dim = (nelements + block_dim - 1) / block_dim;
		divElementWise_d<NUM_TYPE, DENOM_TYPE, STRIDE> << <grid_dim, block_dim >> > (num, denom, add, out, nelements);
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

	template<typename OFFSET_TYPE, typename VALUE_TYPE>
	__global__ void countNnzPerCol(const OFFSET_TYPE* col_ptr, const VALUE_TYPE* vals, const size_t ncols, OFFSET_TYPE* nnzpercol)
	{
		unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
		if (tid >= ncols)
			return;

		OFFSET_TYPE nnz = 0;
		for (OFFSET_TYPE i = col_ptr[tid]; i < col_ptr[tid + 1]; ++i)
			nnz += static_cast<OFFSET_TYPE>(vals[i] != 0);

		nnzpercol[tid] = nnz;
	}


	template<typename OFFSET_TYPE, typename INDEX_TYPE, typename VALUE_TYPE>
	__global__ void compressRowsAndVals(const OFFSET_TYPE* col_ptr, const OFFSET_TYPE* col_ptr_comp, const INDEX_TYPE* row_ids, INDEX_TYPE* row_ids_comp, const VALUE_TYPE* vals, VALUE_TYPE* vals_comp, size_t ncols)
	{
		unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
		if (tid >= ncols)
			return;

		OFFSET_TYPE col_start = col_ptr_comp[tid];
		for (OFFSET_TYPE i = col_ptr[tid]; i < col_ptr[tid + 1]; ++i)
		{
			if (vals[i] != 0)
			{
				vals_comp[col_start] = vals[i];
				row_ids_comp[col_start] = row_ids[i];
				col_start++;
			}
		}
	}


	template<typename OFFSET_TYPE, typename INDEX_TYPE, typename VALUE_TYPE, typename MEM_MANAGER_TYPE>
	void compress(OFFSET_TYPE*& ptr, INDEX_TYPE*& ids, VALUE_TYPE*& vals, size_t rows, size_t cols, size_t& nnz, MEM_MANAGER_TYPE& mem)
	{
		constexpr int block_dim{ 256 };

		OFFSET_TYPE* d_nnz_per_col = reinterpret_cast<OFFSET_TYPE*>(mem.getMemory(cols * sizeof(OFFSET_TYPE)));

		size_t grid_dim = (cols + block_dim - 1) / block_dim;
		countNnzPerCol << <grid_dim, block_dim >> > (ptr, vals, cols, d_nnz_per_col);

		OFFSET_TYPE* d_new_col_ptr = reinterpret_cast<OFFSET_TYPE*>(mem.getMemory((cols + 1) * sizeof(OFFSET_TYPE)));
		CubFunctions::scan_exclusive(d_nnz_per_col, d_new_col_ptr, cols + 1, mem);

		mem.freeMemory(d_nnz_per_col);

		OFFSET_TYPE new_nnz;
		succeed(cudaMemcpy(&new_nnz, d_new_col_ptr + cols, sizeof(OFFSET_TYPE), cudaMemcpyDeviceToHost));
		INDEX_TYPE* d_new_row_ids = reinterpret_cast<INDEX_TYPE*>(mem.getMemory(new_nnz * sizeof(INDEX_TYPE)));
		VALUE_TYPE* d_new_vals = reinterpret_cast<VALUE_TYPE*>(mem.getMemory(new_nnz * sizeof(VALUE_TYPE)));
		grid_dim = (cols + block_dim - 1) / block_dim;
		compressRowsAndVals << <grid_dim, block_dim >> > (ptr, d_new_col_ptr, ids, d_new_row_ids, vals, d_new_vals, cols);

		if (!mem.freeMemory(ptr))
			succeed(cudaFree(ptr));

		if (!mem.freeMemory(ids))
			succeed(cudaFree(ids));

		if (!mem.freeMemory(vals))
			succeed(cudaFree(vals));

		nnz = new_nnz;
		ptr = d_new_col_ptr;
		ids = d_new_row_ids;
		vals = d_new_vals;
	}
}

namespace LAKCCKernels
{
	template<typename OFFSET_TYPE, typename INDEX_TYPE, typename VALUE_TYPE>
	__global__ void extractEdgeInfoFromE(const OFFSET_TYPE* E_col_ptr, const INDEX_TYPE* E_row_ids, const VALUE_TYPE* E_vals, const size_t E_cols,
		INDEX_TYPE* internal0, INDEX_TYPE* internal1, INDEX_TYPE* intids, INDEX_TYPE* external0, INDEX_TYPE* external1, INDEX_TYPE* extids, INDEX_TYPE* nintext)
	{
		int tid = blockIdx.x * blockDim.x + threadIdx.x;
		if (tid >= E_cols)
			return;

		for (OFFSET_TYPE j = E_col_ptr[tid]; j < E_col_ptr[tid + 1]; ++j)
		{
			INDEX_TYPE i = E_row_ids[j];
			unsigned is_external = static_cast<unsigned>(E_vals[j] > 0);
			unsigned off = atomicAdd(nintext + is_external, 1);
			if (is_external)
			{
				external0[off] = i;
				external1[off] = tid;
				extids[off] = E_vals[j] - 1;
			}
			else
			{
				internal0[off] = i;
				internal1[off] = tid;
				intids[off] = -E_vals[j] - 1;
			}
		}
	}

	template<typename OFFSET_TYPE, typename INDEX_TYPE, typename VALUE_TYPE>
	__global__ void extractFaceInfoFromFFt(const OFFSET_TYPE* F_col_ptr, const INDEX_TYPE* F_row_ids, const VALUE_TYPE* F_vals,
		const OFFSET_TYPE* Ft_col_ptr, const INDEX_TYPE* Ft_row_ids, const VALUE_TYPE* Ft_vals,
		INDEX_TYPE* internal0, INDEX_TYPE* internal1, unsigned nintids,
		INDEX_TYPE* F0_ids, INDEX_TYPE* F1_ids)
	{
		int tid = blockIdx.x * blockDim.x + threadIdx.x;
		if (tid >= nintids)
			return;
		INDEX_TYPE col = internal0[tid];
		INDEX_TYPE row = internal1[tid];
		INDEX_TYPE fid = 0;
		for (OFFSET_TYPE j = F_col_ptr[col]; j < F_col_ptr[col + 1]; ++j)
		{
			if (F_row_ids[j] == row)
			{
				fid = j;
				break;
			}
		}
		F0_ids[tid] = F_vals[fid] - 1;

		fid = 0;
		for (OFFSET_TYPE j = Ft_col_ptr[col]; j < Ft_col_ptr[col + 1]; ++j)
		{
			if (Ft_row_ids[j] == row)
			{
				fid = j;
				break;
			}
		}
		F1_ids[tid] = Ft_vals[fid] - 1;
	}


	template<typename OFFSET_TYPE, typename INDEX_TYPE, typename VALUE_TYPE>
	__global__ void refineTopology(const OFFSET_TYPE* __restrict M_col_ptr, const INDEX_TYPE* __restrict M_row_ids,
		const OFFSET_TYPE* __restrict S_col_ptr, const INDEX_TYPE* __restrict S_row_ids, const VALUE_TYPE* __restrict S_epids,
		INDEX_TYPE* M_new_row_ids, int nf, int nv)
	{
		int threadId = blockIdx.x * blockDim.x + threadIdx.x;
		if (threadId >= nf)
			return;

		auto write_off = M_col_ptr[threadId];
		auto face_cnt = 0;

		for (auto i = M_col_ptr[threadId]; i < M_col_ptr[threadId + 1]; ++i)
		{
			auto v0 = M_row_ids[i];

			auto vout = 0;
			if (i == M_col_ptr[threadId + 1] - 1)
				vout = M_row_ids[M_col_ptr[threadId]];
			else
				vout = M_row_ids[i + 1];

			auto vin = 0;
			if (i == M_col_ptr[threadId])
				vin = M_row_ids[M_col_ptr[threadId + 1] - 1];
			else
				vin = M_row_ids[i - 1];

			INDEX_TYPE epoutid = 0;
			for (auto j = S_col_ptr[max(v0, vout)]; j < S_col_ptr[max(v0, vout) + 1]; ++j)
			{
				if (S_row_ids[j] == min(v0, vout))
				{
					epoutid = j;
					break;
				}
			}
			epoutid = S_epids[epoutid] > 0 ? S_epids[epoutid] - 1 + nv + nf : -S_epids[epoutid] - 1 + nv + nf;

			INDEX_TYPE epinid = 0;
			for (auto j = S_col_ptr[max(v0, vin)]; j < S_col_ptr[max(v0, vin) + 1]; ++j)
			{
				if (S_row_ids[j] == min(v0, vin))
				{
					epinid = j;
					break;
				}
			}
			epinid = S_epids[epinid] > 0 ? S_epids[epinid] - 1 + nv + nf : -S_epids[epinid] - 1 + nv + nf;

			reinterpret_cast<int4*>(M_new_row_ids)[write_off + face_cnt] = { v0, epoutid, threadId + nv, epinid };
			face_cnt++;
		}

	}

	template<typename OFFSET_TYPE, typename INDEX_TYPE, typename VALUE_TYPE>
	__global__ void refineTopologyHomogeneous(const INDEX_TYPE* __restrict M_row_ids,
		const OFFSET_TYPE* __restrict S_col_ptr, const INDEX_TYPE* __restrict S_row_ids, const VALUE_TYPE* __restrict S_epids,
		INDEX_TYPE* M_new_row_ids, const size_t nnz, const int nv, const int nf, const int nvf)
	{
		int threadId = blockIdx.x * blockDim.x + threadIdx.x;
		if (threadId >= nnz)
			return;

		auto v0 = M_row_ids[threadId];
		auto vout = M_row_ids[(threadId / nvf * nvf) + ((threadId + 1) % nvf)];
		INDEX_TYPE epid1 = 0;
		for (auto i = S_col_ptr[max(v0, vout)]; i < S_col_ptr[max(v0, vout) + 1]; ++i)
		{
			if (S_row_ids[i] == min(v0, vout))
			{
				epid1 = i;
				break;
			}
		}

		auto vin = M_row_ids[(threadId / nvf * nvf) + ((threadId + (nvf - 1)) % nvf)];
		INDEX_TYPE epid2 = 0;
		for (auto i = S_col_ptr[max(v0, vin)]; i < S_col_ptr[max(v0, vin) + 1]; ++i)
		{
			if (S_row_ids[i] == min(v0, vin))
			{
				epid2 = i;
				break;
			}
		}

		epid1 = S_epids[epid1] < 0 ? -S_epids[epid1] + nv + nf - 1 : S_epids[epid1] + nv + nf - 1;
		epid2 = S_epids[epid2] < 0 ? -S_epids[epid2] + nv + nf - 1 : S_epids[epid2] + nv + nf - 1;
		reinterpret_cast<int4*>(M_new_row_ids)[threadId] = { v0, epid1, threadId / nvf + nv,  epid2 };
	}

	template<typename OFFSET_TYPE>
	__global__ void createQuadColPtr(OFFSET_TYPE* col_ptr, size_t nquads)
	{
		int tid = blockIdx.x * blockDim.x + threadIdx.x;
		if (tid > nquads)
			return;

		col_ptr[tid] = tid * 4;
	}

	template<typename VALUE_TYPE>
	__global__ void createQuadVals(VALUE_TYPE* vals, size_t nquads)
	{
		int tid = blockIdx.x * blockDim.x + threadIdx.x;
		if (tid >= 4 * nquads)
			return;

		vals[tid] = (tid % 4) + 1;
	}


	__device__ __forceinline__ math::float4 to_math_f4(float4 v) { return math::float4(v.x, v.y, v.z, v.w); }
	__device__ __forceinline__ float4 to_f4(math::float4 v) { return{ v.x, v.y, v.z, v.w }; }

	template<typename INDEX_TYPE, typename DATA_TYPE>
	__global__ void calculateInternalEdgepoints(const INDEX_TYPE* ind0, const INDEX_TYPE* ind1, const INDEX_TYPE* indint, const INDEX_TYPE* f0, const INDEX_TYPE* f1,
		const DATA_TYPE* pos, const DATA_TYPE* fp, DATA_TYPE* ep, unsigned nedges)
	{
		unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
		if (tid >= nedges)
			return;

		const float4* pos_load = reinterpret_cast<const float4*>(pos);
		const float4* fp_load = reinterpret_cast<const float4*>(fp);
		reinterpret_cast<float4*>(ep)[indint[tid]] = to_f4(
			(to_math_f4(pos_load[ind0[tid]]) + to_math_f4(pos_load[ind1[tid]])
				+ to_math_f4(fp_load[f0[tid]]) + to_math_f4(fp_load[f1[tid]])) / 4.0);
	}

	template<typename INDEX_TYPE, typename DATA_TYPE>
	__global__ void calculateExternalEdgepoints(const INDEX_TYPE* ind0, const INDEX_TYPE* ind1, const INDEX_TYPE* indext,
		const DATA_TYPE* pos, DATA_TYPE* ep, unsigned nedges)
	{
		unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
		if (tid >= nedges)
			return;

		const float4* pos_load = reinterpret_cast<const float4*>(pos);
		reinterpret_cast<float4*>(ep)[indext[tid]] = to_f4((to_math_f4(pos_load[ind0[tid]]) + to_math_f4(pos_load[ind1[tid]])) / 2.0);
	}

	template<typename INDEX_TYPE, typename DATA_TYPE>
	__global__ void prepareExternalVertexUpdate(const INDEX_TYPE* ind0, const INDEX_TYPE* ind1,
		const DATA_TYPE* pos_in, DATA_TYPE* pos_out, unsigned nedges)
	{
		unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
		if (tid >= nedges)
			return;

		INDEX_TYPE v0 = ind0[tid];
		INDEX_TYPE v1 = ind1[tid];
		const float4* pos_in_load = reinterpret_cast<const float4*>(pos_in);
		float4* pos_out_store = reinterpret_cast<float4*>(pos_out);

		math::float4 p0 = to_math_f4(pos_in_load[v0]);
		math::float4 p1 = to_math_f4(pos_in_load[v1]);
		pos_out_store[v0] = to_f4(p0 * (3.0 / 4.0));
		pos_out_store[v1] = to_f4(p1 * (3.0 / 4.0));
	}

	template<typename INDEX_TYPE, typename DATA_TYPE>
	__global__ void calculateExternalVertexUpdate(const INDEX_TYPE* ind0, const INDEX_TYPE* ind1,
		const DATA_TYPE* pos_in, DATA_TYPE* pos_out, unsigned nedges)
	{
		unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
		if (tid >= nedges)
			return;

		INDEX_TYPE v0 = ind0[tid];
		INDEX_TYPE v1 = ind1[tid];
		const float4* pos_in_load = reinterpret_cast<const float4*>(pos_in);
		float4* pos_out_store = reinterpret_cast<float4*>(pos_out);

		math::float4 p0 = to_math_f4(pos_in_load[v0]);
		math::float4 p1 = to_math_f4(pos_in_load[v1]);
		float* pos_out_addr0 = reinterpret_cast<float*>(pos_out_store + v1);
		float* pos_out_addr1 = reinterpret_cast<float*>(pos_out_store + v0);
		p0 *= 1.0 / 8.0;
		atomicAdd(pos_out_addr0 + 0, p0.x);
		atomicAdd(pos_out_addr0 + 1, p0.y);
		atomicAdd(pos_out_addr0 + 2, p0.z);
		atomicAdd(pos_out_addr0 + 3, p0.w);
		p1 *= 1.0 / 8.0;
		atomicAdd(pos_out_addr1 + 0, p1.x);
		atomicAdd(pos_out_addr1 + 1, p1.y);
		atomicAdd(pos_out_addr1 + 2, p1.z);
		atomicAdd(pos_out_addr1 + 3, p1.w);

	}
}
