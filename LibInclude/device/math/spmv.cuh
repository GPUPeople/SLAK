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

#include "math/bhblas.h"
#include "math/vector.h"

#include "cuda_host_helpers.h"
#include <cuda_runtime_api.h>


__device__ __forceinline__ int map_coordinate(float val)
{
	return __float2int_rn(val);
}

__device__ __forceinline__ int map_coordinate(int val)
{
	return val;
}

template<typename OFFSET_TYPE, typename INDEX_TYPE, typename VALUE_TYPE>
__global__ void k_spmv_left_csc_mapped(const OFFSET_TYPE* d_col_offsets, const INDEX_TYPE* d_row_ids, const VALUE_TYPE* d_values, const VALUE_TYPE* x,
	VALUE_TYPE* y, int cols, cudaTextureObject_t texObj, int map_size)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < cols)
	{
		VALUE_TYPE val = 0;
		for (OFFSET_TYPE i = d_col_offsets[tid]; i < d_col_offsets[tid + 1]; ++i)
		{
			INDEX_TYPE rid = d_row_ids[i];
			int map_coord = map_coordinate(d_values[i]);
			VALUE_TYPE map_val = tex1Dfetch<VALUE_TYPE>(texObj, min(map_coord, map_size));
			val += map_val * x[rid];
		}
		y[tid] = val;
	}
}

template<typename OFFSET_TYPE, typename INDEX_TYPE, typename VALUE_TYPE>
__global__ void k_spmv_left_csc_mapped_f4(const OFFSET_TYPE* __restrict d_col_offsets, const INDEX_TYPE* __restrict d_row_ids, const VALUE_TYPE* __restrict d_values, const float4* __restrict x,
	float4* y, int cols_size_y, cudaTextureObject_t texObj, unsigned int map_size)
{
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < cols_size_y)
	{
		float4 val = { 0.0f, 0.0f, 0.0f, 0.0f };
		for (OFFSET_TYPE i = d_col_offsets[tid]; i < d_col_offsets[tid + 1]; ++i)
		{
			INDEX_TYPE rid = d_row_ids[i];
			int map_coord = map_coordinate(d_values[i]);
			VALUE_TYPE map_val = tex1Dfetch<VALUE_TYPE>(texObj, min(map_coord, map_size));
			val = { val.x + map_val * x[rid].x, val.y + map_val * x[rid].y, val.z + map_val * x[rid].z, val.w + map_val * x[rid].w };
		}
		y[tid] = val;
	}
}

template<typename OFFSET_TYPE, typename INDEX_TYPE, typename VALUE_TYPE>
__global__ void k_spmv_right_csc_mapped(const OFFSET_TYPE* d_col_offsets, const INDEX_TYPE* d_row_ids, const VALUE_TYPE* d_values, const VALUE_TYPE* x,
	VALUE_TYPE* y, int cols, cudaTextureObject_t texObj, int map_size)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < cols)
	{
		VALUE_TYPE xval = x[tid];
		for (OFFSET_TYPE i = d_col_offsets[tid]; i < d_col_offsets[tid + 1]; ++i)
		{
			INDEX_TYPE rid = d_row_ids[i];
			int map_coord = map_coordinate(d_values[i]);
			VALUE_TYPE map_val = tex1Dfetch<VALUE_TYPE>(texObj, min(map_coord, map_size));
			atomicAdd(y + rid, map_val * xval);
		}
	}
}

template<typename OFFSET_TYPE, typename INDEX_TYPE, typename VALUE_TYPE>
__global__ void k_spmv_right_csc_mapped_f4(const OFFSET_TYPE* __restrict d_col_offsets, const INDEX_TYPE* __restrict d_row_ids, const VALUE_TYPE* __restrict d_values, const float4* __restrict x,
	float4* y, int cols, cudaTextureObject_t texObj, int map_size)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < cols)
	{
		float4 xval = x[tid];
		for (OFFSET_TYPE i = d_col_offsets[tid]; i < d_col_offsets[tid + 1]; ++i)
		{
			INDEX_TYPE rid = d_row_ids[i];
			int map_coord = map_coordinate(d_values[i]);
			VALUE_TYPE map_val = tex1Dfetch<VALUE_TYPE>(texObj, min(map_coord, map_size));
			float* y_addr = reinterpret_cast<float*>(y + rid);
			atomicAdd(y_addr + 0, map_val * xval.x);
			atomicAdd(y_addr + 1, map_val * xval.y);
			atomicAdd(y_addr + 2, map_val * xval.z);
			atomicAdd(y_addr + 3, map_val * xval.w);
		}
	}
}

template<typename T>
struct channelFormatPicker
{
	static constexpr cudaChannelFormatKind kind = cudaChannelFormatKindNone;
};

template<>
struct channelFormatPicker<int>
{
	static constexpr cudaChannelFormatKind kind = cudaChannelFormatKindSigned;
};

template<>
struct channelFormatPicker<unsigned>
{
	static constexpr cudaChannelFormatKind kind = cudaChannelFormatKindUnsigned;
};

template<>
struct channelFormatPicker<float>
{
	static constexpr cudaChannelFormatKind kind = cudaChannelFormatKindFloat;
};

template<typename OFFSET_TYPE, typename INDEX_TYPE, typename VALUE_TYPE>
void spmv_left_mapped(const OFFSET_TYPE* d_col_offsets, const INDEX_TYPE* d_row_ids, const VALUE_TYPE* d_values, const VALUE_TYPE* x, VALUE_TYPE* y, int rows, int cols, VALUE_TYPE* map, int map_size)
{
	cudaResourceDesc desc;
	desc.resType = cudaResourceTypeLinear;
	desc.res.linear = { reinterpret_cast<void*>(map), cudaCreateChannelDesc(32, 0, 0, 0, channelFormatPicker<VALUE_TYPE>::kind), map_size * sizeof(VALUE_TYPE) };
	cudaTextureDesc texDesc = { { cudaAddressModeClamp, cudaAddressModeClamp, cudaAddressModeClamp }, cudaFilterModePoint, cudaReadModeElementType, 0, 0, 0, cudaFilterModePoint, 0, 0, 0 };
	cudaTextureObject_t texobj;
	cudaCreateTextureObject(&texobj, &desc, &texDesc, nullptr);

	unsigned block_dim = 256;
	unsigned grid_dim = divup(static_cast<unsigned>(cols), block_dim);
	k_spmv_left_csc_mapped << < grid_dim, block_dim >> > (d_col_offsets, d_row_ids, d_values, x, y, cols, texobj, map_size);

	cudaDestroyTextureObject(texobj);
}

template<typename OFFSET_TYPE, typename INDEX_TYPE, typename VALUE_TYPE>
void spmv_left_mapped_f4(const OFFSET_TYPE* d_col_offsets, const INDEX_TYPE* d_row_ids, const VALUE_TYPE* d_values, const float4* x, float4* y, int rows, int cols, VALUE_TYPE* map, int map_size)
{
	cudaResourceDesc desc;
	desc.resType = cudaResourceTypeLinear;
	desc.res.linear = { reinterpret_cast<void*>(map), cudaCreateChannelDesc(32, 0, 0, 0, channelFormatPicker<VALUE_TYPE>::kind), map_size * sizeof(VALUE_TYPE) };
	cudaTextureDesc texDesc = { { cudaAddressModeClamp, cudaAddressModeClamp, cudaAddressModeClamp }, cudaFilterModePoint, cudaReadModeElementType, 0, 0, 0, cudaFilterModePoint, 0, 0, 0 };
	cudaTextureObject_t texobj;
	cudaCreateTextureObject(&texobj, &desc, &texDesc, nullptr);

	unsigned block_dim = 256;
	unsigned grid_dim = divup(static_cast<unsigned>(cols), block_dim);
	k_spmv_left_csc_mapped_f4 << < grid_dim, block_dim >> > (d_col_offsets, d_row_ids, d_values, x, y, cols, texobj, map_size);

	cudaDestroyTextureObject(texobj);
}

template<typename OFFSET_TYPE, typename INDEX_TYPE, typename VALUE_TYPE>
void spmv_right_mapped(const OFFSET_TYPE* d_col_offsets, const INDEX_TYPE* d_row_ids, const VALUE_TYPE* d_values, const VALUE_TYPE* x, VALUE_TYPE* y, int rows, int cols, VALUE_TYPE* map, int map_size)
{
	cudaResourceDesc desc;
	desc.resType = cudaResourceTypeLinear;
	desc.res.linear = { reinterpret_cast<void*>(map), cudaCreateChannelDesc(32, 0, 0, 0, channelFormatPicker<VALUE_TYPE>::kind), map_size * sizeof(VALUE_TYPE) };
	cudaTextureDesc texDesc = { { cudaAddressModeClamp, cudaAddressModeClamp, cudaAddressModeClamp }, cudaFilterModePoint, cudaReadModeElementType, 0, 0, 0, cudaFilterModePoint, 0, 0, 0 };
	cudaTextureObject_t texobj;
	cudaCreateTextureObject(&texobj, &desc, &texDesc, nullptr);

	unsigned block_dim = 256;
	unsigned grid_dim = divup(static_cast<unsigned>(cols), block_dim);
	k_spmv_right_csc_mapped << < grid_dim, block_dim >> > (d_col_offsets, d_row_ids, d_values, x, y, cols, texobj, map_size);

	cudaDestroyTextureObject(texobj);
}

template<typename OFFSET_TYPE, typename INDEX_TYPE, typename VALUE_TYPE>
void spmv_right_mapped_f4(const OFFSET_TYPE* d_col_offsets, const INDEX_TYPE* d_row_ids, const VALUE_TYPE* d_values, const float4* x, float4* y, int rows, int cols, VALUE_TYPE* map, int map_size)
{
	cudaResourceDesc desc;
	desc.resType = cudaResourceTypeLinear;
	desc.res.linear = { reinterpret_cast<void*>(map), cudaCreateChannelDesc(32, 0, 0, 0, channelFormatPicker<VALUE_TYPE>::kind), map_size * sizeof(VALUE_TYPE) };
	cudaTextureDesc texDesc = { { cudaAddressModeClamp, cudaAddressModeClamp, cudaAddressModeClamp }, cudaFilterModePoint, cudaReadModeElementType, 0, 0, 0, cudaFilterModePoint, 0, 0, 0 };
	cudaTextureObject_t texobj;
	cudaCreateTextureObject(&texobj, &desc, &texDesc, nullptr);

	unsigned block_dim = 256;
	unsigned grid_dim = divup(static_cast<unsigned>(cols), block_dim);
	k_spmv_right_csc_mapped_f4 << < grid_dim, block_dim >> > (d_col_offsets, d_row_ids, d_values, x, y, cols, texobj, map_size);

	cudaDestroyTextureObject(texobj);
}
