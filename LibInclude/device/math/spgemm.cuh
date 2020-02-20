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
#include "math/bhsparse_cuda.cuh"

#include <cuda_runtime_api.h>
#include "cuda_host_helpers.h"

#include <stdexcept>



inline void checkErr(int bhSparseErr)
{
	if (bhSparseErr != BHSPARSE_SUCCESS)
		throw std::runtime_error("bh sparse error");
}


texture<float, cudaTextureType1D, cudaReadModeElementType> textureMap;
__constant__  unsigned int mapSize[2];

struct MappedMul
{
	__device__
		static float multiply(float A, float B)
	{
		printf("THAT SUCKS\n");
		unsigned int av = __float2int_rn(A);
		unsigned int bv = __float2int_rn(B);
		av = min(av, mapSize[0]);
		bv = min(bv, mapSize[1]);
		return tex1Dfetch(textureMap, bv * mapSize[0] + av);
	}
};

__constant__ unsigned int mapPwr[1];
struct MappedMulFunc
{
	__device__ __forceinline__
		static int multiply(int i, int j, int n, int pwr)
	{
		return static_cast<int>(j == (((i + mapPwr[0] * pwr + 1 - 1) % n) + 1));
	}
};

template<typename OFFSET_TYPE, typename INDEX_TYPE, typename VALUE_TYPE>
__global__ void createEVals(const OFFSET_TYPE* col_ptr, const INDEX_TYPE* row_ids, const VALUE_TYPE* vals0, const VALUE_TYPE* vals1, VALUE_TYPE* vals_out, unsigned* nedges, unsigned* nextern, unsigned cols)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= cols)
		return;
	for (OFFSET_TYPE i = col_ptr[tid]; i < col_ptr[tid + 1]; ++i)
	{
		//we only need upper triangular matrix
		if (tid <= row_ids[i])
			break;

		VALUE_TYPE v0 = vals0[i];
		VALUE_TYPE v1 = vals1[i];
		VALUE_TYPE vout = 0;
		if (v0 != 0 || v1 != 0)
			vout = atomicAdd(nedges, 1) + 1;

		if (v0 != 0 && v1 != 0)
		{
			vals_out[i] = -vout;
		}
		else
		{
			if (v0 != 0 ^ v1 != 0)
				atomicAdd(nextern, 1);
			vals_out[i] = vout;
		}
	}
}

template<typename OFFSET_TYPE, typename INDEX_TYPE, typename VALUE_TYPE, typename MEM_MANAGER_TYPE>
float spgemm_mapped(
	const OFFSET_TYPE* d_a_col_ptr, const INDEX_TYPE* d_a_row_ids, const VALUE_TYPE* d_a_values,
	const OFFSET_TYPE* d_b_col_ptr, const INDEX_TYPE* d_b_row_ids, const VALUE_TYPE* d_b_values,
	OFFSET_TYPE*& d_c_col_ptr, INDEX_TYPE*& d_c_row_ids, VALUE_TYPE*& d_c_values,
	OFFSET_TYPE*& d_d_col_ptr, INDEX_TYPE*& d_d_row_ids, VALUE_TYPE*& d_d_values,
	OFFSET_TYPE*& d_e_col_ptr, INDEX_TYPE*& d_e_row_ids, VALUE_TYPE*& d_e_values,
	const size_t rows_a, const size_t cols_a_rows_b, const size_t cols_b,
	const size_t nnz_a, const size_t nnz_b, size_t& nnz_c, size_t& nnz_d, size_t& nnz_e, INDEX_TYPE& nextern,
	const VALUE_TYPE* d_a_major_map0, size_t map_size_a0, size_t map_size_b0,
	const VALUE_TYPE* d_a_major_map1, size_t map_size_a1, size_t map_size_b1, MEM_MANAGER_TYPE& manager)
{
#ifdef EXPLICIT_MAP_LOOKUP
	static bhsparse_cuda<INDEX_TYPE, VALUE_TYPE, MappedMul> bh_sparse;
#else 
	static bhsparse_cuda<INDEX_TYPE, VALUE_TYPE, MappedMulFunc> bh_sparse;
#endif// EXPLICIT_MAP_LOOKUP

	static cudaEvent_t a = 0, b;
	static bool init = false;
	if (!init)
	{
		checkErr(bh_sparse.initRunningDevice());
		cudaEventCreate(&a);
		cudaEventCreate(&b);
		init = true;
	}

	float time = 0.0f;
	float ms = 0.0f;

	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();

	//________________________________________________F_________________________________________________________
	//__________________________________________________________________________________________________________
	unsigned int msize[2] = { static_cast<unsigned int>(map_size_a0), static_cast<unsigned int>(map_size_b0) };
	cudaMemcpyToSymbol(mapSize, &msize[0], 8);

	unsigned int mpwr_F[1] = { 0 };
	cudaMemcpyToSymbol(mapPwr, &mpwr_F[0], 4);



	//cudaMalloc(&d_c_col_ptr, (cols_b + 1) * sizeof(unsigned int));
	d_c_col_ptr = reinterpret_cast<OFFSET_TYPE*>(manager.getMemory((cols_b + 1) * sizeof(OFFSET_TYPE)));
	manager.registerConsumption((cols_b + 1) * sizeof(OFFSET_TYPE));
	cudaMemset(d_c_col_ptr, 0, (cols_b + 1) * sizeof(OFFSET_TYPE));
	//csc -> csr we transpose everything and swith the order 

	cudaEventRecord(a);
	bh_sparse.initDataDevice(cols_b, cols_a_rows_b, rows_a,
		nnz_b, d_b_values, reinterpret_cast<const OFFSET_TYPE*>(d_b_col_ptr), reinterpret_cast<const INDEX_TYPE*>(d_b_row_ids),
		nnz_a, d_a_values, reinterpret_cast<const OFFSET_TYPE*>(d_a_col_ptr), reinterpret_cast<const INDEX_TYPE*>(d_a_row_ids),
		reinterpret_cast<OFFSET_TYPE*>(d_c_col_ptr));

	// STAGE 1 : compute nnzCt
	checkErr(bh_sparse.compute_nnzCt());
	checkErr(bh_sparse.kernel_barrier());

	// STAGE 2 - STEP 1 : statistics
	int nnzCt = bh_sparse.statistics();

	// STAGE 2 - STEP 2 : create Ct
	checkErr(bh_sparse.create_Ct(nnzCt));

	// STAGE 3 - STEP 1 : compute nnzC and Ct
	cudaBindTexture(nullptr, &textureMap, d_a_major_map0, &channelDesc, map_size_a0 * map_size_b0 * sizeof(float));
	checkErr(bh_sparse.compute_nnzC_Ct());

	// STAGE 3 - STEP 2 : malloc C on devices
	nnz_c = bh_sparse.count_nnz_c();
	manager.registerConsumption(nnz_c * sizeof(unsigned)); // TO compensate internal allocs
	manager.registerConsumption(nnz_c * sizeof(float)); // TO compensate internal allocs

	cudaEventRecord(b);
	cudaEventSynchronize(b);
	cudaEventElapsedTime(&ms, a, b);
	time += ms;

	//cudaMalloc(&d_c_row_ids, nnz_c * sizeof(int));
	d_c_row_ids = reinterpret_cast<INDEX_TYPE*>(manager.getMemory(nnz_c * sizeof(INDEX_TYPE)));
	cudaMemset(d_c_row_ids, 0, nnz_c * sizeof(VALUE_TYPE));
	//cudaMalloc(&d_c_values, nnz_c * sizeof(float));
	d_c_values = reinterpret_cast<VALUE_TYPE*>(manager.getMemory(nnz_c * sizeof(VALUE_TYPE)));

	cudaMemset(d_c_values, 0, nnz_c * sizeof(VALUE_TYPE));

	cudaEventRecord(a);
	bh_sparse.set_c_device(d_c_row_ids, d_c_values);

	// STAGE 4 : copy Ct to C
	checkErr(bh_sparse.copy_Ct_to_C());
	cudaEventRecord(b);
	cudaEventSynchronize(b);
	cudaEventElapsedTime(&ms, a, b);
	time += ms;

	//temporary structures not needed anymore
	manager.unregisterConsumption((cols_b + 1) * sizeof(OFFSET_TYPE));
	manager.unregisterConsumption(nnz_c * sizeof(INDEX_TYPE));
	manager.unregisterConsumption(nnz_c * sizeof(VALUE_TYPE));

	//________________________________________________F^T_______________________________________________________
	//__________________________________________________________________________________________________________
	msize[0] = map_size_a1; msize[1] = map_size_b1;
	cudaMemcpyToSymbol(mapSize, &msize[0], 8);

	unsigned int mpwr_Ft[1] = { 1 };
	cudaMemcpyToSymbol(mapPwr, &mpwr_Ft[0], 4);

	//cudaMalloc(&d_d_col_ptr, (cols_b + 1) * sizeof(unsigned int));
	d_d_col_ptr = reinterpret_cast<OFFSET_TYPE*>(manager.getMemory((cols_b + 1) * sizeof(OFFSET_TYPE)));
	manager.registerConsumption((cols_b + 1) * sizeof(OFFSET_TYPE));
	cudaMemset(d_d_col_ptr, 0, (cols_b + 1) * sizeof(OFFSET_TYPE));

	cudaEventRecord(a);
	//csc -> csr we transpose everything and swith the order 
	bh_sparse.initDataDevice(cols_b, cols_a_rows_b, rows_a,
		nnz_b, d_b_values, reinterpret_cast<const OFFSET_TYPE*>(d_b_col_ptr), reinterpret_cast<const INDEX_TYPE*>(d_b_row_ids),
		nnz_a, d_a_values, reinterpret_cast<const OFFSET_TYPE*>(d_a_col_ptr), reinterpret_cast<const INDEX_TYPE*>(d_a_row_ids),
		reinterpret_cast<OFFSET_TYPE*>(d_d_col_ptr));

	// STAGE 1 : compute nnzCt
	checkErr(bh_sparse.compute_nnzCt());
	checkErr(bh_sparse.kernel_barrier());

	// STAGE 2 - STEP 1 : statistics
	nnzCt = bh_sparse.statistics();

	// STAGE 2 - STEP 2 : create Ct
	checkErr(bh_sparse.create_Ct(nnzCt));

	// STAGE 3 - STEP 1 : compute nnzC and Ct
	cudaBindTexture(nullptr, &textureMap, d_a_major_map1, &channelDesc, map_size_a1 * map_size_b1 * sizeof(float));
	checkErr(bh_sparse.compute_nnzC_Ct());

	// STAGE 3 - STEP 2 : malloc C on devices
	nnz_d = bh_sparse.count_nnz_c();
	manager.registerConsumption(nnz_d * sizeof(INDEX_TYPE));
	manager.registerConsumption(nnz_d * sizeof(VALUE_TYPE));

	cudaEventRecord(b);
	cudaEventSynchronize(b);
	cudaEventElapsedTime(&ms, a, b);
	time += ms;

	//cudaMalloc(&d_d_row_ids, nnz_d * sizeof(int));
	d_d_row_ids = reinterpret_cast<INDEX_TYPE*>(manager.getMemory(nnz_d * sizeof(INDEX_TYPE)));
	cudaMemset(d_d_row_ids, 0, nnz_d * sizeof(INDEX_TYPE));
	//cudaMalloc(&d_d_values, nnz_d * sizeof(float));
	d_d_values = reinterpret_cast<VALUE_TYPE*>(manager.getMemory(nnz_d * sizeof(VALUE_TYPE)));
	cudaMemset(d_d_values, 0, nnz_d * sizeof(VALUE_TYPE));

	cudaEventRecord(a);
	bh_sparse.set_c_device(reinterpret_cast<INDEX_TYPE*>(d_d_row_ids), d_d_values);

	// STAGE 4 : copy Ct to C
	checkErr(bh_sparse.copy_Ct_to_C());

	cudaEventRecord(b);
	cudaEventSynchronize(b);
	cudaEventElapsedTime(&ms, a, b);
	time += ms;

	//temporary structures not needed anymore
	manager.unregisterConsumption((cols_b + 1) * sizeof(OFFSET_TYPE));
	manager.unregisterConsumption(nnz_d * sizeof(INDEX_TYPE));
	manager.unregisterConsumption(nnz_d * sizeof(VALUE_TYPE));

	//________________________________________________E_______________________________________________________
	//__________________________________________________________________________________________________________

	nnz_e = nnz_d;
	//cudaMalloc((void**)&d_e_col_ptr, (cols_b + 1) * sizeof(unsigned int));
	d_e_col_ptr = reinterpret_cast<OFFSET_TYPE*>(manager.getMemory((cols_b + 1) * sizeof(OFFSET_TYPE)));
	cudaMemcpy(d_e_col_ptr, d_d_col_ptr, (cols_b + 1) * sizeof(OFFSET_TYPE), cudaMemcpyHostToDevice);

	//cudaMalloc(&d_e_row_ids, nnz_e * sizeof(unsigned int));
	d_e_row_ids = reinterpret_cast<INDEX_TYPE*>(manager.getMemory(nnz_e * sizeof(INDEX_TYPE)));
	cudaMemcpy(d_e_row_ids, d_d_row_ids, nnz_e * sizeof(INDEX_TYPE), cudaMemcpyHostToDevice);

	//cudaMalloc(&d_e_values, nnz_e * sizeof(float));
	d_e_values = reinterpret_cast<VALUE_TYPE*>(manager.getMemory(nnz_e * sizeof(VALUE_TYPE)));
	cudaMemset(d_e_values, 0, nnz_e * sizeof(VALUE_TYPE));

	//cudaMalloc((void**)&nedges, sizeof(unsigned));
	unsigned* nedges = reinterpret_cast<unsigned*>(manager.getMemory(sizeof(unsigned)));
	cudaMemset(nedges, 0, sizeof(unsigned));
	//cudaMalloc((void**)&d_nextern, sizeof(unsigned));
	unsigned* d_nextern = reinterpret_cast<unsigned*>(manager.getMemory(sizeof(unsigned)));
	cudaMemset(d_nextern, 0, sizeof(unsigned));

	cudaEventRecord(a);

	size_t block_dim = 256;
	size_t grid_dim = divup(size_t(nnz_e), block_dim);
	createEVals << <grid_dim, block_dim >> > (d_c_col_ptr, d_c_row_ids, d_c_values, d_d_values, d_e_values, nedges, d_nextern, cols_b);
	cudaMemcpy(&nextern, d_nextern, sizeof(unsigned), cudaMemcpyDeviceToHost);
	cudaEventRecord(b);
	cudaEventSynchronize(b);
	cudaEventElapsedTime(&ms, a, b);
	time += ms;

	manager.freeMemory(nedges);
	manager.freeMemory(d_nextern);

	return time;
}
