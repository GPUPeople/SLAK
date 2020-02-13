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
#include <builtin_types.h>


template<typename OFFSET_TYPE, typename INDEX_TYPE, typename VALUE_TYPE>
void spmv_left_mapped(const OFFSET_TYPE* d_col_offsets, const INDEX_TYPE* d_row_ids, const VALUE_TYPE* d_values, const VALUE_TYPE* x, VALUE_TYPE* y, int rows, int cols, const VALUE_TYPE* map, int map_size);
template<typename OFFSET_TYPE, typename INDEX_TYPE, typename VALUE_TYPE>
void spmv_left_mapped_f4(const OFFSET_TYPE* d_col_offsets, const INDEX_TYPE* d_row_ids, const VALUE_TYPE* d_values, const float4* x, float4* y, int rows, int cols, const VALUE_TYPE* map, int map_size);
template<typename OFFSET_TYPE, typename INDEX_TYPE, typename VALUE_TYPE>
void spmv_right_mapped(const OFFSET_TYPE* d_col_offsets, const INDEX_TYPE* d_row_ids, const VALUE_TYPE* d_values, const VALUE_TYPE* x, VALUE_TYPE* y, int rows, int cols, const VALUE_TYPE* map, int map_size);
template<typename OFFSET_TYPE, typename INDEX_TYPE, typename VALUE_TYPE>
void spmv_right_mapped_f4(const OFFSET_TYPE* d_col_offsets, const INDEX_TYPE* d_row_ids, const VALUE_TYPE* d_values, const float4* x, float4* y, int rows, int cols, const VALUE_TYPE* map, int map_size);



template<typename OFFSET_TYPE, typename INDEX_TYPE, typename VALUE_TYPE, typename MEM_MANAGER_TYPE>
float spgemm_mapped(
	const OFFSET_TYPE* d_a_col_offsets, const INDEX_TYPE* d_a_row_ids, const VALUE_TYPE* d_a_values,
	const OFFSET_TYPE* d_b_col_offsets, const INDEX_TYPE* d_b_row_ids, const VALUE_TYPE* d_b_values,
	OFFSET_TYPE*& d_c_col_offsets, INDEX_TYPE*& d_c_row_ids, VALUE_TYPE*& d_c_values,
	OFFSET_TYPE*& d_d_col_offsets, INDEX_TYPE*& d_d_row_ids, VALUE_TYPE*& d_d_values,
	OFFSET_TYPE*& d_e_col_offsets, INDEX_TYPE*& d_e_row_ids, VALUE_TYPE*& d_e_values,
	const size_t rows_a, const size_t cols_a_rows_b, const size_t cols_b,
	const size_t nnz_a, const size_t nnz_b, const size_t& nnz_c, size_t& nnz_d, size_t& nnz_e, INDEX_TYPE& nextern,
	const VALUE_TYPE* d_a_major_map0, size_t map_size_a0, size_t map_size_b0,
	const VALUE_TYPE* d_a_major_map1, size_t map_size_a1, size_t map_size_b1, MEM_MANAGER_TYPE& manager);

