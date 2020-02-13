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

#include "cub/device/device_scan.cuh"

struct CubFunctions
{
	template<typename T, typename MEM_MANAGER>
	static void scan_exclusive(T*& d_in, T*& d_out, unsigned int num_items, MEM_MANAGER& mem)
	{
		void* d_temp_storage = nullptr;
		size_t temp_storage_bytes = 0;
		cudaMemset(d_out, 0, sizeof(T));
		cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_in, d_out + 1, num_items - 1);
		d_temp_storage = mem.getMemory(temp_storage_bytes);
		cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_in, d_out + 1, num_items - 1);
		mem.freeMemory(d_temp_storage);
	}
};

