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

#include <iostream>
#include <map>
#include <string>
#include <vector>

#include <cuda_runtime_api.h>
#include "cuda_host_helpers.h"


class DeviceMemManager
{
public:
	DeviceMemManager() = default;
	~DeviceMemManager()
	{
		freeAll();
	}

	void* getMemory(size_t bytes)
	{
		void* ptr = NULL;
		succeed(cudaMalloc(&ptr, bytes));

		pointers.insert(std::pair<void const*, size_t>(ptr, bytes));
		current_consumption += bytes;
		peak_consumption = peak_consumption < current_consumption ? current_consumption : peak_consumption;
		alloc_cnt++;
		return ptr;
	}

	template<typename T>
	bool freeMemory(T*& ptr_in)
	{
		if (ptr_in == nullptr)
			return false;

		void* ptr = ptr_in;

		const auto it = pointers.find(ptr);
		if (it != pointers.end())
		{
			succeed(cudaFree(ptr));
			current_consumption -= it->second;
			pointers.erase(it);
			ptr_in = nullptr;
			free_cnt++;
			return true;
		}
		return false;
	}

	void freeAll()
	{
		//TODO: put back in
		for (auto it = pointers.begin(); it != pointers.end();)
		{
			succeed(cudaFree((void*)it->first));
			current_consumption -= it->second;
			it = pointers.erase(it);
		}
		if (current_consumption != 0)
		{
			std::cout << "ERROR: something went wrong: no pointers left but consumption != 0" << std::endl;
		}
	}

	uint64_t currentConsumption() const { return current_consumption; }
	uint64_t peakConsumption() const { return peak_consumption; }

	void printStatistics() const
	{
		std::cout << "Allocs: " << alloc_cnt << std::endl;
		std::cout << "Frees: " << free_cnt << std::endl;
		std::cout << "Currently allocated " << current_consumption << "Bytes" << std::endl;
		std::cout << "Peak consumption " << peak_consumption << "Bytes" << std::endl;
	}

	void printPointers() const
	{
		for (const auto& p : pointers)
			printf("%p: %llu\n", p.first, p.second);
	}

	void giveOwnership(void const* ptr, size_t bytes)
	{
		if (ptr == NULL)
			return;

		auto it = pointers.insert(std::pair<void const*, size_t>(ptr, bytes));
		if (it.second)
		{
			current_consumption += bytes;
			peak_consumption = peak_consumption < current_consumption ? current_consumption : peak_consumption;
		}
	}

	void takeOwnership(void const* ptr)
	{
		auto const it = pointers.find(ptr);
		if (it != pointers.end())
		{
			current_consumption -= it->second;
			pointers.erase(it);
		}
	}

	void registerConsumption(size_t bytes)
	{
		//printf("REGISTERING %d bytes\n", bytes);
		current_consumption += bytes;
		peak_consumption = peak_consumption < current_consumption ? current_consumption : peak_consumption;
	}

	void unregisterConsumption(size_t bytes)
	{
		//printf("UNREGISTERING %d bytes\n", bytes);
		current_consumption -= bytes;
	}

	void expropriate()
	{
		alloc_cnt = 0;
		free_cnt = 0;
		current_consumption = 0;
		peak_consumption = 0;
		pointers.clear();
	}
private:
	uint64_t alloc_cnt{ 0 };
	uint64_t free_cnt{ 0 };

	uint64_t current_consumption{ 0 };
	uint64_t peak_consumption{ 0 };
	std::map<void const*, size_t> pointers;
};


struct NoStateClock
{
	static void start(cudaEvent_t start, std::string msg = "")
	{
		succeed(cudaEventRecord(start));
	}

	static float stop(cudaEvent_t start, cudaEvent_t end, std::string msg = "")
	{
		float time;
		succeed(cudaEventRecord(end));
		succeed(cudaEventSynchronize(end));
		succeed(cudaEventElapsedTime(&time, start, end));
		return time;
	}
};

template<typename CLOCK>
struct TimeProfileIndividual
{
	std::vector<std::pair<std::string, float>> time;
	NoStateClock c;

	void start(cudaEvent_t start, std::string msg = "")
	{
		c.start(start, msg);
		
	}
	float stop(cudaEvent_t start, cudaEvent_t end, std::string msg = "")
	{
		float t = c.stop(start, end, msg);
		time.push_back(std::make_pair(msg, t));
		return t;
	}

};

template<typename CLOCK>
struct TimeProfileAccumulate
{
	float time{ 0 };
	NoStateClock c;

	void start(cudaEvent_t start, std::string msg = "")
	{
		c.start(start, msg);

	}
	float stop(cudaEvent_t start, cudaEvent_t end, std::string msg = "")
	{
		float t = c.stop(start, end, msg);
		time += t;
		return time;
	}
};

template<typename CLOCK>
struct NoProfiling
{
	void start(cudaEvent_t start, std::string msg = "")
	{
	}
	float stop(cudaEvent_t start, cudaEvent_t end, std::string msg = "")
	{
		return 0.0f;
	}
};