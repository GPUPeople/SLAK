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

#include <algorithm>
#include<vector>

#include"profiling.h"

struct CatmullClark
{
	////////////////////////////////////////////////////////////////////////////////
	/// Data Structures

	template<typename MESH_INFO>
	struct Context
	{
		using index_t = typename MESH_INFO::index_t;
		// config
		size_t block_dim{ 256 };

		// state
		bool is_initialized{ false };

		//data pointers
		index_t* d_order_buffer{ nullptr };
		index_t* d_S_buffer{ nullptr };
		index_t* d_mapping_buffer{ nullptr };
		index_t* d_face_adjacency{ nullptr };

		// boundary handling
		bool has_boundary{ true };
		index_t nextern{ 0 };
		index_t* d_nextern{ 0 };
		index_t* d_vextern{ nullptr };
	};

	template<typename PROFILER_TYPE, typename MEM_MANAGER_TYPE>
	struct ProfilinInfo
	{
		PROFILER_TYPE prof;
		MEM_MANAGER_TYPE mem;
	};


	template<typename OFFSET_TYPE, typename INDEX_TYPE, typename VALUE_TYPE, typename VERTEX_TYPE>
	struct MeshInfo
	{
		enum class MeshType { TRI, QUAD, POLY };
		MeshType type;

		size_t nfaces{ 0 };
		size_t nverts{ 0 };
		size_t nnz{ 0 };
		size_t max_face_order{ 0 };

		bool is_reduced{ false };

		using offset_t = OFFSET_TYPE;
		OFFSET_TYPE* ptr{ nullptr };

		using index_t = INDEX_TYPE;
		INDEX_TYPE* ids{ nullptr };

		using value_t = VALUE_TYPE;
		VALUE_TYPE* vals{ nullptr };

		using vertex_t = VERTEX_TYPE;
		VERTEX_TYPE* verts{ nullptr };

		void reset()
		{
			*this = MeshInfo();
		}

		void freeAndReset()
		{
			if (ptr != nullptr) succeed(cudaFree(ptr)); ptr = nullptr;
			if (ids != nullptr) succeed(cudaFree(ids)); ids = nullptr;
			if (vals != nullptr) succeed(cudaFree(vals)); vals = nullptr;
			if (verts != nullptr) succeed(cudaFree(verts)); verts = nullptr;
			*this = MeshInfo();
		}

		template<typename MEM_MANAGER>
		void freeAndReset(MEM_MANAGER& mem)
		{
			if (ptr != nullptr && !mem.freeMemory(ptr))
			{
				succeed(cudaFree(ptr)); ptr = nullptr;
			}
			if (ids != nullptr && !mem.freeMemory(ids))
			{
				succeed(cudaFree(ids)); ids = nullptr;
			}
			if (vals != nullptr && !mem.freeMemory(vals))
			{
				succeed(cudaFree(vals)); vals = nullptr;
			}
			if (verts != nullptr && !mem.freeMemory(verts))
			{
				succeed(cudaFree(verts)); verts = nullptr;
			}

			*this = MeshInfo();
		}
	};

	////////////////////////////////////////////////////////////////////////////////
	/// Methods
private:

	////////////////////////////////////////////////////////////////////////////////
	/// Quadrilateral Mesh Subdiv
	template<typename MESH_INFO, typename MEMORY_MANAGER, typename PROFILING>
	static void subdivideVertexDataQuadMesh(MESH_INFO const& cmesh, MESH_INFO& rmesh, Context<MESH_INFO>& ctx, MEMORY_MANAGER& mem, PROFILING& prof);

	template<typename MESH_INFO, typename MEMORY_MANAGER, typename PROFILING>
	static void subdivideTopologyQuadMesh(MESH_INFO const& cmesh, MESH_INFO& rmesh, Context<MESH_INFO>& ctx, MEMORY_MANAGER& mem, PROFILING& prof);

	template<typename MESH_INFO, typename MEMORY_MANAGER, typename PROFILING>
	static void initQuadMesh(MESH_INFO const& cmesh, MESH_INFO& rmesh, Context<MESH_INFO>& ctx, MEMORY_MANAGER& mem, PROFILING& prof);


	////////////////////////////////////////////////////////////////////////////////
	/// Polygonal Mesh Subdiv
	template<typename MESH_INFO, typename MEMORY_MANAGER, typename PROFILING>
	static void subdivideVertexDataPolyMesh(MESH_INFO const& cmesh, MESH_INFO& rmesh, Context<MESH_INFO>& ctx, MEMORY_MANAGER& mem, PROFILING& prof);

	template<typename MESH_INFO, typename MEMORY_MANAGER, typename PROFILING>
	static void subdivideTopologyPolyMesh(MESH_INFO const& cmesh, MESH_INFO& rmesh, Context<MESH_INFO>& ctx, MEMORY_MANAGER& mem, PROFILING& prof);

	template<typename MESH_INFO, typename MEMORY_MANAGER, typename PROFILING>
	static void initPolyMesh(MESH_INFO const& cmesh, MESH_INFO& rmesh, Context<MESH_INFO>& ctx, MEMORY_MANAGER& mem, PROFILING& prof);



public:

	template<typename MESH_INFO>
	static void subdivideIteratively(MESH_INFO const& cmesh, MESH_INFO& rmesh, int target_level);
};
