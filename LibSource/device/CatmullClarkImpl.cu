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

#include "CatmullClark.h"
#include "CatmullClarkKernels.cuh"

#include <iostream>
#include <iomanip>
#include <cusparse.h>
#include "cub_wrappers.cuh"


////////////////////////////////////////////////////////////////////////////////
/// Quadrilateral Mesh Subdiv
template<typename MESH_INFO, typename MEMORY_MANAGER, typename PROFILING>
void CatmullClark::subdivideVertexDataQuadMesh(MESH_INFO const& cmesh, MESH_INFO& rmesh, Context<MESH_INFO>& ctx, MEMORY_MANAGER& mem, PROFILING& prof)
{
	using offset_t = typename MESH_INFO::offset_t;
	using index_t = typename MESH_INFO::index_t;
	using vertex_t = typename MESH_INFO::vertex_t;

	cudaEvent_t start, stop;
	succeed(cudaEventCreate(&start));
	succeed(cudaEventCreate(&stop));

	constexpr size_t ncomponents = 4;

	mem.giveOwnership(reinterpret_cast<const void*>(cmesh.ids), cmesh.nnz * sizeof(offset_t));
	mem.giveOwnership(reinterpret_cast<const void*>(cmesh.verts), cmesh.nverts * sizeof(vertex_t)* ncomponents);
	mem.giveOwnership(reinterpret_cast<const void*>(rmesh.ids), rmesh.nnz * sizeof(offset_t));


	size_t nedges = (cmesh.nnz - ctx.nextern) / 2 + ctx.nextern;
	rmesh.verts = reinterpret_cast<vertex_t*>(mem.getMemory(rmesh.nverts * ncomponents * sizeof(vertex_t)));
	vertex_t* d_updated_vertex_data = rmesh.verts;
	vertex_t* d_facepoints = d_updated_vertex_data + cmesh.nverts * ncomponents;
	vertex_t* d_edgepoints = d_facepoints + cmesh.nfaces * ncomponents;
	succeed(cudaMemset(d_edgepoints, 0, nedges * ncomponents * sizeof(vertex_t)));

	size_t grid_dim = divup(cmesh.nnz * ncomponents, ctx.block_dim);
	prof.start(start, "Calculating facepoints and edgepoints");
	CCKernels::calculateFaceEdgepointsQuad << <grid_dim, ctx.block_dim >> > (cmesh.ids, rmesh.ids, cmesh.verts, d_facepoints, d_edgepoints, cmesh.nnz, cmesh.nverts, ncomponents);
	prof.stop(start, stop);

	index_t* d_S_ptr = ctx.d_S_buffer;
	index_t* d_S_ids = d_S_ptr + cmesh.nverts + 1;
	index_t* d_S_vals = d_S_ids + cmesh.nnz;

	grid_dim = divup(cmesh.nverts, ctx.block_dim);
	prof.start(start, "vertexSum");
	CCKernels::vertexSum << <grid_dim, ctx.block_dim >> > (d_S_ptr, d_S_ids, cmesh.verts, ctx.d_order_buffer, d_updated_vertex_data, cmesh.nverts, ncomponents);
	prof.stop(start, stop);

	grid_dim = divup(cmesh.nfaces * ncomponents, ctx.block_dim >> 1);
	prof.start(start, "facepointSum");
	CCKernels::facepointSumQuad << < grid_dim, ctx.block_dim >> 1 >> > (cmesh.ids, d_facepoints, ctx.d_order_buffer, d_updated_vertex_data, cmesh.nverts, cmesh.nfaces, ncomponents);
	prof.stop(start, stop);


	if (ctx.has_boundary)
	{
		prof.start(start, "Correcting boundary position");
		CCKernels::correctBoundaryS1 << <divup(static_cast<size_t>(ctx.nextern), ctx.block_dim >> 1), ctx.block_dim >> > (
			cmesh.ids, d_S_ptr, d_S_ids, d_S_vals, ctx.d_vextern,
			cmesh.verts, rmesh.verts, ctx.nextern, cmesh.nverts, cmesh.nfaces);
		CCKernels::correctBoundaryS2 << <divup(static_cast<size_t>(ctx.nextern), ctx.block_dim >> 1), ctx.block_dim >> > (
			cmesh.ids, d_S_ptr, d_S_ids, d_S_vals, ctx.d_vextern,
			cmesh.verts, rmesh.verts, ctx.nextern, cmesh.nverts, cmesh.nfaces);
		prof.stop(start, stop);
	}

	//TODO: re-implement (aka port) creases
	//if (has_creases && !creases_decayed)
	//	time += handleCreases(C, d_vertex_data, d_refined_vertexdata, nf, cmesh.nverts);

	mem.takeOwnership(reinterpret_cast<const void*>(cmesh.ids));
	mem.takeOwnership(reinterpret_cast<const void*>(cmesh.verts));
	mem.takeOwnership(reinterpret_cast<const void*>(rmesh.ids));
	mem.takeOwnership(reinterpret_cast<const void*>(rmesh.verts));

	succeed(cudaEventDestroy(start));
	succeed(cudaEventDestroy(stop));
}

template<typename MESH_INFO, typename MEMORY_MANAGER, typename PROFILING>
void CatmullClark::subdivideTopologyQuadMesh(MESH_INFO const& cmesh, MESH_INFO& rmesh, Context<MESH_INFO>& ctx, MEMORY_MANAGER& mem, PROFILING& prof)
{
	using index_t = typename MESH_INFO::index_t;

	cudaEvent_t start, stop;
	succeed(cudaEventCreate(&start));
	succeed(cudaEventCreate(&stop));

	mem.giveOwnership(reinterpret_cast<const void*>(cmesh.ids), cmesh.nnz * sizeof(index_t));

	rmesh.nfaces = cmesh.nfaces * 4;
	rmesh.nnz = rmesh.nfaces * 4;
	rmesh.nverts = cmesh.nverts + cmesh.nfaces + (cmesh.nnz - ctx.nextern) / 2 + ctx.nextern;


	rmesh.ids = reinterpret_cast<index_t*>(mem.getMemory(rmesh.nnz * sizeof(index_t)));
	rmesh.ptr = nullptr;
	rmesh.vals = nullptr;

	size_t grid_dim = divup(cmesh.nnz, ctx.block_dim);
	index_t* d_S_ptr = ctx.d_S_buffer;
	index_t* d_S_ids = d_S_ptr + cmesh.nverts + 1;
	index_t* d_S_vals = d_S_ids + cmesh.nnz;
	if (!ctx.has_boundary)
	{
		prof.start(start, "Refining topology");
		CCKernels::refineTopologyQuad << <grid_dim, ctx.block_dim >> > (cmesh.ids, d_S_ptr, d_S_ids, d_S_vals, rmesh.ids, cmesh.nnz, cmesh.nverts, cmesh.nfaces);
		prof.stop(start, stop);
	}
	else
	{
		prof.start(start, "Refining topology with boundary");
		CCKernels::refineTopologyQuadExtern << <grid_dim, ctx.block_dim >> > (cmesh.ids, d_S_ptr, d_S_ids, d_S_vals, rmesh.ids, cmesh.nnz, cmesh.nverts, cmesh.nfaces);
		prof.stop(start, stop);
	}

	mem.takeOwnership(reinterpret_cast<const void*>(cmesh.ids));
	mem.takeOwnership(reinterpret_cast<const void*>(rmesh.ids));

	succeed(cudaEventDestroy(start));
	succeed(cudaEventDestroy(stop));
}

template<typename MESH_INFO, typename MEMORY_MANAGER, typename PROFILING>
void CatmullClark::initQuadMesh(MESH_INFO const& cmesh, MESH_INFO& rmesh, Context<MESH_INFO>& ctx, MEMORY_MANAGER& mem, PROFILING& prof)
{
	using index_t = typename MESH_INFO::index_t;

	cudaEvent_t start, stop;
	succeed(cudaEventCreate(&start));
	succeed(cudaEventCreate(&stop));

	mem.giveOwnership(reinterpret_cast<const void*>(cmesh.ids), cmesh.nnz * sizeof(index_t));

	ctx.d_order_buffer = reinterpret_cast<index_t*>(mem.getMemory(cmesh.nverts * sizeof(index_t)));
	succeed(cudaMemset(reinterpret_cast<void*>(ctx.d_order_buffer), 0, cmesh.nverts * sizeof(index_t)));

	ctx.d_S_buffer = reinterpret_cast<index_t*>(mem.getMemory((cmesh.nverts + 1 + cmesh.nnz + cmesh.nnz) * sizeof(index_t)));
	index_t* d_S_col_ptr = ctx.d_S_buffer;
	index_t* d_S_row_ids = d_S_col_ptr + cmesh.nverts + 1;
	index_t* d_S_epids_lt = d_S_row_ids + cmesh.nnz;

	index_t* d_tmp_buffer = reinterpret_cast<index_t*>(mem.getMemory((cmesh.nnz + cmesh.nnz) * sizeof(index_t)));
	index_t* d_offsets = d_tmp_buffer;
	index_t* d_epids_tmp = d_offsets + cmesh.nnz;

	size_t grid_dim = divup(cmesh.nnz, ctx.block_dim);
	prof.start(start, "Evaluating valences");
	CCKernels::evaluateValences << <grid_dim, ctx.block_dim >> > (cmesh.ids, ctx.d_order_buffer, d_offsets, cmesh.nnz);
	prof.stop(start, stop);

	prof.start(start, "Scanning valences");
	CubFunctions::scan_exclusive(ctx.d_order_buffer, d_S_col_ptr, static_cast<unsigned>(cmesh.nverts + 1), mem);
	prof.stop(start, stop);

	prof.start(start, "Evaluating vertex adjacency");
	CCKernels::evaluateAdjacenciesQuad << <grid_dim, ctx.block_dim >> > (cmesh.ids, d_S_col_ptr, d_offsets, d_S_row_ids, d_epids_tmp, cmesh.nnz);
	prof.stop(start, stop);

	ctx.nextern = 0;
	if (ctx.has_boundary)
	{
		ctx.d_nextern = reinterpret_cast<index_t*>(mem.getMemory(sizeof(index_t)));
		succeed(cudaMemset(reinterpret_cast<void*>(ctx.d_nextern), 0, sizeof(index_t)));
		prof.start(start, "Checking for boundary");
		CCKernels::checkForBoundaryQuad << <grid_dim, ctx.block_dim >> > (cmesh.ids, d_S_col_ptr, d_S_row_ids, d_epids_tmp, ctx.d_nextern, cmesh.nnz);
		prof.stop(start, stop);

		succeed(cudaMemcpy(&ctx.nextern, ctx.d_nextern, sizeof(index_t), cudaMemcpyDeviceToHost));
		if (ctx.nextern != 0)
		{
			ctx.has_boundary = true;
			ctx.d_vextern = reinterpret_cast<index_t*>(mem.getMemory(ctx.nextern * sizeof(index_t)));
		}
		else
		{
			ctx.has_boundary = false;
			mem.freeMemory(ctx.d_nextern);
		}
	}

	prof.start(start, "Scanning vids");
	CubFunctions::scan_exclusive(d_epids_tmp, d_S_epids_lt, static_cast<unsigned>(cmesh.nnz), mem);
	prof.stop(start, stop);

	if (ctx.has_boundary)
	{
		prof.start(start, "Marking boundary boundary");
		CCKernels::markBoundaryQuad << <grid_dim, ctx.block_dim >> > (cmesh.ids, d_S_col_ptr, d_S_row_ids, d_S_epids_lt, ctx.d_vextern, ctx.d_nextern, cmesh.nnz);
		prof.stop(start, stop);
	}

	mem.freeMemory(d_tmp_buffer);

	mem.takeOwnership(reinterpret_cast<const void*>(cmesh.ids));

	succeed(cudaEventDestroy(start));
	succeed(cudaEventDestroy(stop));
}


////////////////////////////////////////////////////////////////////////////////
/// Polygonal Mesh Subdiv
template<typename MESH_INFO, typename MEMORY_MANAGER, typename PROFILING>
void CatmullClark::subdivideVertexDataPolyMesh(MESH_INFO const& cmesh, MESH_INFO& rmesh, Context<MESH_INFO>& ctx, MEMORY_MANAGER& mem, PROFILING& prof)
{
	using offset_t = typename MESH_INFO::offset_t;
	using index_t = typename MESH_INFO::index_t;
	using vertex_t = typename MESH_INFO::vertex_t;

	cudaEvent_t start, stop;
	succeed(cudaEventCreate(&start));
	succeed(cudaEventCreate(&stop));

	constexpr size_t ncomponents = 4;

	mem.giveOwnership(reinterpret_cast<const void*>(cmesh.ptr), (cmesh.nfaces + 1) * sizeof(offset_t));
	mem.giveOwnership(reinterpret_cast<const void*>(cmesh.ids), cmesh.nnz * sizeof(offset_t));
	mem.giveOwnership(reinterpret_cast<const void*>(cmesh.verts), cmesh.nverts * sizeof(vertex_t)* ncomponents);
	mem.giveOwnership(reinterpret_cast<const void*>(rmesh.ids), rmesh.nnz * sizeof(offset_t));

	size_t nedges = (cmesh.nnz - ctx.nextern) / 2 + ctx.nextern;
	rmesh.verts = reinterpret_cast<vertex_t*>(mem.getMemory(rmesh.nverts * ncomponents * sizeof(vertex_t)));
	vertex_t* d_updated_vertex_data = rmesh.verts;
	vertex_t* d_facepoints = d_updated_vertex_data + cmesh.nverts * ncomponents;
	vertex_t* d_edgepoints = d_facepoints + cmesh.nfaces * ncomponents;
	succeed(cudaMemset(d_edgepoints, 0, nedges * ncomponents * sizeof(vertex_t)));

	size_t grid_dim = divup(cmesh.nfaces * ncomponents, ctx.block_dim);
	prof.start(start, "Calculating facepoints - Poly");
	CCKernels::calculateFacepoints << <grid_dim, ctx.block_dim >> >
		(cmesh.ptr, cmesh.ids, cmesh.verts, d_facepoints, cmesh.nfaces, cmesh.nverts, ncomponents);
	prof.stop(start, stop);

	index_t* d_S_ptr = ctx.d_S_buffer;
	index_t* d_S_ids = d_S_ptr + cmesh.nverts + 1;
	index_t* d_S_vals = d_S_ids + cmesh.nnz;

	grid_dim = divup(cmesh.nnz * ncomponents, ctx.block_dim);
	prof.start(start, "Calculating edgepoints - Poly");
	CCKernels::calculateEdgepoints << <grid_dim, ctx.block_dim >> >
		(rmesh.ids, d_S_ptr, d_S_ids, d_S_vals, cmesh.verts, d_facepoints, d_edgepoints, cmesh.nnz, cmesh.nverts, cmesh.nfaces, ncomponents);
	prof.stop(start, stop);

	grid_dim = divup(cmesh.nverts, ctx.block_dim);
	prof.start(start, "vertexSum - Poly");
	CCKernels::vertexSum << <grid_dim, ctx.block_dim >> >
		(d_S_ptr, d_S_ids, cmesh.verts, ctx.d_order_buffer, d_updated_vertex_data, cmesh.nverts, ncomponents);
	prof.stop(start, stop);

	grid_dim = divup(cmesh.nfaces * ncomponents, ctx.block_dim);
	prof.start(start, "facepointSum - Poly");
	CCKernels::facepointSum << < grid_dim, ctx.block_dim >> >
		(cmesh.ptr, cmesh.ids, d_facepoints, ctx.d_order_buffer, d_updated_vertex_data, cmesh.nverts, cmesh.nfaces, ncomponents);
	prof.stop(start, stop);

	if (ctx.has_boundary)
	{
		prof.start(start, "Correcting boundary position - Poly");
		CCKernels::correctBoundaryS1 << <divup(static_cast<size_t>(ctx.nextern), ctx.block_dim >> 1), ctx.block_dim >> > (
			cmesh.ids, d_S_ptr, d_S_ids, d_S_vals, ctx.d_vextern,
			cmesh.verts, rmesh.verts, ctx.nextern, cmesh.nverts, cmesh.nfaces);
		CCKernels::correctBoundaryS2 << <divup(static_cast<size_t>(ctx.nextern), ctx.block_dim >> 1), ctx.block_dim >> > (
			cmesh.ids, d_S_ptr, d_S_ids, d_S_vals, ctx.d_vextern,
			cmesh.verts, rmesh.verts, ctx.nextern, cmesh.nverts, cmesh.nfaces);
		prof.stop(start, stop);
	}

	//TODO: re-implement (aka port) creases
	//if (has_creases && !creases_decayed)
	//	time += handleCreases(C, d_vertex_data, d_refined_vertexdata, nf, cmesh.nverts);

	mem.takeOwnership(reinterpret_cast<const void*>(cmesh.ptr));
	mem.takeOwnership(reinterpret_cast<const void*>(cmesh.ids));
	mem.takeOwnership(reinterpret_cast<const void*>(cmesh.verts));
	mem.takeOwnership(reinterpret_cast<const void*>(rmesh.ids));
	mem.takeOwnership(reinterpret_cast<const void*>(rmesh.verts));


	succeed(cudaEventDestroy(start));
	succeed(cudaEventDestroy(stop));
}

template<typename MESH_INFO, typename MEMORY_MANAGER, typename PROFILING>
void CatmullClark::subdivideTopologyPolyMesh(MESH_INFO const& cmesh, MESH_INFO& rmesh, Context<MESH_INFO>& ctx, MEMORY_MANAGER& mem, PROFILING& prof)
{
	using offset_t = typename MESH_INFO::offset_t;
	using index_t = typename MESH_INFO::index_t;

	cudaEvent_t start, stop;
	succeed(cudaEventCreate(&start));
	succeed(cudaEventCreate(&stop));

	mem.giveOwnership(reinterpret_cast<const void*>(cmesh.ptr), (cmesh.nfaces + 1) * sizeof(offset_t));
	mem.giveOwnership(reinterpret_cast<const void*>(cmesh.ids), cmesh.nnz * sizeof(index_t));

	succeed(cudaMemcpy(reinterpret_cast<void*>(&rmesh.nfaces), cmesh.ptr + cmesh.nfaces, sizeof(index_t), cudaMemcpyDeviceToHost));
	rmesh.nnz = rmesh.nfaces * 4;
	rmesh.nverts = cmesh.nverts + cmesh.nfaces + (cmesh.nnz - ctx.nextern) / 2 + ctx.nextern;


	rmesh.ids = reinterpret_cast<index_t*>(mem.getMemory(rmesh.nnz * sizeof(index_t)));
	rmesh.ptr = nullptr;
	rmesh.vals = nullptr;

	size_t grid_dim = divup(cmesh.nnz, ctx.block_dim);
	index_t* d_S_ptr = ctx.d_S_buffer;
	index_t* d_S_ids = d_S_ptr + cmesh.nverts + 1;
	index_t* d_S_vals = d_S_ids + cmesh.nnz;
	if (!ctx.has_boundary)
	{
		prof.start(start, "Refining topology");
		CCKernels::refineTopology << <grid_dim, ctx.block_dim >> >
			(ctx.d_mapping_buffer, cmesh.ids, cmesh.ptr, d_S_ptr, d_S_ids, d_S_vals, rmesh.ids, cmesh.nfaces, cmesh.nverts, cmesh.nnz);
		prof.stop(start, stop);
	}
	else
	{
		prof.start(start, "Refining topology with boundary");
		CCKernels::refineTopologyExtern << <grid_dim, ctx.block_dim >> >
			(ctx.d_mapping_buffer, cmesh.ids, cmesh.ptr, d_S_ptr, d_S_ids, d_S_vals, rmesh.ids, cmesh.nfaces, cmesh.nverts, cmesh.nnz);
		prof.stop(start, stop);
	}

	mem.takeOwnership(reinterpret_cast<const void*>(cmesh.ptr));
	mem.takeOwnership(reinterpret_cast<const void*>(cmesh.ids));
	mem.takeOwnership(reinterpret_cast<const void*>(rmesh.ids));

	succeed(cudaEventDestroy(start));
	succeed(cudaEventDestroy(stop));
}

template<typename MESH_INFO, typename MEMORY_MANAGER, typename PROFILING>
void CatmullClark::initPolyMesh(MESH_INFO const& cmesh, MESH_INFO& rmesh, Context<MESH_INFO>& ctx, MEMORY_MANAGER& mem, PROFILING& prof)
{
	using offset_t = typename MESH_INFO::offset_t;
	using index_t = typename MESH_INFO::index_t;
	using value_t = typename MESH_INFO::value_t;
	
	cudaEvent_t start, stop;
	succeed(cudaEventCreate(&start));
	succeed(cudaEventCreate(&stop));

	mem.giveOwnership(reinterpret_cast<const void*>(cmesh.ptr), (cmesh.nfaces + 1) * sizeof(offset_t));
	mem.giveOwnership(reinterpret_cast<const void*>(cmesh.ids), cmesh.nnz * sizeof(index_t));

	ctx.d_order_buffer = reinterpret_cast<index_t*>(mem.getMemory((cmesh.nverts + cmesh.nfaces) * sizeof(index_t)));
	index_t* d_valences = ctx.d_order_buffer;
	index_t* d_face_orders = d_valences + cmesh.nverts;
	succeed(cudaMemset(reinterpret_cast<void*>(d_valences), 0, cmesh.nverts * sizeof(index_t)));

	ctx.d_S_buffer = reinterpret_cast<index_t*>(mem.getMemory((cmesh.nverts + 1 + cmesh.nnz + cmesh.nnz) * sizeof(index_t)));
	index_t* d_S_col_ptr = ctx.d_S_buffer;
	index_t* d_S_row_ids = d_S_col_ptr + cmesh.nverts + 1;
	index_t* d_S_epids_lt = d_S_row_ids + cmesh.nnz;

	ctx.d_mapping_buffer = reinterpret_cast<index_t*>(mem.getMemory(cmesh.nnz * sizeof(index_t)));

	index_t* d_tmp_buffer = reinterpret_cast<index_t*>(mem.getMemory((cmesh.nnz + cmesh.nnz) * sizeof(index_t)));
	index_t* d_offsets = d_tmp_buffer;
	index_t* d_epids_tmp = d_offsets + cmesh.nnz;
	
	size_t grid_dim = divup(cmesh.nnz, ctx.block_dim);
	prof.start(start, "Evaluating valences - Poly");
	CCKernels::evaluateValences << <grid_dim, ctx.block_dim >> > (cmesh.ids, ctx.d_order_buffer, d_offsets, cmesh.nnz);
	prof.stop(start, stop);

	prof.start(start, "Scanning valences - Poly");
	CubFunctions::scan_exclusive(d_valences, d_S_col_ptr, static_cast<unsigned>(cmesh.nverts + 1), mem);
	prof.stop(start, stop);

	grid_dim = divup(cmesh.nfaces, ctx.block_dim);
	prof.start(start, "Evaluating vertex adjacency - Poly");
	CCKernels::evaluateAdjacencies << <grid_dim, ctx.block_dim >> > (cmesh.ptr, cmesh.ids, d_S_col_ptr, d_offsets, d_S_row_ids, d_epids_tmp, d_face_orders, cmesh.nfaces);
	prof.stop(start, stop);

	prof.start(start, "Creating thread-face mapping - Poly");
	CCKernels::createThreadFaceMapping << <grid_dim, ctx.block_dim >> > (cmesh.ptr, ctx.d_mapping_buffer, cmesh.nfaces);
	prof.stop(start, stop);

	ctx.nextern = 0;

	ctx.d_face_adjacency = reinterpret_cast<index_t*>(mem.getMemory(cmesh.nnz * sizeof(index_t)));

	ctx.d_nextern = reinterpret_cast<index_t*>(mem.getMemory(sizeof(index_t)));
	succeed(cudaMemset(reinterpret_cast<void*>(ctx.d_nextern), 0, sizeof(index_t)));
	grid_dim = divup(cmesh.nnz, ctx.block_dim);
	prof.start(start, "Checking for boundary");
	CCKernels::checkForBoundaryFillFaceAdjacency << <grid_dim, ctx.block_dim >> >
		(ctx.d_mapping_buffer, cmesh.ids, cmesh.ptr, d_S_col_ptr, d_S_row_ids, d_epids_tmp, ctx.d_face_adjacency, ctx.d_nextern, cmesh.nnz);
	prof.stop(start, stop);

	succeed(cudaMemcpy(&ctx.nextern, ctx.d_nextern, sizeof(index_t), cudaMemcpyDeviceToHost));
	if (ctx.nextern != 0)
	{
		ctx.has_boundary = true;
		ctx.d_vextern = reinterpret_cast<index_t*>(mem.getMemory(ctx.nextern * sizeof(index_t)));
	}
	else
	{
		ctx.has_boundary = false;
		mem.freeMemory(ctx.d_nextern);
	}

	prof.start(start, "Scanning vids");
	CubFunctions::scan_exclusive(d_epids_tmp, d_S_epids_lt, static_cast<unsigned>(cmesh.nnz), mem);
	prof.stop(start, stop);

	if (ctx.has_boundary)
	{
		grid_dim = divup(cmesh.nnz, ctx.block_dim);
		prof.start(start, "Marking boundary boundary");
		CCKernels::markBoundary << <grid_dim, ctx.block_dim >> >
			(ctx.d_mapping_buffer, cmesh.ids, cmesh.ptr, d_S_col_ptr, d_S_row_ids, d_S_epids_lt, ctx.d_vextern, ctx.d_nextern, cmesh.nnz);
		prof.stop(start, stop);
	}

	mem.freeMemory(d_tmp_buffer);

	mem.takeOwnership(reinterpret_cast<const void*>(cmesh.ptr));
	mem.takeOwnership(reinterpret_cast<const void*>(cmesh.ids));

	succeed(cudaEventDestroy(start));
	succeed(cudaEventDestroy(stop));
}


////////////////////////////////////////////////////////////////////////////////
/// Publicly exposed
template<typename MESH_INFO>
void CatmullClark::subdivideIteratively(MESH_INFO const& cmesh, MESH_INFO& rmesh, int target_level)
{
	using offset_t = typename MESH_INFO::offset_t;
	using index_t = typename MESH_INFO::index_t;
	using vertex_t = typename MESH_INFO::vertex_t;

	int current_level = 0;

	ProfilinInfo<TimeProfileAccumulate<NoStateClock>, DeviceMemManager> profiling;
	Context<MESH_INFO> ctx;

	MESH_INFO tmp_cmesh = cmesh;

	if (tmp_cmesh.type != MESH_INFO::MeshType::QUAD)
	{
		initPolyMesh(tmp_cmesh, rmesh, ctx, profiling.mem, profiling.prof);
		subdivideTopologyPolyMesh(tmp_cmesh, rmesh, ctx, profiling.mem, profiling.prof);
		subdivideVertexDataPolyMesh(tmp_cmesh, rmesh, ctx, profiling.mem, profiling.prof);

		rmesh.type = MESH_INFO::MeshType::QUAD;
		rmesh.is_reduced = true;

		tmp_cmesh = rmesh;

		profiling.mem.freeAll();

		current_level++;
	}


	for (; current_level < target_level; ++current_level)
	{
		initQuadMesh(tmp_cmesh, rmesh, ctx, profiling.mem, profiling.prof);
		subdivideTopologyQuadMesh(tmp_cmesh, rmesh, ctx, profiling.mem, profiling.prof);
		subdivideVertexDataQuadMesh(tmp_cmesh, rmesh, ctx, profiling.mem, profiling.prof);

		rmesh.type = MESH_INFO::MeshType::QUAD;
		rmesh.is_reduced = true;

		if (current_level != 0) tmp_cmesh.freeAndReset();

		tmp_cmesh = rmesh;

		profiling.mem.freeAll();
	}

	std::cout << "==========SLAK==========\n";
	std::cout << "Subdivision to level " << target_level;
	std::cout << " took " << std::setprecision(2) << std::fixed << profiling.prof.time << " ms.";
	std::cout << " peak mem " << profiling.mem.peakConsumption() / (1000 * 1000) << " MB";
	std::cout << " \nCtrl. Mesh:";
	std::cout << "  nf: " << cmesh.nfaces;
	std::cout << "  nv: " << cmesh.nverts;
	std::cout << " \nSubd. Mesh:";
	std::cout << "  nf: " << rmesh.nfaces;
	std::cout << "  nv: " << rmesh.nverts;
	std::cout << "\n\n";
}


////////////////////////////////////////////////////////////////////////////////
/// Instantiations
using CCMeshInfo = CatmullClark::MeshInfo<int, int, int, float>;

template void CatmullClark::subdivideIteratively(CCMeshInfo const& cmesh, CCMeshInfo& rmesh, int target_level);
