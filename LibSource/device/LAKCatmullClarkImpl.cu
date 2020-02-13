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

#include "LAKCatmullClark.h"
#include "LAKCatmullClarkKernels.cuh"

#include <iostream>
#include <iomanip>
#include <cusparse.h>
#include "math/spgemm.cuh"
#include "math/spmv.cuh"


namespace
{
	template<typename T>
	void getCircMapQ(std::vector<T>& Q, unsigned size, unsigned pwr, unsigned index_base = 0)
	{
		Q.resize((size + index_base) * (size + index_base), 0);
		for (auto j = 1; j < size + 1; ++j)
		{
			for (auto i = 1; i < size + 1; ++i)
			{
				Q[(i - 1 + index_base) * (size + index_base) + (j - 1 + index_base)] = (j == ((i + pwr - 1) % size) + 1 ? 1 : 0);
			}
		}
	};

	template<typename T>
	std::vector<T> getFromDev(const T* d, size_t num)
	{
		std::vector<T> h(num);
		succeed(cudaMemcpy(&h[0], d, num * sizeof(T), cudaMemcpyDeviceToHost));
		return h;
	}
}

////////////////////////////////////////////////////////////////////////////////
/// Quadrilateral Mesh Subdiv
template<typename MESH_INFO, typename MEMORY_MANAGER, typename PROFILING>
void LAKCatmullClark::subdivideVertexDataQuadMesh(MESH_INFO const& cmesh, MESH_INFO& rmesh, Context<MESH_INFO>& ctx, MEMORY_MANAGER& mem, PROFILING& prof)
{
	using offset_t = typename MESH_INFO::offset_t;
	using index_t = typename MESH_INFO::index_t;
	using vertex_t = typename MESH_INFO::vertex_t;

	cudaEvent_t start, stop;
	succeed(cudaEventCreate(&start));
	succeed(cudaEventCreate(&stop));

	subdivideVertexDataPolyMesh(cmesh, rmesh, ctx, mem, prof);

	succeed(cudaEventDestroy(start));
	succeed(cudaEventDestroy(stop));
}

template<typename MESH_INFO, typename MEMORY_MANAGER, typename PROFILING>
void LAKCatmullClark::subdivideTopologyQuadMesh(MESH_INFO const& cmesh, MESH_INFO& rmesh, Context<MESH_INFO>& ctx, MEMORY_MANAGER& mem, PROFILING& prof)
{
	using offset_t = typename MESH_INFO::offset_t;
	using index_t = typename MESH_INFO::index_t;
	using value_t = typename MESH_INFO::value_t;

	cudaEvent_t start, stop;
	succeed(cudaEventCreate(&start));
	succeed(cudaEventCreate(&stop));

	mem.giveOwnership(cmesh.ptr, (cmesh.nfaces + 1) * sizeof(index_t));
	mem.giveOwnership(cmesh.ids, cmesh.nnz * sizeof(offset_t));
	mem.giveOwnership(cmesh.vals, cmesh.nnz * sizeof(value_t));

	rmesh.nfaces = cmesh.nnz;
	rmesh.nnz = rmesh.nfaces * 4;
	rmesh.nverts = cmesh.nverts + cmesh.nfaces + (cmesh.nnz - ctx.nextern) / 2 + ctx.nextern;
	rmesh.max_face_order = 4;

	size_t grid_dim = divup(cmesh.nfaces, ctx.block_dim);
	prof.start(start, "sorting row ids for topo refine");
	LAKHelperKernels::sortKeyValuePairsSegmentedInPlace(cmesh.ptr, cmesh.vals, cmesh.ids, cmesh.nfaces);
	prof.stop(start, stop);

	rmesh.ptr = reinterpret_cast<offset_t*>(mem.getMemory((rmesh.nfaces + 1) * sizeof(offset_t)));
	rmesh.ids = reinterpret_cast<offset_t*>(mem.getMemory(rmesh.nnz * sizeof(index_t)));
	rmesh.vals = reinterpret_cast<offset_t*>(mem.getMemory(rmesh.nnz * sizeof(value_t)));

	grid_dim = divup(cmesh.nnz, ctx.block_dim);
	prof.start(start, "Refining topology - universal");
	LAKCCKernels::refineTopologyHomogeneous << <grid_dim, ctx.block_dim >> > (
		cmesh.ids, ctx.d_E_ptr, ctx.d_E_ids, ctx.d_E_vals, rmesh.ids, cmesh.nnz, cmesh.nverts, cmesh.nfaces, cmesh.max_face_order);

	grid_dim = divup(rmesh.nfaces + 1, ctx.block_dim);
	LAKCCKernels::createQuadColPtr << <grid_dim, ctx.block_dim >> > (rmesh.ptr, rmesh.nfaces);

	grid_dim = divup(rmesh.nnz, ctx.block_dim);
	LAKCCKernels::createQuadVals << <grid_dim, ctx.block_dim >> > (rmesh.vals, rmesh.nfaces);
	prof.stop(start, stop);

	mem.takeOwnership(rmesh.ptr);
	mem.takeOwnership(rmesh.ids);
	mem.takeOwnership(rmesh.vals);

	mem.takeOwnership(cmesh.ptr);
	mem.takeOwnership(cmesh.ids);
	mem.takeOwnership(cmesh.vals);

	succeed(cudaEventDestroy(start));
	succeed(cudaEventDestroy(stop));
}

template<typename MESH_INFO, typename MEMORY_MANAGER, typename PROFILING>
void LAKCatmullClark::initQuadMesh(MESH_INFO const& cmesh, MESH_INFO& rmesh, Context<MESH_INFO>& ctx, MEMORY_MANAGER& mem, PROFILING& prof)
{
	using index_t = typename MESH_INFO::index_t;

	cudaEvent_t start, stop;
	succeed(cudaEventCreate(&start));
	succeed(cudaEventCreate(&stop));

	initPolyMesh(cmesh, rmesh, ctx, mem, prof);

	succeed(cudaEventDestroy(start));
	succeed(cudaEventDestroy(stop));
}


////////////////////////////////////////////////////////////////////////////////
/// Polygonal Mesh Subdiv
template<typename MESH_INFO, typename MEMORY_MANAGER, typename PROFILING>
void LAKCatmullClark::subdivideVertexDataPolyMesh(MESH_INFO const& cmesh, MESH_INFO& rmesh, Context<MESH_INFO>& ctx, MEMORY_MANAGER& mem, PROFILING& prof)
{
	using offset_t = typename MESH_INFO::offset_t;
	using index_t = typename MESH_INFO::index_t;
	using value_t = typename MESH_INFO::value_t;
	using vertex_t = typename MESH_INFO::vertex_t;
	constexpr int ncomponents{ 4 };

	cudaEvent_t start, stop;
	succeed(cudaEventCreate(&start));
	succeed(cudaEventCreate(&stop));

	mem.giveOwnership(cmesh.ptr, (cmesh.nfaces + 1) * sizeof(index_t));
	mem.giveOwnership(cmesh.ids, cmesh.nnz * sizeof(offset_t));
	mem.giveOwnership(cmesh.vals, cmesh.nnz * sizeof(value_t));
	mem.giveOwnership(cmesh.verts, cmesh.nverts * ncomponents * sizeof(vertex_t));

	rmesh.verts = reinterpret_cast<vertex_t*>(mem.getMemory(rmesh.nverts * ncomponents * sizeof(vertex_t)));

	//caclulate facepoints
	vertex_t* d_facepoints = reinterpret_cast<vertex_t*>(rmesh.verts + ncomponents * cmesh.nverts);
	size_t grid_dim = divup(cmesh.nfaces, ctx.block_dim);

	prof.start(start, ("calculating facepoints"));
	spmv_left_mapped_f4(cmesh.ptr, cmesh.ids, cmesh.vals,
		reinterpret_cast<const float4*>(cmesh.verts), reinterpret_cast<float4*>(d_facepoints), cmesh.nverts, cmesh.nfaces, ctx.d_map, cmesh.max_face_order + 1);

	LAKHelperKernels::divElementWise(d_facepoints, ctx.d_order_buffer + cmesh.nverts, vertex_t(0), d_facepoints, ncomponents * cmesh.nfaces);
	prof.stop(start, stop);

	//calculate edgepoints
	vertex_t* d_edgepoints = reinterpret_cast<vertex_t*>(rmesh.verts + (cmesh.nverts + cmesh.nfaces) * ncomponents);
	grid_dim = divup(size_t(ctx.nedges - ctx.nextern), ctx.block_dim);

	prof.start(start, ("calculating internal edgepoints"));
	LAKCCKernels::calculateInternalEdgepoints << <grid_dim, ctx.block_dim >> > (
		ctx.d_internal0, ctx.d_internal1, ctx.d_intids, ctx.d_f0, ctx.d_f1, cmesh.verts, d_facepoints, d_edgepoints, ctx.nedges - ctx.nextern);
	prof.stop(start, stop);


	if (ctx.nextern != 0)
	{
		grid_dim = divup(size_t(ctx.nextern), ctx.block_dim);
		prof.start(start, ("calculating external edgepoints"));
		LAKCCKernels::calculateExternalEdgepoints << <grid_dim, ctx.block_dim >> > (
			ctx.d_external0, ctx.d_external1, ctx.d_extids, cmesh.verts, d_edgepoints, ctx.nextern);
		prof.stop(start, stop);
	}

	//update original vertices
	vertex_t* d_p_norm = reinterpret_cast<vertex_t*>(mem.getMemory(cmesh.nverts * ncomponents * sizeof(vertex_t)));
	vertex_t* d_fp_vp_sum_norm = reinterpret_cast<vertex_t*>(mem.getMemory(cmesh.nverts * ncomponents * sizeof(vertex_t)));
	succeed(cudaMemset(d_fp_vp_sum_norm, 0, cmesh.nverts * ncomponents * sizeof(vertex_t)));


	grid_dim = divup(cmesh.nverts, ctx.block_dim);
	prof.start(start, "updating internal positions");
	LAKHelperKernels::multElementWise(cmesh.verts, ctx.d_order_buffer, vertex_t(-2), d_p_norm, ncomponents * cmesh.nverts);

	spmv_right_mapped_f4(cmesh.ptr, cmesh.ids, cmesh.vals,
		reinterpret_cast<float4*>(d_facepoints), reinterpret_cast<float4*>(d_fp_vp_sum_norm), cmesh.nverts, cmesh.nfaces, ctx.d_map, cmesh.max_face_order + 1);

	spmv_right_mapped_f4(ctx.d_F_ptr, ctx.d_F_ids, ctx.d_F_vals,
		reinterpret_cast<float4*>(cmesh.verts), reinterpret_cast<float4*>(d_fp_vp_sum_norm), cmesh.nverts, cmesh.nverts, ctx.d_F_map, cmesh.nfaces + 1);

	LAKHelperKernels::divElementWise(d_fp_vp_sum_norm, ctx.d_order_buffer, vertex_t(0), d_fp_vp_sum_norm, ncomponents * cmesh.nverts);

	LAKHelperKernels::addElementWise(d_fp_vp_sum_norm, d_p_norm, rmesh.verts, ncomponents * cmesh.nverts);

	LAKHelperKernels::divElementWise(rmesh.verts, ctx.d_order_buffer, vertex_t(0), rmesh.verts, ncomponents * cmesh.nverts);
	prof.stop(start, stop);

	if (ctx.nextern)
	{

		grid_dim = divup(size_t(ctx.nextern), ctx.block_dim);
		prof.start(start, "updating external positions");
		LAKCCKernels::prepareExternalVertexUpdate << <grid_dim, ctx.block_dim >> > (
			ctx.d_external0, ctx.d_external1, cmesh.verts, rmesh.verts, ctx.nextern);

		LAKCCKernels::calculateExternalVertexUpdate << <grid_dim, ctx.block_dim >> > (
			ctx.d_external0, ctx.d_external1, cmesh.verts, rmesh.verts, ctx.nextern);
		prof.stop(start, stop);

	}

	mem.freeMemory(d_p_norm);
	mem.freeMemory(d_fp_vp_sum_norm);

	//TODO: re-implement aka. port
	//if (has_creases && !creases_decayed)
	//	time += handleCreases(C, d_vertex_data, d_refined_vertexdata, nf, nv);

	mem.takeOwnership(rmesh.verts);

	mem.takeOwnership(cmesh.ptr);
	mem.takeOwnership(cmesh.ids);
	mem.takeOwnership(cmesh.vals);
	mem.takeOwnership(cmesh.verts);

	succeed(cudaEventDestroy(start));
	succeed(cudaEventDestroy(stop));
}

template<typename MESH_INFO, typename MEMORY_MANAGER, typename PROFILING>
void LAKCatmullClark::subdivideTopologyPolyMesh(MESH_INFO const& cmesh, MESH_INFO& rmesh, Context<MESH_INFO>& ctx, MEMORY_MANAGER& mem, PROFILING& prof)
{
	using offset_t = typename MESH_INFO::offset_t;
	using index_t = typename MESH_INFO::index_t;
	using value_t = typename MESH_INFO::value_t;

	cudaEvent_t start, stop;
	succeed(cudaEventCreate(&start));
	succeed(cudaEventCreate(&stop));

	mem.giveOwnership(cmesh.ptr, (cmesh.nfaces + 1) * sizeof(index_t));
	mem.giveOwnership(cmesh.ids, cmesh.nnz * sizeof(offset_t));
	mem.giveOwnership(cmesh.vals, cmesh.nnz * sizeof(value_t));

	rmesh.nfaces = cmesh.nnz;
	rmesh.nnz = rmesh.nfaces * 4;
	rmesh.nverts = cmesh.nverts + cmesh.nfaces + (cmesh.nnz - ctx.nextern) / 2 + ctx.nextern;
	rmesh.max_face_order = 4;

	size_t grid_dim = divup(cmesh.nfaces, ctx.block_dim);
	prof.start(start, "sorting row ids for topo refine");
	LAKHelperKernels::sortKeyValuePairsSegmentedInPlace(cmesh.ptr, cmesh.vals, cmesh.ids, cmesh.nfaces);
	prof.stop(start, stop);

	rmesh.ptr = reinterpret_cast<offset_t*>(mem.getMemory((rmesh.nfaces + 1) * sizeof(offset_t)));
	rmesh.ids = reinterpret_cast<offset_t*>(mem.getMemory(rmesh.nnz * sizeof(index_t)));
	rmesh.vals = reinterpret_cast<offset_t*>(mem.getMemory(rmesh.nnz * sizeof(value_t)));

	grid_dim = divup(cmesh.nfaces, ctx.block_dim);
	prof.start(start, "Refining topology - universal");
	LAKCCKernels::refineTopology << <grid_dim, ctx.block_dim >> > (
		cmesh.ptr, cmesh.ids,
		ctx.d_E_ptr, ctx.d_E_ids, ctx.d_E_vals, rmesh.ids, cmesh.nfaces, cmesh.nverts);

	grid_dim = divup(rmesh.nfaces + 1, ctx.block_dim);
	LAKCCKernels::createQuadColPtr << <grid_dim, ctx.block_dim >> > (rmesh.ptr, rmesh.nfaces);

	grid_dim = divup(rmesh.nnz, ctx.block_dim);
	LAKCCKernels::createQuadVals << <grid_dim, ctx.block_dim >> > (rmesh.vals, rmesh.nfaces);
	prof.stop(start, stop);

	mem.takeOwnership(rmesh.ptr);
	mem.takeOwnership(rmesh.ids);
	mem.takeOwnership(rmesh.vals);

	mem.takeOwnership(cmesh.ptr);
	mem.takeOwnership(cmesh.ids);
	mem.takeOwnership(cmesh.vals);

	succeed(cudaEventDestroy(start));
	succeed(cudaEventDestroy(stop));
}

template<typename MESH_INFO, typename MEMORY_MANAGER, typename PROFILING>
void LAKCatmullClark::initPolyMesh(MESH_INFO const& cmesh, MESH_INFO& rmesh, Context<MESH_INFO>& ctx, MEMORY_MANAGER& mem, PROFILING& prof)
{
	using offset_t = typename MESH_INFO::offset_t;
	using index_t = typename MESH_INFO::index_t;
	using value_t = typename MESH_INFO::value_t;

	cudaEvent_t start, stop;
	succeed(cudaEventCreate(&start));
	succeed(cudaEventCreate(&stop));

	mem.giveOwnership(cmesh.ptr, (cmesh.nfaces + 1) * sizeof(index_t));
	mem.giveOwnership(cmesh.ids, cmesh.nnz * sizeof(offset_t));
	mem.giveOwnership(cmesh.vals, cmesh.nnz * sizeof(value_t));

	size_t grid_dim = divup(cmesh.nfaces, ctx.block_dim);
	prof.start(start, "sorting row ids");
	LAKHelperKernels::sortKeyValuePairsSegmentedInPlace(cmesh.ptr, cmesh.ids, cmesh.vals, cmesh.nfaces);
	prof.stop(start, stop);


	//create map for spmv (+1 because values start at 1)
	std::vector<value_t> spmv_map(cmesh.max_face_order + 1, static_cast<value_t>(1));
	ctx.d_map = reinterpret_cast<value_t*>(mem.getMemory((cmesh.max_face_order + 1) * sizeof(value_t)));
	succeed(cudaMemcpy(ctx.d_map, &spmv_map[0], (cmesh.max_face_order + 1) * sizeof(value_t), cudaMemcpyHostToDevice));

	//create map for F*p (+1 because values start at 1)
	std::vector<value_t> F_map(cmesh.nfaces + 1, static_cast<value_t>(1));
	ctx.d_F_map = reinterpret_cast<value_t*>(mem.getMemory((cmesh.nfaces + 1) * sizeof(value_t)));
	succeed(cudaMemcpy(ctx.d_F_map, &F_map[0], (cmesh.nfaces + 1) * sizeof(value_t), cudaMemcpyHostToDevice));

	std::vector<value_t> nf_ones(cmesh.nfaces, 1.0f);
	value_t* d_nf_ones = reinterpret_cast<value_t*>(mem.getMemory(cmesh.nfaces * sizeof(value_t)));
	succeed(cudaMemcpy(d_nf_ones, &nf_ones[0], cmesh.nfaces * sizeof(value_t), cudaMemcpyHostToDevice));

	ctx.d_order_buffer = reinterpret_cast<value_t*>(mem.getMemory((cmesh.nverts + cmesh.nfaces) * sizeof(value_t)));
	succeed(cudaMemset(ctx.d_order_buffer, 0, cmesh.nverts * sizeof(value_t)));

	prof.start(start, ("calculating vertex orders"));
	spmv_right_mapped(
		cmesh.ptr,
		cmesh.ids,
		cmesh.vals, d_nf_ones, ctx.d_order_buffer,
		cmesh.nverts,
		cmesh.nfaces,
		ctx.d_map,
		cmesh.max_face_order + 1);
	prof.stop(start, stop);

	mem.freeMemory(d_nf_ones);

	//face order
	std::vector<value_t> nv_ones(cmesh.nverts, 1.0f);
	value_t* d_nv_ones = reinterpret_cast<value_t*>(mem.getMemory(cmesh.nverts * sizeof(value_t)));
	succeed(cudaMemcpy(d_nv_ones, &nv_ones[0], cmesh.nverts * sizeof(value_t), cudaMemcpyHostToDevice));

	value_t* d_faceorders = ctx.d_order_buffer + cmesh.nverts;

	prof.start(start, ("calculating face orders"));
	spmv_left_mapped(cmesh.ptr, cmesh.ids, cmesh.vals, d_nv_ones, d_faceorders, cmesh.nverts, cmesh.nfaces, ctx.d_map, cmesh.max_face_order + 1);
	prof.stop(start, stop);

	mem.freeMemory(d_nv_ones);

	mem.registerConsumption((cmesh.nverts + 1) * sizeof(unsigned));
	mem.registerConsumption(cmesh.nnz * sizeof(unsigned));
	mem.registerConsumption(cmesh.nnz * sizeof(float));

	offset_t* d_ptr_t = reinterpret_cast<offset_t*>(mem.getMemory((cmesh.nverts + 1) * sizeof(offset_t)));
	index_t* d_ids_t = reinterpret_cast<index_t*>(mem.getMemory(cmesh.nnz * sizeof(index_t)));
	value_t* d_vals_t = reinterpret_cast<value_t*>(mem.getMemory(cmesh.nnz * sizeof(value_t)));

	cusparseHandle_t handle;
	cuSparseSucceed(cusparseCreate(&handle));

	prof.start(start, "transposing M");
	cuSparseSucceed(cusparseScsr2csc(handle,
		cmesh.nfaces, cmesh.nverts, cmesh.nnz,
		reinterpret_cast<const float*>(cmesh.vals), cmesh.ptr, cmesh.ids,
		reinterpret_cast<float*>(d_vals_t), d_ids_t, d_ptr_t,
		CUSPARSE_ACTION_NUMERIC,
		CUSPARSE_INDEX_BASE_ZERO));
	prof.stop(start, stop);

	//This would be the non-deprecated version... doesn't work
	//size_t buffer_size{ 42 };
	//prof.start(start, "transposing M 1/2");
	//cuSparseSucceed(cusparseCsr2cscEx2_bufferSize(
	//	handle,
	//	cmesh.nfaces,
	//	cmesh.nverts,
	//	cmesh.nnz,
	//	cmesh.vals,
	//	cmesh.ptr,
	//	cmesh.ids,
	//	d_vals_t,
	//	d_ptr_t,
	//	d_ids_t,
	//	CUDA_R_32I,
	//	CUSPARSE_ACTION_SYMBOLIC,
	//	CUSPARSE_INDEX_BASE_ZERO,
	//	CUSPARSE_CSR2CSC_ALG1,
	//	&buffer_size));
	//prof.stop(start, stop);

	//void* buffer = mem.getMemory(buffer_size);
	//prof.start(start, "transposing M 2/2");
	//cuSparseSucceed(cusparseCsr2cscEx2(handle,
	//	cmesh.nfaces,
	//	cmesh.nverts,
	//	cmesh.nnz,
	//	cmesh.vals,
	//	cmesh.ptr,
	//	cmesh.ids,
	//	d_vals_t,
	//	d_ptr_t,
	//	d_ids_t,
	//	CUDA_R_32I,
	//	CUSPARSE_ACTION_NUMERIC,
	//	CUSPARSE_INDEX_BASE_ZERO,
	//	CUSPARSE_CSR2CSC_ALG1,
	//	buffer));
	//prof.stop(start, stop);
	//mem.freeMemory(buffer);


	std::vector<value_t> map;
	getCircMapQ(map, cmesh.max_face_order, 1, 1); // Q_{cmesh.max_face_order}

	ctx.d_map0 = reinterpret_cast<value_t*>(mem.getMemory(map.size() * sizeof(value_t)));
	succeed(cudaMemcpy(ctx.d_map0, &map[0], map.size() * sizeof(value_t), cudaMemcpyHostToDevice));

	getCircMapQ(map, cmesh.max_face_order, cmesh.max_face_order - 1, 1); // // Q_{cmesh.max_face_order}^{cmesh.max_face_order-1}
	ctx.d_map1 = reinterpret_cast<value_t*>(mem.getMemory(map.size() * sizeof(value_t)));
	succeed(cudaMemcpy(ctx.d_map1, &map[0], map.size() * sizeof(value_t), cudaMemcpyHostToDevice));

	offset_t* d_F_ptr_t;
	index_t* d_F_ids_t;
	value_t* d_F_vals_t;
	size_t f_nnz, f_nnz_t, e_nnz;

	prof.time += spgemm_mapped<offset_t, index_t, value_t, MEMORY_MANAGER>(cmesh.ptr, cmesh.ids, cmesh.vals,
		d_ptr_t, d_ids_t, d_vals_t,
		ctx.d_F_ptr, ctx.d_F_ids, ctx.d_F_vals,
		d_F_ptr_t, d_F_ids_t, d_F_vals_t,
		ctx.d_E_ptr, ctx.d_E_ids, ctx.d_E_vals,
		cmesh.nverts, cmesh.nfaces, cmesh.nverts, cmesh.nnz, cmesh.nnz,
		f_nnz, f_nnz_t, e_nnz, ctx.nextern,
		ctx.d_map0, cmesh.max_face_order + 1, cmesh.max_face_order + 1,
		ctx.d_map1, cmesh.max_face_order + 1, cmesh.max_face_order + 1, mem);

	mem.freeMemory(d_ptr_t);
	mem.freeMemory(d_ids_t);
	mem.freeMemory(d_vals_t);

	mem.unregisterConsumption((cmesh.nverts + 1) * sizeof(offset_t));
	mem.unregisterConsumption(cmesh.nnz * sizeof(index_t));
	mem.unregisterConsumption(cmesh.nnz * sizeof(value_t));
	

	prof.start(start, "compressing F F^T and E");
	LAKHelperKernels::compress(ctx.d_F_ptr, ctx.d_F_ids, ctx.d_F_vals, cmesh.nverts, cmesh.nverts, f_nnz, mem);
	LAKHelperKernels::compress(d_F_ptr_t, d_F_ids_t, d_F_vals_t, cmesh.nverts, cmesh.nverts, f_nnz_t, mem);
	LAKHelperKernels::compress(ctx.d_E_ptr, ctx.d_E_ids, ctx.d_E_vals, cmesh.nverts, cmesh.nverts, e_nnz, mem);
	prof.stop(start, stop);

	ctx.nedges = e_nnz;
	auto nintern = ctx.nedges - ctx.nextern;

	ctx.d_internal0 = reinterpret_cast<index_t*>(mem.getMemory(nintern * sizeof(index_t)));
	ctx.d_internal1 = reinterpret_cast<index_t*>(mem.getMemory(nintern * sizeof(index_t)));
	ctx.d_intids = reinterpret_cast<index_t*>(mem.getMemory(nintern * sizeof(index_t)));

	if (ctx.nextern)
	{
		ctx.d_external0 = reinterpret_cast<index_t*>(mem.getMemory(ctx.nextern * sizeof(unsigned)));
		ctx.d_external1 = reinterpret_cast<index_t*>(mem.getMemory(ctx.nextern * sizeof(unsigned)));
		ctx.d_extids = reinterpret_cast<index_t*>(mem.getMemory(ctx.nextern * sizeof(unsigned)));
	}

	index_t* d_nintext = reinterpret_cast<index_t*>(mem.getMemory(2 * sizeof(unsigned)));
	succeed(cudaMemset((void*)d_nintext, 0, 2 * sizeof(unsigned)));
	grid_dim = divup(cmesh.nverts, ctx.block_dim);
	prof.start(start, "getting Edge info from E");
	LAKCCKernels::extractEdgeInfoFromE << <grid_dim, ctx.block_dim >> > (
		ctx.d_E_ptr, ctx.d_E_ids, ctx.d_E_vals, cmesh.nverts,
		ctx.d_internal0, ctx.d_internal1, ctx.d_intids, ctx.d_external0, ctx.d_external1, ctx.d_extids, d_nintext);
	prof.stop(start, stop);

	ctx.d_f0 = reinterpret_cast<index_t*>(mem.getMemory(nintern * sizeof(unsigned)));
	ctx.d_f1 = reinterpret_cast<index_t*>(mem.getMemory(nintern * sizeof(unsigned)));
	grid_dim = divup(size_t(nintern), ctx.block_dim);
	prof.start(start, "getting Face info from F and Ft");
	LAKCCKernels::extractFaceInfoFromFFt << <grid_dim, ctx.block_dim >> > (
		ctx.d_F_ptr, ctx.d_F_ids, ctx.d_F_vals, d_F_ptr_t, d_F_ids_t, d_F_vals_t,
		ctx.d_internal0, ctx.d_internal1, nintern, ctx.d_f0, ctx.d_f1);
	prof.stop(start, stop);

	mem.freeMemory(d_F_ptr_t);
	mem.freeMemory(d_F_ids_t);
	mem.freeMemory(d_F_vals_t);

	mem.takeOwnership(cmesh.ptr);
	mem.takeOwnership(cmesh.ids);
	mem.takeOwnership(cmesh.vals);

	succeed(cudaEventDestroy(start));
	succeed(cudaEventDestroy(stop));
}


////////////////////////////////////////////////////////////////////////////////
/// Publicly exposed
template<typename MESH_INFO>
void LAKCatmullClark::subdivideIteratively(MESH_INFO const& cmesh, MESH_INFO& rmesh, int target_level)
{
	using offset_t = typename MESH_INFO::offset_t;
	using index_t = typename MESH_INFO::index_t;
	using value_t = typename MESH_INFO::value_t;
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
		rmesh.is_reduced = false;
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
		rmesh.is_reduced = false;

		if (current_level != 0) tmp_cmesh.freeAndReset();

		tmp_cmesh = rmesh;

		profiling.mem.freeAll();
	}


	std::cout << "==========LAK===========\n";
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
using LAKCCMeshInfo = LAKCatmullClark::MeshInfo<int, int, int, float>;

template void LAKCatmullClark::subdivideIteratively(LAKCCMeshInfo const&, LAKCCMeshInfo&, int);

