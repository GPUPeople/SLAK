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

#include <cuda_runtime_api.h>
#include "cuda_host_helpers.h"

#include <numeric>
#include <algorithm>
#include <vector>
#include <stdexcept>


#include "obj.h"

namespace CPU
{
	namespace Helper
	{
		// .*_permutation_.* source:
		// https://stackoverflow.com/questions/17074324/
		// last visited 13.02.2020
		template <typename T, typename Compare>
		std::vector<std::size_t> sort_permutation(const std::vector<T>& vec, Compare& compare,
			int begin, int end)
		{
			std::vector<std::size_t> p(end - begin);
			std::iota(p.begin(), p.end(), 0);
			std::sort(p.begin(), p.end(),
				[&](std::size_t i, std::size_t j) { return compare(vec[i + begin], vec[j + begin]); });
			return p;
		}

		template <typename T>
		std::vector<T> apply_permutation(const std::vector<T>& vec, const std::vector<std::size_t>& p,
			int begin, int end)
		{
			std::vector<T> sorted_vec = vec;
			std::transform(p.begin(), p.end(), sorted_vec.begin() + begin,
				[&](std::size_t i) { return vec[i]; });
			return sorted_vec;
		}

		template <typename T>
		void apply_permutation_in_place(std::vector<T>& vec, const std::vector<std::size_t>& p,
			int begin, int end)
		{
			std::vector<bool> done(end - begin, false);
			for (std::size_t i = 0; i < done.size(); ++i)
			{
				if (done[i])
					continue;

				done[i] = true;
				std::size_t prev_j = i;
				std::size_t j = p[i];
				while (i != j)
				{
					std::swap(vec[begin + prev_j], vec[begin + j]);
					done[j] = true;
					prev_j = j;
					j = p[j];
				}
			}
		}
	}
}


namespace GPU
{

	template<
		template<typename, typename, typename, typename>class FROM_TYPE,
		template<typename, typename, typename, typename>class TO_TYPE,
		typename OFFSET_TYPE, typename INDEX_TYPE, typename VALUE_TYPE, typename VERTEX_TYPE>
		static void CSCMeshCCMeshInfoConversion(
			const FROM_TYPE<OFFSET_TYPE, INDEX_TYPE, VALUE_TYPE, VERTEX_TYPE>& in,
			TO_TYPE<OFFSET_TYPE, INDEX_TYPE, VALUE_TYPE, VERTEX_TYPE>& out)
	{
		using InType = FROM_TYPE<OFFSET_TYPE, INDEX_TYPE, VALUE_TYPE, VERTEX_TYPE>;
		using OutType = TO_TYPE<OFFSET_TYPE, INDEX_TYPE, VALUE_TYPE, VERTEX_TYPE>;

		out.nverts = in.nverts;
		out.nfaces = in.nfaces;
		out.nnz = in.nnz;
		out.max_face_order = in.max_face_order;

		out.ptr = in.ptr;
		out.ids = in.ids;
		out.vals = in.vals;

		out.verts = in.verts;

		if (in.type == InType::MeshType::QUAD)
			out.type = OutType::MeshType::QUAD;
		else if (in.type == InType::MeshType::TRI)
			out.type = OutType::MeshType::TRI;
		else
			out.type = OutType::MeshType::POLY;

		out.is_reduced = in.is_reduced;
	}

	template<typename OFFSET_TYPE = int, typename INDEX_TYPE = int, typename VALUE_TYPE = int, typename VERTEX_TYPE = float>
	struct CSCMesh
	{
		using offset_t = OFFSET_TYPE;
		using index_t = INDEX_TYPE;
		using value_t = VALUE_TYPE;
		using vertex_t = VERTEX_TYPE;

		using MeshType = OBJ::Data::FaceType;
		MeshType type{ MeshType::POLY };


		size_t nfaces{ 0 };
		size_t nverts{ 0 };
		size_t nnz{ 0 };
		size_t max_face_order{ 0 };

		bool is_reduced{ false };
		offset_t* ptr{ nullptr };
		index_t* ids{ nullptr };
		value_t* vals{ nullptr };
		vertex_t* verts{ nullptr };

		CSCMesh() = default;

		explicit CSCMesh(const std::string& path)
		{
			OBJ::Data obj;
			OBJ::read(obj, path.c_str());

			type = obj.type;

			nnz = obj.f_vi.size();
			nfaces = obj.f_offs.size() - 1;
			nverts = obj.v.size();
			max_face_order = obj.max_face_order;

			is_reduced = true;

			std::vector<value_t> h_vals(nnz);
			for (size_t c = 0; c < nfaces; ++c)
				for (offset_t off = obj.f_offs[c]; off < obj.f_offs[c + 1]; ++off)
					h_vals[off] = off - obj.f_offs[c] + 1;


			std::vector<math::float4> h_verts(nverts);
			for (size_t i = 0; i < nverts; ++i)
				h_verts[i] = math::float4(obj.v[i], 1.0f);

			setFromCPU(obj.f_offs, obj.f_vi, h_vals, h_verts);
		}

		CSCMesh(const CSCMesh& rhs)
		{
			alloc(rhs.nfaces, rhs.nnz);
			copy(*this, rhs);
		}

		~CSCMesh()
		{
			try
			{
				free();
			}
			catch (...)
			{
			}
			
		}

		CSCMesh& operator=(const CSCMesh& rhs)
		{
			if (&rhs == this)
				return *this;

			free();
			CSCMesh::copy(*this, rhs);

			return *this;
		}

		void alloc(size_t nfaces, size_t nverts, size_t nnz)
		{
			succeed(cudaMalloc(reinterpret_cast<void**>(&ptr), (nfaces + 1) * sizeof(offset_t)));
			succeed(cudaMalloc(reinterpret_cast<void**>(&ids), nnz * sizeof(index_t)));
			succeed(cudaMalloc(reinterpret_cast<void**>(&vals), nnz * sizeof(value_t)));
			succeed(cudaMalloc(reinterpret_cast<void**>(&verts), nverts * 4 * sizeof(vertex_t)));
		}

		void free()
		{
			if (ptr != nullptr)
				succeed(cudaFree(ptr));

			if (ids != nullptr)
				succeed(cudaFree(ids));

			if (vals != nullptr)
				succeed(cudaFree(vals));

			if (verts != nullptr)
				succeed(cudaFree(verts));

			ptr = nullptr, ids = nullptr, vals = nullptr, verts = nullptr;

			nfaces = 0, nverts = 0, nnz = 0;
		}

		static void copy(CSCMesh& dst, const CSCMesh& src)
		{
			if (&dst == &src)
				return;

			dst.nfaces = src.nfaces, dst.nverts = src.nverts, dst.nnz = src.nnz; dst.max_face_order = src.max_face_order;

			if (dst.ptr != nullptr && src.ptr != nullptr)
				succeed(cudaMemcpy(dst.ptr, src.ptr, (src.nfaces + 1) * sizeof(offset_t), cudaMemcpyDeviceToDevice));

			if (dst.ids != nullptr && src.ids != nullptr)
				succeed(cudaMemcpy(dst.ids, src.ids, src.nnz * sizeof(index_t), cudaMemcpyDeviceToDevice));

			if (dst.vals != nullptr && src.vals != nullptr)
				succeed(cudaMemcpy(dst.vals, src.vals, src.nnz * sizeof(value_t), cudaMemcpyDeviceToDevice));

			if (dst.verts != nullptr && src.verts != nullptr)
				succeed(cudaMemcpy(dst.verts, src.verts, src.nverts * 4 * sizeof(vertex_t), cudaMemcpyDeviceToDevice));
		}

		void accessFromCpu(std::vector<offset_t>& cpu_ptr, std::vector<index_t>& cpu_ids, std::vector<value_t>& cpu_vals, std::vector<math::float4>& cpu_verts) const
		{
			if (this->ptr != nullptr)
			{
				cpu_ptr.resize(nfaces + 1);
				succeed(cudaMemcpy(&cpu_ptr[0], this->ptr, (nfaces + 1) * sizeof(offset_t), cudaMemcpyDeviceToHost));
			}
			if (this->ids != nullptr)
			{
				cpu_ids.resize(nnz);
				succeed(cudaMemcpy(&cpu_ids[0], this->ids, nnz * sizeof(index_t), cudaMemcpyDeviceToHost));
			}
			if (this->vals != nullptr)
			{
				cpu_vals.resize(nnz);
				succeed(cudaMemcpy(&cpu_vals[0], this->vals, nnz * sizeof(value_t), cudaMemcpyDeviceToHost));
			}
			if (this->verts != nullptr)
			{
				cpu_verts.resize(nverts);
				succeed(cudaMemcpy(&cpu_verts[0], verts, nverts * sizeof(math::float4), cudaMemcpyDeviceToHost));
			}
		}

		void accessFromCpu(std::vector<offset_t>& cpu_ptr, std::vector<index_t>& cpu_ids, std::vector<value_t>& cpu_vals, std::vector<math::float3>& cpu_verts) const
		{
			std::vector<math::float4> cpu_verts_f4;
			accessFromCpu(cpu_ptr, cpu_ids, cpu_vals, cpu_verts_f4);
			cpu_verts.reserve(cpu_verts_f4.size());
			for (const auto& vert : cpu_verts_f4)
				cpu_verts.emplace_back(vert.xyz());
		}

		void setFromCPU(std::vector<offset_t>& cpu_ptr, std::vector<index_t>& cpu_ids, std::vector<value_t>& cpu_vals, std::vector<math::float4>& cpu_verts)
		{
			free();

			nfaces = cpu_ptr.size() - 1;
			nverts = *std::max_element(cpu_ids.begin(), cpu_ids.end()) + 1;
			nnz = cpu_ids.size();

			alloc(nfaces, nverts, nnz);

			succeed(cudaMemcpy(this->ptr, &cpu_ptr[0], (nfaces + 1) * sizeof(offset_t), cudaMemcpyHostToDevice));
			succeed(cudaMemcpy(this->ids, &cpu_ids[0], nnz * sizeof(index_t), cudaMemcpyHostToDevice));
			succeed(cudaMemcpy(this->vals, &cpu_vals[0], nnz * sizeof(value_t), cudaMemcpyHostToDevice));
			succeed(cudaMemcpy(this->verts, &cpu_verts[0], nverts * sizeof(math::float4), cudaMemcpyHostToDevice));
		}

		void reduce()
		{
			if (type == MeshType::POLY)
				throw std::runtime_error("mesh type not homogeneous");

			if (is_reduced)
				return;

			std::vector<offset_t> cpu_ptr;
			std::vector<index_t> cpu_ids;
			std::vector<value_t> cpu_vals;
			std::vector<math::float4> cpu_verts;

			accessFromCpu(cpu_ptr, cpu_ids, cpu_vals, cpu_verts);

			// This sorts the row ids and values according to the values
			auto const less = [](auto const& a, auto const& b) { return a < b; };
			for (size_t c = 0; c < nfaces; ++c)
			{
				const std::pair<int, int> range(cpu_ptr[c], cpu_ptr[c + 1]);
				const auto perm = CPU::Helper::sort_permutation(cpu_vals, less, range.first, range.second);
				CPU::Helper::apply_permutation_in_place(cpu_ids, perm, range.first, range.second);
				CPU::Helper::apply_permutation_in_place(cpu_vals, perm, range.first, range.second);
			}

			setFromCPU(cpu_ptr, cpu_ids, cpu_vals, cpu_verts);

			is_reduced = true;
		}

		void unreduce()
		{
			if (type == MeshType::POLY)
				throw std::runtime_error("mesh type not homogeneous");

			if (!is_reduced)
				return;

			std::vector<offset_t> cpu_ptr;
			std::vector<index_t> cpu_ids;
			std::vector<value_t> cpu_vals;
			std::vector<math::float4> cpu_verts;

			accessFromCpu(cpu_ptr, cpu_ids, cpu_vals, cpu_verts);

			if (cpu_ptr.empty())
			{
				cpu_ptr.resize(nfaces + 1);
				int nvface = type == MeshType::QUAD ? 4 : 3;
				std::generate(cpu_ptr.begin(), cpu_ptr.end(), [nvface, c = -nvface]() mutable { return c += nvface; });
			}

			if (cpu_vals.empty())
			{
				cpu_vals.resize(nnz);
				int nvface = type == MeshType::QUAD ? 4 : 3;
				std::generate(cpu_vals.begin(), cpu_vals.end(), [nvface, c = 0]() mutable { return (c++ - 1) % nvface; });
			}

			// This sorts the row ids and values according to the row ids
			auto const less = [](auto const& a, auto const& b) { return a < b; };
			for (size_t c = 0; c < nfaces; ++c)
			{
				const std::pair<int, int> range(cpu_ptr[c], cpu_ptr[c + 1]);
				const auto perm = CPU::Helper::sort_permutation(cpu_ids, less, range.first, range.second);
				CPU::Helper::apply_permutation_in_place(cpu_ids, perm, range.first, range.second);
				CPU::Helper::apply_permutation_in_place(cpu_vals, perm, range.first, range.second);
			}

			setFromCPU(cpu_ptr, cpu_ids, cpu_vals, cpu_verts);

			is_reduced = false;
		}

		void toObj(const std::string& fname)
		{
			OBJ::Data data;
			data.type = this->type;

			const bool was_reduced = is_reduced;
			if (!is_reduced)
				reduce();

			if (data.type == MeshType::POLY && this->ptr != nullptr)
			{
				data.f_offs.resize(nfaces + 1);
				succeed(cudaMemcpy(&data.f_offs[0], this->ptr, (nfaces + 1) * sizeof(offset_t), cudaMemcpyDeviceToHost));
			}
			if (this->ids != nullptr)
			{
				data.f_vi.resize(nnz);
				succeed(cudaMemcpy(&data.f_vi[0], this->ids, nnz * sizeof(index_t), cudaMemcpyDeviceToHost));
			}
			if (this->verts != nullptr)
			{
				std::vector<math::float4> cpu_verts(nverts);
				succeed(cudaMemcpy(&cpu_verts[0], verts, nverts * sizeof(math::float4), cudaMemcpyDeviceToHost));

				data.v.reserve(cpu_verts.size());
				for (const auto& vert : cpu_verts)
					data.v.emplace_back(vert.xyz());
			}

			OBJ::write(data, fname.c_str());

			if (!was_reduced)
				unreduce();
		}
	};


}
