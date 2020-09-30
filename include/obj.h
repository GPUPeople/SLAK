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


#include <limits>
#include <cctype>
#include <algorithm>
#include <string>
#include <streambuf>
#include <sstream>
#include <fstream>
#include <iostream>
#include <stdexcept>

#include <unordered_map>

#include "math/vector.h"

namespace
{
	inline void printVertexPos(std::ofstream& out, const math::float3& v)
	{
		out << "v " << v.x << " " << v.y << " " << v.z << "\n";
	}

	inline void printVertexNormal(std::ofstream& out, const math::float3& vn)
	{
		out << "vn " << vn.x << " " << vn.y << " " << vn.z << "\n";
	}

	inline void printTexcoord(std::ofstream& out, const math::float2& vt)
	{
		out << "vt " << vt.x << " " << vt.y << "\n";
	}

	inline void printFace(std::ofstream& out, int const* first, const int fsize)
	{
		out << "f";
		for (int off = 0; off < fsize; ++off)
		{
			out << " " << *(first + off) + 1;
		}
		out << "\n";
	}

}

namespace OBJ
{

	struct Data
	{
		enum class FaceType { TRI, QUAD, POLY };
		FaceType type{ FaceType::POLY };
		int max_face_order{ -1 };

		std::vector<int> f_offs;

		std::vector<int> f_vi;
		std::vector<math::float3> v;

		std::vector<int> f_vni;
		std::vector<math::float3> vn;

		std::vector<int> f_vti;
		std::vector<math::float2> vt;

		struct Tag
		{
			std::string type;
			std::vector<int> intargs;
			std::vector<float> floatargs;
		};

		std::vector<Tag> t;
	};


	////////////////////////////////////////////////////////////////////////////////
	/// Read functionality


	inline void read(Data& data, const char* fname)
	{
		std::ifstream in_file(fname);
		if (!in_file.is_open())
			throw std::runtime_error("Could not open obj file\n");

		data.f_offs.push_back(0);
		
		int ln_nbr = 0;
		std::string in_line;
		while (std::getline(in_file, in_line))
		{
			++ln_nbr;

			if (in_line.empty())
				continue;
			
			std::stringstream in(in_line);
			std::string id;
			in >> id;

			if (id == "#")
				continue;

			if (id == "g")
				continue;

			if (id == "v")
			{

				math::float3 p;
				in >> p.x >> p.y >> p.z;
				data.v.push_back(p);
				continue;

			}

			if (id == "vn")
			{
				math::float3 n;
				in >> n.x >> n.y >> n.z;
				n.z = -n.z;
				data.vn.push_back(n);
				continue;
			}

			if (id == "vt")
			{
				math::float2 t;
				in >> t.x >> t.y;
				data.vt.push_back(t);
				continue;
			}

			if (id == "f")
			{
				int nverts = 0;
				int vi{ -1 };
				while (in >> vi && !in.fail())
				{
					data.f_vi.push_back(vi < 0 ? vi + static_cast<int>(data.v.size()) : vi - 1);
					vi = -1;
					
					if (in.peek() == '/')
					{
						in.get();

						int ti{ -1 };
						in >> ti;

						if (in.fail())
							in.clear();
						else
							data.f_vti.push_back(ti < 0 ? ti + static_cast<int>(data.vt.size()) : ti - 1);

						if (in.peek() == '/')
						{
							in.get();

							int ni{ -1 };
							in >> ni;

							if (in.fail())
								in.clear();
							else
								data.f_vni.push_back(ni < 0 ? ni + static_cast<int>(data.vn.size()) : ni - 1);
						}
					}

					++nverts;
				}
				data.f_offs.push_back(data.f_offs.back() + nverts);
				continue;
			}

			if (id == "t")
			{
				Data::Tag ct;
				in >> ct.type;

				auto intargs = 0;
				in >> intargs;

				in.get();

				auto floatargs = 0;
				in >> floatargs;

				in.get();

				auto stringargs = 0;
				in >> stringargs;

				ct.intargs.resize(intargs, 0);
				for (auto i = 0; i < intargs; ++i)
					in >> ct.intargs[i];

				ct.floatargs.resize(floatargs, 0);
				for (auto i = 0; i < floatargs; ++i)
					in >> ct.floatargs[i];

				data.t.push_back(ct);

				continue;
			}

			////UnknownEntry:
			//std::cout << "Warning: Unknown Entry \"" << in_line << "\" in line " << ln_nbr << "!" << std::endl;
		}

		int min = std::numeric_limits<int>::max();
		data.max_face_order = 0;
		for(auto it = data.f_offs.cbegin() + 1; it != data.f_offs.end(); ++it)
		{
			int order = *it - *(it - 1);
			min = std::min(order, min);
			data.max_face_order = std::max(order, data.max_face_order);
		}
		
		if (min != data.max_face_order || data.max_face_order > 4)
			data.type = Data::FaceType::POLY;
		else if (data.max_face_order == 4)
			data.type = Data::FaceType::QUAD;
		else if (data.max_face_order == 3)
			data.type = Data::FaceType::TRI;
		else
			throw std::runtime_error("Error loading obj: Face with less than three vertices detected.\n");
	}


	////////////////////////////////////////////////////////////////////////////////
	/// Write functionality

	inline void write(const Data& data, const char* fname)
	{
		std::ofstream out(fname);
		if (!out.is_open())
		{
			std::cerr << "Error opening mesh file for writing!" << std::endl;
			return;
		}

		std::cout << "Writing file " << fname << std::endl;
		std::cout << "Writing " << data.v.size() << " vertices" << std::endl;
		for (auto& vert : data.v)
			printVertexPos(out, vert);

		if (!data.f_offs.empty())
		{
			std::cout << "Writing " << data.f_offs.size() - 1 << " faces" << std::endl;
			for (size_t fid = 0; fid < data.f_offs.size() - 1; ++fid)
				printFace(out, &data.f_vi[data.f_offs[fid]], data.f_offs[fid + 1] - data.f_offs[fid]);
		}
		else
		{
			if (data.type == Data::FaceType::POLY)
			{
				std::cerr << "Error: Face-offsets required to write a poly mesh!" << std::endl;
				out.close();
				return;
			}

			const int fsize = data.type == Data::FaceType::QUAD ? 4 : 3;
			std::cout << "Writing " << data.f_vi.size() / fsize << " faces" << std::endl;
			for (size_t first = 0; first < data.f_vi.size(); first += fsize)
				printFace(out, &data.f_vi[0] + first, fsize);
		}

		std::cout << "Done writing" << std::endl;
		out.close();
	}

}
