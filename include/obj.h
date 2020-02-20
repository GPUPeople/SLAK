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

		std::ifstream in(fname);
		if (!in.is_open())
			throw std::runtime_error("Could not open obj file\n");

		int min_face = std::numeric_limits<int>::max();
		int max_face = 0;
		int row = 0;
		while (true)
		{

			while (in && std::isspace(in.peek()))
				in.get();

			if (in)
			{
				char c = in.get();

				switch (c)
				{
				case 'v':
				{
					if (std::isspace(in.peek()))
					{
						math::float3 p;
						in >> p.x >> p.y >> p.z;
						data.v.push_back(p);
						using namespace math;
					}
					else
					{
						char c2 = in.get();

						switch (c2)
						{
						case 'n':
						{
							math::float3 n;
							in >> n.x >> n.y >> n.z;
							n.z = -n.z;
							data.vn.push_back(n);
							break;
						}
						case 't':
						{
							math::float2 t;
							in >> t.x >> t.y;
							data.vt.push_back(t);

							while (in && in.peek() != '\n' && std::isspace(in.peek()))
								in.get();

							float t3;
							if (in.peek() != '\n')
							{
								in >> t3;
								if (in.fail())
								{
									in.clear();
									std::cout << "\rWARNING(ln " << row << "): additional content after 2D texture coordinate" << '\n';
								}
								else if (t3 != 0)
								{
									static bool warned = false;
									if (!warned)
										std::cout << "\rWARNING(ln " << row << "): 3D texture coordinates not supported" << '\n';
									warned = true;
								}

							}
						}
						}
					}
					break;
				}

				case 'f':
				{

					if (data.f_offs.size() > 1)
					{
						int face_order = data.f_vi.size() - data.f_offs.back();
						min_face = std::min(face_order, min_face);
						max_face = std::max(face_order, max_face);
					}
					data.f_offs.push_back(data.f_vi.size());

					while (in && in.peek() != '\n')
					{
						int vi, ni = -1, ti = -1;

						in >> vi;

						if (vi < 0)
							vi += static_cast<int>(data.v.size());
						else
							--vi;

						if (in.peek() == '/')
						{
							in.get();

							in >> ti;

							if (in.fail())
								in.clear();
							else
							{
								if (ti < 0)
									ti += static_cast<int>(data.vt.size());
								else
									--ti;

								data.f_vti.push_back(ti);
							}

							if (in.peek() == '/')
							{
								in.get();

								in >> ni;

								if (in.fail())
									in.clear();
								else
								{
									if (ni < 0)
										ni += static_cast<int>(data.vn.size());
									else
										--ni;

									data.f_vni.push_back(ni);
								}
							}
						}
						data.f_vi.push_back(vi);

						while (in && std::isblank(in.peek()))
							in.get();
					}


					break;
				}

				case 't':
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

					break;
				}

				case 'g':
				{
					std::string group_name_unused;
					in >> group_name_unused;
					break;
				}

				case '#':
					// ignore comments
					while (in && in.get() != '\n') {};
					break;

				default:
					//UnknownEntry:
				{
					std::stringstream unknown;
					bool nontrivial = (!std::isspace(c) && !std::iscntrl(c) && (c != std::char_traits<char>::eof()));
					unknown << c;
					while (in && (c = in.get()) != '\n')
						nontrivial |= (!std::isspace(c) && !std::iscntrl(c) && (c != std::char_traits<char>::eof())),
						unknown << c;
					//if (nontrivial)
					//	std::cout << "\rWARNING (ln " << row << "): unknown entry in obj: \"" << unknown.str() << "\"\n";
					break;
				}
				}
			}
			else
				break;
		}

		min_face = std::min(static_cast<int>(data.f_vi.size() - data.f_offs.back()), min_face);
		max_face = std::max(static_cast<int>(data.f_vi.size() - data.f_offs.back()), max_face);
		data.f_offs.push_back(data.f_vi.size());
		data.max_face_order = max_face;


		bool homogeneous_face_type = min_face == max_face;
		data.type = !homogeneous_face_type ? (Data::FaceType::POLY) : (min_face == 3 ? Data::FaceType::TRI : (min_face == 4 ? Data::FaceType::QUAD : Data::FaceType::POLY));

		if (homogeneous_face_type && min_face == 3)
		{
			data.type = Data::FaceType::TRI;
		}
		else if (homogeneous_face_type && min_face == 4)
		{
			data.type = Data::FaceType::QUAD;
		}
		else
			data.type = Data::FaceType::POLY;


		if (data.type == Data::FaceType::QUAD)
		{
			if (data.vn.size() != data.v.size())
			{
				//std::cout << "Generating smooth normals\n";
				data.vn.resize(data.v.size());
				std::fill(data.vn.begin(), data.vn.end(), math::float3(0));
				for (size_t i = 3; i < data.f_vi.size(); i += 4)
				{
					const math::float3& p0 = data.v[data.f_vi[i - 3]];
					const math::float3& p1 = data.v[data.f_vi[i - 2]];
					const math::float3& p2 = data.v[data.f_vi[i - 1]];
					const math::float3& p3 = data.v[data.f_vi[i - 0]];
					const math::float3 n012 = cross(p1 - p0, p2 - p1);
					const math::float3 n023 = cross(p3 - p2, p0 - p3);
					const math::float3 n123 = cross(p2 - p1, p3 - p2);
					const math::float3 n013 = cross(p0 - p3, p1 - p0);
					data.vn[data.f_vi[i - 3]] += n023 + n123 + n013;
					data.vn[data.f_vi[i - 2]] += n012 + n023 + n123;
					data.vn[data.f_vi[i - 1]] += n012 + n123 + n013;
					data.vn[data.f_vi[i - 0]] += n012 + n023 + n013;
				}
				for (auto& n : data.vn)
					if (length2(n) > 0.000001f)
						n = normalize(n);
			}
		}
		else if (data.type == Data::FaceType::TRI)
		{
			if (data.vn.size() != data.v.size())
			{
				//std::cout << "Generating smooth normals\n";
				data.vn.resize(data.v.size());
				std::fill(data.vn.begin(), data.vn.end(), math::float3(0));
				for (size_t i = 2; i < data.f_vi.size(); i += 3)
				{
					const math::float3& p0 = data.v[data.f_vi[i - 2]];
					const math::float3& p1 = data.v[data.f_vi[i - 1]];
					const math::float3& p2 = data.v[data.f_vi[i - 0]];
					const math::float3 n = cross(p1 - p0, p2 - p1);
					data.vn[data.f_vi[i - 2]] += n;
					data.vn[data.f_vi[i - 1]] += n;
					data.vn[data.f_vi[i - 0]] += n;
				}
				for (auto& n : data.vn)
					if (length2(n) > 0.000001f)
						n = normalize(n);
			}
		}
		else
		{
			if (data.vn.size() != data.v.size())
			{
				//std::cout << "Generating smooth normals\n";
				data.vn.resize(data.v.size());
				std::fill(data.vn.begin(), data.vn.end(), math::float3(0));

				for (auto fid = 0; fid < data.f_offs.size() - 1; ++fid)
				{
					std::vector<int>id_current(data.f_vi.begin() + data.f_offs[fid], data.f_vi.begin() + data.f_offs[fid + 1]);
					unsigned i0 = (0 - 1) % 4;

					auto face_order = data.f_offs[fid + 1] - data.f_offs[fid];
					for (auto i = 0; i < face_order; ++i)
					{
						math::float3 p0 = data.v[id_current[(i + face_order - 1) % face_order]];
						math::float3 p1 = data.v[id_current[i]];
						math::float3 p2 = data.v[id_current[(i + 1) % face_order]];
						math::float3 n = cross(p1 - p0, p2 - p1);
						data.vn[id_current[(i + face_order - 1) % face_order]] += n;
						data.vn[id_current[i]] += n;
						data.vn[id_current[(i + 1) % face_order]] += n;

					}
				}
			}
			for (auto& n : data.vn)
				if (length2(n) > 0.000001f)
					n = normalize(n);
		}
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
			if(data.type == Data::FaceType::POLY)
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
