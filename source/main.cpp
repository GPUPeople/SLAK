// Copyright (c) 2020 	Daniel Mlakar daniel.mlakar@icg.tugraz.at
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

#include <fstream>

#include "Mesh.h"

#include "LAKCatmullClark.h"
#include "CatmullClark.h"

int main(int argc, char* argv[])
{
	try
	{
		std::vector<std::string> paths;
		std::vector<int> schemes;
		std::vector<int> target_levels;
		std::vector<bool> save_results;

		if (argc == 2)
		{
			std::ifstream file;
			file.open(argv[1], std::ios::in);
			if (!file.is_open())
				std::cerr << "Error: could not open file\n" << std::endl;

			while (!file.eof())
			{
				std::string line;
				std::getline(file, line);

				if (line[0] == '#')
					continue;

				std::istringstream ss(line);
				std::string path; int scheme; int target_level; bool save_result;
				ss >> path >> scheme >> target_level >> save_result;

				paths.emplace_back(path);
				schemes.emplace_back(scheme);
				target_levels.emplace_back(target_level);
				save_results.emplace_back(save_result);
			}

			file.close();
		}
		else if (argc == 5)
		{
			paths.emplace_back(std::string(argv[1]));
			schemes.emplace_back(atoi(argv[2]));
			target_levels.emplace_back(atoi(argv[3]));
			save_results.emplace_back(static_cast<bool>(atoi(argv[4])));
		}
		else
		{
			std::cerr << "Usage: NoViewerSLAK <path_to_obj> <scheme> <target_level> <save_result>" << std::endl;
			std::cerr << "Usage: NoViewerSLAK config_file" << std::endl;
			return -1;
		}



		for (size_t i = 0; i < paths.size(); ++i)
		{
			std::cout << "Mesh: " << paths[i] << std::endl;
			const GPU::CSCMesh<int, int, int, float> cmesh(paths[i]);
			GPU::CSCMesh<int, int, int, float> rmesh;
			if (schemes[i] == 0)
			{
				using MeshInfo = LAKCatmullClark::MeshInfo<int, int, int, float>;
				MeshInfo cmesh_info, rmesh_info;
				GPU::CSCMeshCCMeshInfoConversion(cmesh, cmesh_info);
				LAKCatmullClark::subdivideIteratively(cmesh_info, rmesh_info, target_levels[i]);
				GPU::CSCMeshCCMeshInfoConversion(rmesh_info, rmesh);
			}
			else if (schemes[i] == 1)
			{
				using MeshInfo = CatmullClark::MeshInfo<int, int, int, float>;
				MeshInfo cmesh_info, rmesh_info;
				GPU::CSCMeshCCMeshInfoConversion(cmesh, cmesh_info);
				CatmullClark::subdivideIteratively(cmesh_info, rmesh_info, target_levels[i]);
				GPU::CSCMeshCCMeshInfoConversion(rmesh_info, rmesh);
			}

			if (save_results[i])
			{
				std::string scheme_id = schemes[i] == 0 ? "LAK" : "SLAK";
				std::string out_file_prefix = paths[i].substr(0, paths[i].find_last_of("."));
				rmesh.toObj(out_file_prefix + "_" + std::to_string(target_levels[i]) + "_" + scheme_id + ".obj");
			}

			std::cout << "\n\n";
		}
	}
	catch (const std::exception & e)
	{
		printf("%s\n", e.what());
	}
	catch (...)
	{

	}
	return 0;
}

