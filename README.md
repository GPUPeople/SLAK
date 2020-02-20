# SLAK
This repository holds code accompanying the paper

Subdivision-Specialized Linear Algebra Kernels for Static and Dynamic Mesh Connectivity on the GPU.
D. Mlakar, M. Winter, P. Stadlbauer, H.-P. Seidel, M.Steinberger, R. Zayer.

and is aimed to apply for the Graphics Replicability Stamp. The code reproduces the results given in
Figure 7 in the above paper. 

To compile and run the experiments on Linux or Windows simply run the python script "setupAndRun.py", located in
the same path as this "README.md". The script downloads the test data, extracts it to "./data/"
(it will ask for the password to decrypt the archive), runs cmake, downloads dependencies, builds the code and runs the experiments.
The configuration file "./data/config.txt" holds one line per experiment, where each is structured as follows:

<mesh> <scheme> <level> <save>
<mesh>: the path to an .obj file to perform subdivision on
<scheme>:
	0...LAK Catmull-Clark
	1...SLAK Catmull-Clark
<level>: number of subdivision iterations to perform
<save>:
	0...discard the refined mesh
	1...save the refined mesh to disk (same location as input)
		

The "setupAndRun.py" script re-builds the executable on each invocation.
To manually run the experiments without the script use:
SLAK <config>
	<config>: path to a configuration file where each line contains <mesh> <scheme> <level> <save>


Requirements:
Linux or Windows
python
CMake Version 3.16
gcc-8 or Visual Studio 2019
CUDA 10.2
Nvidia GPU with CC 6.1, 7.0 or 7.5