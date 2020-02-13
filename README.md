# SLAK
Sparse Linear Algebra Subdivision Kernels

This is a library that performs mesh subdivision on Nvidia GPUs, based on specialized sparse linear algebra kernels.
The repository will be continuously updated as more functionality gets ported from the research implementation.
Keep posted!

To compile the library and the application that demonstrates its usage:
1. cd path/to/this/README.md
2. mkdir build
3. cd build
4. cmake ..
5. make


There are two usage options:

Option 1: subdivide single mesh
SLAK <mesh> <scheme> <level> <save>
	<mesh>: the path to an .obj file to perform subdivision on
	<scheme>:
		0...LAK Catmull-Clark
		1...SLAK Catmull-Clark
	<level>: number of subdivision iterations to perform
	<save>:
		0...discard the refined mesh
		1...save the refined mesh to disk (same location as input)
		
Option 2: subdivide multiple meshes
SLAK <config>
	<config>: path to a configuration file where each line contains <mesh> <scheme> <level> <save>
