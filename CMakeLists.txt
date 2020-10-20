# Copyright (c) 2020  Daniel Mlakar daniel.mlakar@icg.tugraz.at
					# Martin Winter martin.winter@icg.tugraz.at
					# Pascal Stadlbauer pascal.stadlbauer@icg.tugraz.at
					# Hans-Peter Seidel hpseidel@mpi-inf.mpg.de
					# Markus Steinberger steinberger@icg.tugraz.at
					# Rhaleb Zayer rzayer@mpi-inf.mpg.de

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

cmake_minimum_required(VERSION 3.16)
project(SLAK LANGUAGES CXX CUDA)

##########################################################################
# Shared Include Directories
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})


##########################################################################
# CXX Flags

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if (WIN32)

else()
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
	set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")
	set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
endif()

set(CUDA_PROPAGATE_HOST_FLAGS ON)

##########################################################################
# Setup library
set(LIBSRC "LibSource/device/")
file(GLOB LIBSRCFILES ${LIBSRC}*.cu ${LIBSRC}/math/*.cu)

set(LIBHDRPRIV "LibInclude/device/")
file(GLOB LIBHDRPRIVFILES ${LIBHDRPRIV}*.cuh ${LIBHDRPRIV}/math/*.cuh)

set(LIBHDRPUB "LibInclude/")
file(GLOB LIBHDRPUBFILES ${LIBHDRPUB}.*h)

# this is a dummy file to fix intellisense
set(FIXINTELLISENSE "LibSource/FixIntellisense.cpp")

add_library(slaklib STATIC ${FIXINTELLISENSE} ${LIBSRCFILES} ${LIBHDRPRIVFILES} ${LIBHDRPUBFILES})

target_include_directories(slaklib
PRIVATE ${LIBHDRPRIV}
PUBLIC ${LIBHDRPUB}
)

if(NOT CMAKE_CUDA_ARCHITECTURES)
	set_property(TARGET slaklib PROPERTY CUDA_ARCHITECTURES 61 75)
endif()

##########################################################################
# Setup executable
set(SRC "source/")
file(GLOB SRCFILES ${SRC}*.cpp)

set(HDR "include/")
file(GLOB HDRFILES ${HDR}*.h)

add_executable(${PROJECT_NAME} ${SRCFILES} ${HDRFILES})

target_include_directories(${PROJECT_NAME}
PRIVATE ${HDR}
)

##########################################################################
# Executable properties
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD_REQUIRED ON)



##########################################################################
# Link libraries
target_link_libraries(slaklib cusparse)
target_link_libraries(${PROJECT_NAME} slaklib)
