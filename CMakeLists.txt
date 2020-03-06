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

cmake_minimum_required(VERSION 3.10)
project(SLAK LANGUAGES CXX CUDA)

#some problem with gcc-9 for whatever reason
if(WIN32)
elseif(CMAKE_CXX_COMPILER_ID EQUAL GNU AND CMAKE_CXX_COMPILER_VERSION GREATER 8.4)
message(FATAL_ERROR "ERROR: GCC ${CMAKE_CXX_COMPILER_VERSION} not supported. Please specify a different generator.")
endif()

include(ExternalProject)
find_package(Git REQUIRED)

##########################################################################
# Setup cub
ExternalProject_Add(
	cub
	GIT_REPOSITORY https://github.com/NVlabs/cub
	TIMEOUT 10
    UPDATE_COMMAND ${GIT_EXECUTABLE} pull
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
)
ExternalProject_Get_Property(cub SOURCE_DIR)
set(CUBHDRS ${SOURCE_DIR})

##########################################################################
# Shared Include Directories
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

##########################################################################
# Choose for which CC to build and if to enable Debug
option(CUDA_BUILD_CC61 "Build with compute capability 6.1 support" TRUE)
option(CUDA_BUILD_CC70_SYNC "Build with compute capability 7.0 support - SYNC" FALSE)
option(CUDA_BUILD_CC70_ASYNC "Build with compute capability 7.0 support - ASYNC" FALSE)
option(CUDA_BUILD_CC75_SYNC "Build with compute capability 7.5 support - SYNC" FALSE)
option(CUDA_BUILD_CC75_ASYNC "Build with compute capability 7.5 support - ASYNC" FALSE)
option(CUDA_BUILD_INFO "Build with kernel statistics and line numbers" TRUE)
option(CUDA_BUILD_DEBUG "Build with kernel debug" FALSE)

##########################################################################
# CUDA Flags
if (CUDA_BUILD_CC61)
	string(APPEND CMAKE_CUDA_FLAGS " -gencode=arch=compute_61,code=sm_61")
endif ()
if (CUDA_BUILD_CC70_SYNC)
	string(APPEND CMAKE_CUDA_FLAGS " -gencode=arch=compute_60,code=sm_70")
endif ()
if (CUDA_BUILD_CC70_ASYNC)
	string(APPEND CMAKE_CUDA_FLAGS " -gencode=arch=compute_70,code=sm_70")
endif ()
if (CUDA_BUILD_CC75_SYNC)
	string(APPEND CMAKE_CUDA_FLAGS " -gencode=arch=compute_60,code=sm_75")
endif ()
if (CUDA_BUILD_CC75_ASYNC)
	string(APPEND CMAKE_CUDA_FLAGS " -gencode=arch=compute_75,code=sm_75")
endif ()
if (CUDA_BUILD_INFO)
	string(APPEND CMAKE_CUDA_FLAGS " -keep --ptxas-options=-v -lineinfo")
endif ()
if (CUDA_BUILD_DEBUG)
	string(APPEND CMAKE_CUDA_FLAGS " -G")
endif ()

##########################################################################
# CXX Flags
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
PRIVATE ${LIBHDRPRIV} ${CUBHDRS}
PUBLIC ${LIBHDRPUB}
)

add_dependencies(slaklib cub)

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
