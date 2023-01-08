cmake_minimum_required(VERSION 3.13 FATAL_ERROR)
set(CMAKE_CXX_STANDARD 14)

project(libaccvr)

find_package(Torch REQUIRED)
find_package(PythonLibs REQUIRED)
find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib")

cuda_add_library(libaccvr SHARED)
target_include_directories(
    libaccvr PRIVATE
    cpp/include)
file(GLOB SRC_FILES cpp/src/*.cpp cpp/src/*.c cpp/src/*.cu)
target_sources(libaccvr PUBLIC ${SRC_FILES})

set_target_properties(libaccvr PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(libaccvr PROPERTIES PREFIX "")
target_link_libraries(libaccvr ${TORCH_PYTHON_LIBRARY})
target_compile_definitions(libaccvr PUBLIC TORCH_EXTENSION_NAME=libaccvr)
target_compile_definitions(libaccvr PUBLIC TORCH_API_INCLUDE_EXTENSION_H)
target_compile_definitions(libaccvr PUBLIC ${TORCH_CXX_FLAGS})
target_include_directories(libaccvr PUBLIC ${TORCH_INCLUDE_DIRS})
target_include_directories(libaccvr PUBLIC ${PYTHON_INCLUDE_DIRS})
