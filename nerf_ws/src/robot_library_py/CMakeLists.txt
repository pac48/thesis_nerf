cmake_minimum_required(VERSION 3.8)
project(robot_library_py)

set(CMAKE_CXX_STANDARD 17)

find_package(PythonInterp REQUIRED)
find_package(PythonLibs "${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}" REQUIRED )
if (PYTHONLIBS_FOUND)
    include_directories(${PYTHON_INCLUDE_DIRS})
    message("Using default python version: " ${PYTHONLIBS_VERSION_STRING})
endif()

# add binding library/module
add_subdirectory(parse_urdf)