project(laplacian_py)
cmake_minimum_required(VERSION 3.22.1)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
enable_language(CUDA)
find_package(CUDA REQUIRED)
find_package(pybind11 REQUIRED)
find_package(PythonInterp REQUIRED)

execute_process(
    COMMAND "${PYTHON_EXECUTABLE}" -c "import torch;print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TORCH_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE)

set(TORCH_LIB_DIR ${TORCH_CMAKE_DIR}/../../lib)
set(TORCH_INCLUDE_DIR ${TORCH_CMAKE_DIR}/../../include/torch/csrc/api/include/)
set(TORCH_INCLUDE_DIR_ROOT ${TORCH_CMAKE_DIR}/../../include/)


if (PYTHONLIBS_FOUND)
  include_directories(${PYTHON_INCLUDE_DIRS})
  message("Using default python version: " ${PYTHONLIBS_VERSION_STRING})
endif ()

pybind11_add_module(laplacian_solver_py liblaplacian_py.cu)
add_library(laplacian_solver liblaplacian.cu)
target_link_libraries(laplacian_solver_py PUBLIC laplacian_solver)
#add_executable(main main.cu)

set(TARGETS laplacian_solver_py)
foreach (target ${TARGETS})
  target_compile_definitions(${target} PRIVATE NDEBUG)
  target_compile_definitions(${target} PRIVATE STRIP_ERROR_MESSAGES)
  target_include_directories(${target} PUBLIC ${TORCH_INCLUDE_DIR})
  target_include_directories(${target} PUBLIC ${TORCH_INCLUDE_DIR_ROOT})
  target_link_libraries(${target} PUBLIC ${TORCH_LIB_DIR}/libc10_cuda.so)
  target_link_libraries(${target} PUBLIC ${TORCH_LIB_DIR}/libtorch_cuda_cpp.so)
  target_link_libraries(${target} PUBLIC ${TORCH_LIB_DIR}/libc10.so)
  target_link_libraries(${target} PUBLIC ${TORCH_LIB_DIR}/libtorch_cuda_cu.so)
  target_link_libraries(${target} PUBLIC ${TORCH_LIB_DIR}/libcaffe2_nvrtc.so)
  target_link_libraries(${target} PUBLIC ${TORCH_LIB_DIR}/libtorch_cuda_linalg.so)
  target_link_libraries(${target} PUBLIC ${TORCH_LIB_DIR}/libcudart-e409450e.so.11.0)
  target_link_libraries(${target} PUBLIC ${TORCH_LIB_DIR}/libtorch_cuda.so)
  target_link_libraries(${target} PUBLIC ${TORCH_LIB_DIR}/libgomp-a34b3233.so.1)
  target_link_libraries(${target} PUBLIC ${TORCH_LIB_DIR}/libtorch_global_deps.so)
  target_link_libraries(${target} PUBLIC ${TORCH_LIB_DIR}/libnvToolsExt-847d78f2.so.1)
  target_link_libraries(${target} PUBLIC ${TORCH_LIB_DIR}/libtorch_python.so)
  target_link_libraries(${target} PUBLIC ${TORCH_LIB_DIR}/libshm.so)
  target_link_libraries(${target} PUBLIC ${TORCH_LIB_DIR}/libtorch.so)
  target_link_libraries(${target} PUBLIC ${TORCH_LIB_DIR}/libtorch_cpu.so)
endforeach ()




install(TARGETS laplacian_solver_py LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/python3.10/site-packages/laplacian_py)
install(FILES python/__init__.py DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/python3.10/site-packages/laplacian_py)
install(FILES python/network.py DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/python3.10/site-packages/laplacian_py)
