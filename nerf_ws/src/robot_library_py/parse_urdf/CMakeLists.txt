find_package(pybind11 REQUIRED)

find_library(TINYXML2_LIB tinyxml2)
find_package(RL REQUIRED)


pybind11_add_module(robot_library_py _core/cpp_library.cpp)
target_include_directories(robot_library_py PRIVATE _core)
include_directories(${Eigen3_INCLUDE_DIRS})

target_link_libraries(robot_library_py PUBLIC ${TINYXML2_LIB})
target_link_libraries(robot_library_py PUBLIC ${RL_LIBRARIES})

install(TARGETS robot_library_py LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/python3.10/site-packages/)
