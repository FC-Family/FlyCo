cmake_minimum_required(VERSION 2.8.3)
cmake_policy(SET CMP0048 OLD)
project(hierarchical_coverage_planner)

add_definitions(-D _GLIBCXX_USE_CXX11_ABI=1)
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++14")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall")
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
get_filename_component(PROJECT_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR} DIRECTORY)
# LIBIGL
set(LIBIGL_INCLUDE_DIR "${PROJECT_ROOT_DIR}/3rdparty/libigl/include")
set(LIBIGL_LIBRARY "${PROJECT_ROOT_DIR}/3rdparty/libigl/libigl.a")
# EMBREE
set(EMBREE_INCLUDE_DIR "${PROJECT_ROOT_DIR}/3rdparty/embree-3/include")
set(EMBREE_LIBRARY "${PROJECT_ROOT_DIR}/3rdparty/embree-3/lib/libembree3.so")
add_compile_options(-I${EMBREE_INCLUDE_DIR})

find_package(catkin REQUIRED COMPONENTS
    skeleton_decomp
    vis_utils
    roscpp
    roslib
    std_msgs
    plan_env
    active_perception
    path_searching
    quadrotor_msgs
    viewpoint_manager
)
find_package(roslib)
find_package(PCL REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(ompl REQUIRED)
find_package(OpenCV REQUIRED)
find_package(yaml-cpp REQUIRED)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES planner_node manual_node init_traj
  CATKIN_DEPENDS skeleton_decomp vis_utils plan_env
  DEPENDS OpenCV
)

include_directories( 
    SYSTEM 
    include 
    ${catkin_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR}/include ${PROJECT_ROOT_DIR}
    ${PROJECT_ROOT_DIR}/flyco_planner_manager/include
    ${LIBIGL_INCLUDE_DIR}
    ${EIGEN3_INCLUDE_DIR} 
    ${PCL_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${OMPL_INCLUDE_DIRS}
    ${EMBREE_INCLUDE_DIR}
)

link_directories(
    ${PCL_LIBRARY_DIRS})

add_library (planner_node
    src/fcplanner_pp.cpp
    src/local_replan.cpp
    src/hcsolver.cpp
    src/hctraj.cpp
    src/planner_node.cpp
)

target_link_libraries( planner_node
    ${catkin_LIBRARIES} 
    ${PCL_LIBRARIES}
    ${OMPL_LIBRARIES}
    ${LIBIGL_LIBRARY}
    ${EMBREE_LIBRARY}
    yaml-cpp
)

add_library (manual_node
    src/manual_node.cpp
)

target_link_libraries( manual_node
    ${catkin_LIBRARIES} 
    ${PCL_LIBRARIES}
    ${EMBREE_LIBRARY}
)

add_library (init_traj
    src/init_traj.cpp
    src/hctraj.cpp
)

target_link_libraries( init_traj
    ${catkin_LIBRARIES} 
    ${PCL_LIBRARIES}
    ${OMPL_LIBRARIES}
    yaml-cpp
)
