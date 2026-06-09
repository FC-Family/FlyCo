cmake_minimum_required(VERSION 2.8.3)
project(viewpoint_manager)

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
    roscpp
    std_msgs
    skeleton_decomp
    vis_utils
    plan_env
    active_perception
    quadrotor_msgs
)
find_package(PCL REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES viewpoint_manager
  CATKIN_DEPENDS vis_utils plan_env skeleton_decomp
)

include_directories( 
    SYSTEM 
    include 
    ${catkin_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR}/include ${PROJECT_ROOT_DIR}
    ${LIBIGL_INCLUDE_DIR}
    ${EIGEN3_INCLUDE_DIR} 
    ${PCL_INCLUDE_DIRS}
    ${EMBREE_INCLUDE_DIR}
)

link_directories(
    ${PCL_LIBRARY_DIRS})

add_library( viewpoint_manager
    src/viewpoint_manager.cpp
)

target_link_libraries( viewpoint_manager
    ${catkin_LIBRARIES} 
    ${PCL_LIBRARIES}
    ${LIBIGL_LIBRARY}
    ${EMBREE_LIBRARY}
    )  

add_dependencies(viewpoint_manager ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
