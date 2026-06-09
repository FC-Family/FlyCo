cmake_minimum_required(VERSION 2.8.3)
project(skeleton_decomp)

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
    vis_utils
    roscpp
    std_msgs
    quadrotor_msgs
)
find_package(PCL REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(ompl REQUIRED)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES sk_decomp
  CATKIN_DEPENDS vis_utils
)

include_directories( 
    SYSTEM 
    include 
    ${catkin_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR}/include ${PROJECT_ROOT_DIR}
    ${LIBIGL_INCLUDE_DIR}
    ${EIGEN3_INCLUDE_DIR} 
    ${PCL_INCLUDE_DIRS}
    ${OMPL_INCLUDE_DIRS}
    ${EMBREE_INCLUDE_DIR}
)

link_directories(
    ${PCL_LIBRARY_DIRS})

add_library(adaptive_utils
    src/adaptive_utils.cpp
    )
target_link_libraries(adaptive_utils
    ${catkin_LIBRARIES}
    ${PCL_LIBRARIES}
    ${LIBIGL_LIBRARY}
    ${EMBREE_LIBRARY}
    )  

add_library(sk_decomp
   src/sk_decomp.cpp
   src/Extra_Del.cpp
   src/adaptive_utils.cpp
)

target_link_libraries(sk_decomp
    ${catkin_LIBRARIES}
    ${PCL_LIBRARIES}
    ${LIBIGL_LIBRARY}
    ${EMBREE_LIBRARY}
    ${OMPL_LIBRARIES}
    ) 

add_executable(sk_exec
    src/sk_exec.cpp
    src/sk_decomp.cpp
    src/Extra_Del.cpp
    src/adaptive_utils.cpp
    )

target_link_libraries( sk_exec
    ${catkin_LIBRARIES}
    ${PCL_LIBRARIES}
    ${LIBIGL_LIBRARY}
    ${EMBREE_LIBRARY}
    ${OMPL_LIBRARIES}
    )
