cmake_minimum_required(VERSION 2.8.3)
project(rgb_point_map)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++14")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native")
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
get_filename_component(PROJECT_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR} DIRECTORY)
# LIBIGL
set(LIBIGL_INCLUDE_DIR "${PROJECT_ROOT_DIR}/3rdparty/libigl/include")
set(LIBIGL_LIBRARY "${PROJECT_ROOT_DIR}/3rdparty/libigl/libigl.a")

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(PCL 1.7 REQUIRED)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  roslib
  message_filters
  cv_bridge
  quadrotor_msgs
)
find_package(roslib)

catkin_package(
 INCLUDE_DIRS include
 LIBRARIES rgb_point_map
 CATKIN_DEPENDS roscpp
)

include_directories( 
    SYSTEM 
    include 
    ${catkin_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR}/include ${PROJECT_ROOT_DIR}
    ${EIGEN3_INCLUDE_DIR} 
    ${PCL_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${LIBIGL_INCLUDE_DIR}
)

link_directories(${PCL_LIBRARY_DIRS})


add_library( rgb_point_map SHARED
    src/rgb_point_map.cpp 
    src/rgb_point_map_utils.cpp
    )
target_link_libraries( rgb_point_map
    ${catkin_LIBRARIES} 
    ${PCL_LIBRARIES}
    ${LIBIGL_LIBRARY}
    )  

add_executable( rgb_point_mapping_node 
    src/mapping.cpp
    )
target_link_libraries( rgb_point_mapping_node
    rgb_point_map
    ${catkin_LIBRARIES} 
    ${PCL_LIBRARIES}
    ${LIBIGL_LIBRARY}
    )  
