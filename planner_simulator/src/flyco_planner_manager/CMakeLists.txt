cmake_minimum_required(VERSION 2.8.3)
cmake_policy(SET CMP0048 OLD)
project(flyco_planner_manager)

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
    vis_utils
    roscpp
    roslib
    std_msgs
    path_searching
    plan_env
    quadrotor_msgs
    active_perception
    airsim_ros_pkgs
    message_filters
    hierarchical_coverage_planner
    viewpoint_manager
    rgb_point_map
    cv_bridge
)
find_package(roslib)
find_package(PCL REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(ompl REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(OpenCV COMPONENTS highgui)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES
  CATKIN_DEPENDS vis_utils plan_env active_perception airsim_ros_pkgs planner_node manual_node init_traj hierarchical_coverage_planner rgb_point_map
)

include_directories( 
    SYSTEM 
    ${EIGEN3_INCLUDE_DIR} 
    include 
    ${catkin_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR}/include ${PROJECT_ROOT_DIR}
    ${PCL_INCLUDE_DIRS}
    ${OMPL_INCLUDE_DIRS}
    ${EMBREE_INCLUDE_DIR}
    ${LIBIGL_INCLUDE_DIR}
)

link_directories(
    ${PCL_LIBRARY_DIRS})

add_executable( traj_server
    src/traj_server.cpp
)

target_link_libraries( traj_server
    ${catkin_LIBRARIES}
    yaml-cpp
)

add_executable(flyco_planner
    src/flyco_planner_node.cpp 
    src/flyco_planner_manager.cpp
  )
target_link_libraries(flyco_planner 
    ${catkin_LIBRARIES}
    ${PCL_LIBRARIES}
    ${OMPL_LIBRARIES}
    ${LIBIGL_LIBRARY}
    ${EMBREE_LIBRARY}
    yaml-cpp
  )

add_executable(fc_data
  src/fc_data_collection.cpp
)

target_link_libraries(fc_data
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  ${OMPL_LIBRARIES}
  ${LIBIGL_LIBRARY}
  ${EMBREE_LIBRARY}
  yaml-cpp
)
