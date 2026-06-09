cmake_minimum_required(VERSION 3.0.2)
project(ue4_control)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
add_compile_options(-Wall)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  sensor_msgs
  tf2
  tf
  mavros_msgs
  rosgraph_msgs
  quadrotor_msgs
  message_filters
  airsim_ros_pkgs
  active_perception
)
find_package(Eigen3 REQUIRED)
catkin_package(
  INCLUDE_DIRS
  LIBRARIES
  ${PROJECT_NAME}
  CATKIN_DEPENDS
  roscpp
  geometry_msgs
  tf2
  tf
  mavros_msgs
  rosgraph_msgs
  quadrotor_msgs
  airsim_ros_pkgs
  DEPENDS
  EIGEN3
)

include_directories(
  ${catkin_INCLUDE_DIRS} ${PROJECT_ROOT_DIR}
  ${EIGEN3_INCLUDE_DIR}
)


add_executable(joy_control src/joy_control.cpp)
target_link_libraries(joy_control
${catkin_LIBRARIES}
)

add_executable(joy_cmd src/joy_cmd.cpp)
target_link_libraries(joy_cmd
${catkin_LIBRARIES}
)
