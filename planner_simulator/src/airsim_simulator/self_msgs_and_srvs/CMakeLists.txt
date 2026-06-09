cmake_minimum_required(VERSION 2.8.3)
project(self_msgs_and_srvs)

set(CMAKE_CXX_FLAGS "-std=c++14 ${CMAKE_CXX_FLAGS} -O3 -Wall")
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(catkin REQUIRED COMPONENTS
  message_generation
  std_msgs
)


add_service_files(
  FILES
  GlbObsRcv.srv
)

generate_messages(
  DEPENDENCIES
  std_msgs
)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES self_msgs_and_srvs
  CATKIN_DEPENDS message_runtime std_msgs
#  DEPENDS system_lib
)

include_directories( 
  ${catkin_INCLUDE_DIRS}
)

