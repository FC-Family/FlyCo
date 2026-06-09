cmake_minimum_required(VERSION 3.0.2)
project(sam_tracking)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

add_definitions(-D _GLIBCXX_USE_CXX11_ABI=1)
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++14")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall")
set(CMAKE_CXX_STANDARD 14)
## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  rospy
  cv_bridge
  geometry_msgs
  message_filters
  perception_msgs
  sensor_msgs
  std_msgs
  pcl_ros
)

find_package(OpenCV REQUIRED)
find_package(PCL REQUIRED)
find_package(Boost REQUIRED COMPONENTS system)
find_package(Eigen3 REQUIRED)


## System dependencies are found with CMake's conventions
# find_package(Boost REQUIRED COMPONENTS system)


## Uncomment this if the package has a setup.py. This macro ensures
## modules and global scripts declared therein get installed
## See http://ros.org/doc/api/catkin/html/user_guide/setup_dot_py.html
catkin_python_setup()
catkin_add_env_hooks(50.sam_tracking_pythonpath SHELLS bash DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/env-hooks)

################################################
## Declare ROS dynamic reconfigure parameters ##
################################################

## To declare and build dynamic reconfigure parameters within this
## package, follow these steps:
## * In the file package.xml:
##   * add a build_depend and a exec_depend tag for "dynamic_reconfigure"
## * In this file (CMakeLists.txt):
##   * add "dynamic_reconfigure" to
##     find_package(catkin REQUIRED COMPONENTS ...)
##   * uncomment the "generate_dynamic_reconfigure_options" section below
##     and list every .cfg file to be processed

## Generate dynamic reconfigure parameters in the 'cfg' folder
# generate_dynamic_reconfigure_options(
#   cfg/DynReconf1.cfg
#   cfg/DynReconf2.cfg
# )

###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if your package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES ros_edge_sam
   CATKIN_DEPENDS geometry_msgs message_filters perception_msgs rospy sensor_msgs std_msgs pcl_ros
#  DEPENDS system_lib
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}

)

## Declare a C++ library
# add_library(${PROJECT_NAME}
#   src/${PROJECT_NAME}/ros_edge_sam.cpp
# )

## Add cmake target dependencies of the library
## as an example, code may need to be generated before libraries
## either from message generation or dynamic reconfigure
# add_dependencies(${PROJECT_NAME} ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

## Declare a C++ executable
## With catkin_make all packages are built within a single CMake context
## The recommended prefix ensures that target names across packages don't collide
add_executable(${PROJECT_NAME}_node
  src/cluster/test_cluster_node.cpp
  src/cluster/pc_cluster.cpp
  src/cluster/pc_utils.cpp)

## Rename C++ executable without prefix
## The above recommended prefix causes long target names, the following renames the
## target back to the shorter version for ease of user use
## e.g. "rosrun someones_pkg node" instead of "rosrun someones_pkg someones_pkg_node"
# set_target_properties(${PROJECT_NAME}_node PROPERTIES OUTPUT_NAME node PREFIX "")

## Add cmake target dependencies of the executable
## same as for the library above
# add_dependencies(${PROJECT_NAME}_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

## Specify libraries to link a library or executable target against
target_link_libraries(${PROJECT_NAME}_node
  ${catkin_LIBRARIES}
  ${OpenCV_LIBS}
  ${PCL_LIBRARIES}
  ${Boost_LIBRARIES}
  ${EIGEN3_LIBRARIES}

)

#############
## Install ##
#############

# all install targets should use catkin DESTINATION variables
# See http://ros.org/doc/api/catkin/html/adv_user_guide/variables.html

## Mark executable scripts (Python etc.) for installation
## in contrast to setup.py, you can choose the destination
catkin_install_python(PROGRAMS
  nodes/tracking_node.py
  nodes/prompt_node.py
  nodes/prompt_ui_node.py
  nodes/prompt_ui_cv_node.py
  nodes/world_rgb_pointcloud_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

## Mark executables for installation
## See http://docs.ros.org/melodic/api/catkin/html/howto/format1/building_executables.html
install(TARGETS ${PROJECT_NAME}_node
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

## Mark libraries for installation
## See http://docs.ros.org/melodic/api/catkin/html/howto/format1/building_libraries.html
# install(TARGETS ${PROJECT_NAME}
#   ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   RUNTIME DESTINATION ${CATKIN_GLOBAL_BIN_DESTINATION}
# )

## Mark cpp header files for installation
# install(DIRECTORY include/${PROJECT_NAME}/
#   DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
#   FILES_MATCHING PATTERN "*.h"
#   PATTERN ".svn" EXCLUDE
# )

## Mark other files for installation (e.g. launch and bag files, etc.)
install(DIRECTORY
  launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
  FILES_MATCHING PATTERN "*.launch"
  PATTERN "sam_tracking.launch" EXCLUDE
  PATTERN "sam_tracking_bag_test.launch" EXCLUDE
  PATTERN "ui_bag_test.launch" EXCLUDE
)

install(DIRECTORY
  config
  ckpt
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
  PATTERN "__pycache__" EXCLUDE
  PATTERN "*/__pycache__" EXCLUDE
  PATTERN "__pycache__/*" EXCLUDE
  PATTERN "*.egg-info" EXCLUDE
  PATTERN "*.egg-info/*" EXCLUDE
  PATTERN "*/.DS_Store" EXCLUDE
  PATTERN "*.egg-info/*" EXCLUDE
  PATTERN "build" EXCLUDE
  PATTERN "*/build" EXCLUDE
  PATTERN "build/*" EXCLUDE
  PATTERN "dist" EXCLUDE
  PATTERN "*/dist" EXCLUDE
  PATTERN "dist/*" EXCLUDE
  PATTERN ".DS_Store" EXCLUDE
  PATTERN "*.pyc" EXCLUDE
)

install(CODE "
  file(REMOVE_RECURSE
    \"\$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/${CATKIN_PACKAGE_SHARE_DESTINATION}/thirdparty/aot_tracking\"
    \"\$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/${CATKIN_PACKAGE_SHARE_DESTINATION}/thirdparty/SAM2Long\"
    \"\$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/${CATKIN_PACKAGE_SHARE_DESTINATION}/thirdparty/EVF-SAM\"
    \"\$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/${CATKIN_PACKAGE_SHARE_DESTINATION}/thirdparty/sam_tk\"
  )
")

install(FILES
  thirdparty/sam_tk/sam_tk/__init__.py
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/thirdparty/sam_tk/sam_tk
)

install(FILES
  thirdparty/sam_tk/sam_tk/config/__init__.py
  thirdparty/sam_tk/sam_tk/config/model_args.py
  thirdparty/sam_tk/sam_tk/config/sam2_hiera_s.yaml
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/thirdparty/sam_tk/sam_tk/config
)

install(FILES
  thirdparty/sam_tk/sam_tk/utils/__init__.py
  thirdparty/sam_tk/sam_tk/utils/Tracker.py
  thirdparty/sam_tk/sam_tk/utils/SamTracker.py
  thirdparty/sam_tk/sam_tk/utils/utils.py
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/thirdparty/sam_tk/sam_tk/utils
)

#############
## Testing ##
#############

## Add gtest based cpp test target and link libraries
# catkin_add_gtest(${PROJECT_NAME}-test test/test_ros_edge_sam.cpp)
# if(TARGET ${PROJECT_NAME}-test)
#   target_link_libraries(${PROJECT_NAME}-test ${PROJECT_NAME})
# endif()

## Add folders to be run by python nosetests
# catkin_add_nosetests(test)
