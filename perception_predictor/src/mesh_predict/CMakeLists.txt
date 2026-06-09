cmake_minimum_required(VERSION 3.0.2)
project(mesh_predict)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  geometry_msgs
  image_geometry
  perception_msgs
  rospy
  sensor_msgs
  std_msgs
  tf
  visualization_msgs
)

## System dependencies are found with CMake's conventions
# find_package(Boost REQUIRED COMPONENTS system)


## Uncomment this if the package has a setup.py. This macro ensures
## modules and global scripts declared therein get installed
## See http://ros.org/doc/api/catkin/html/user_guide/setup_dot_py.html
catkin_python_setup()

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
  CATKIN_DEPENDS
    cv_bridge
    geometry_msgs
    image_geometry
    perception_msgs
    rospy
    sensor_msgs
    std_msgs
    tf
    visualization_msgs
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
# include
  ${catkin_INCLUDE_DIRS}
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
# add_executable(${PROJECT_NAME}_node src/ros_edge_sam_node.cpp)

## Rename C++ executable without prefix
## The above recommended prefix causes long target names, the following renames the
## target back to the shorter version for ease of user use
## e.g. "rosrun someones_pkg node" instead of "rosrun someones_pkg someones_pkg_node"
# set_target_properties(${PROJECT_NAME}_node PROPERTIES OUTPUT_NAME node PREFIX "")

## Add cmake target dependencies of the executable
## same as for the library above
# add_dependencies(${PROJECT_NAME}_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

## Specify libraries to link a library or executable target against
# target_link_libraries(${PROJECT_NAME}_node
#   ${catkin_LIBRARIES}
# )

#############
## Install ##
#############

# all install targets should use catkin DESTINATION variables
# See http://ros.org/doc/api/catkin/html/adv_user_guide/variables.html

## Mark executable scripts (Python etc.) for installation
## in contrast to setup.py, you can choose the destination
catkin_install_python(PROGRAMS
  scripts/mesh_predict_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(PROGRAMS
  scripts/mesh_predict_node.py
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/scripts
)

## Mark executables for installation
## See http://docs.ros.org/melodic/api/catkin/html/howto/format1/building_executables.html
# install(TARGETS ${PROJECT_NAME}_node
#   RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

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
set(_mesh_predict_share_dir "${CMAKE_INSTALL_PREFIX}/${CATKIN_PACKAGE_SHARE_DESTINATION}")
set(_mesh_predict_config_dir "${_mesh_predict_share_dir}/config")
install(CODE "
  file(REMOVE
    \"${_mesh_predict_share_dir}/mesh_predict.launch\"
    \"${_mesh_predict_share_dir}/mesh_predict.rviz\"
    \"${_mesh_predict_share_dir}/sim.json\"
    \"${_mesh_predict_share_dir}/camera.json\"
    \"${_mesh_predict_share_dir}/pc_mesh_predict_runtime.yaml\"
    \"${_mesh_predict_share_dir}/pc_mesh_predict_model.yaml\"
    \"${_mesh_predict_config_dir}/mesh_predict.rviz\"
    \"${_mesh_predict_config_dir}/sim.json\"
    \"${_mesh_predict_config_dir}/camera.json\"
  )
")

install(FILES
  launch/mesh_predict.launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)

install(FILES
  config/examples/mesh_predict.rviz
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/config/examples
)

install(FILES
  config/cameras/sim.json
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/config/cameras
)

install(FILES
  config/runtime/pc_mesh_predict_runtime.yaml
  config/runtime/pc_mesh_predict_model.yaml
  config/runtime/pc_mesh_predict_runtime_alternating.yaml
  config/runtime/pc_mesh_predict_model_alternating.yaml
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/config/runtime
)

install(FILES
  resources/checkpoints/README.md
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/resources/checkpoints
)

set(_mesh_predict_runtime_weight_files "")
foreach(_weight_file
  resources/checkpoints/pmp.pt
  resources/checkpoints/snet-n3k-wonormal.pth
)
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${_weight_file}")
    list(APPEND _mesh_predict_runtime_weight_files "${_weight_file}")
  endif()
endforeach()

if(_mesh_predict_runtime_weight_files)
  install(FILES
    ${_mesh_predict_runtime_weight_files}
    DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/resources/checkpoints
  )
endif()

install(FILES
  config/profiles/sim.yaml
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/config/profiles
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
