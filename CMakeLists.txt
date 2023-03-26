CMAKE_MINIMUM_REQUIRED(VERSION 3.16.3)
PROJECT(voxel_lio)

# debug mdoe.
SET(CMAKE_BUILD_TYPE "Debug")

SET(CMAKE_CXX_STANDARD_REQUIRED ON)
SET(CMAKE_CXX_EXTENSIONS OFF)
SET(CMAKE_CXX_STANDARD 17)

# Don't change the values of CMAKE_CXX_FLAGS,
# as those are mainly intended to be changed through toolchain files or the cache.
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -fexceptions -g -ggdb")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 ${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS}")

ADD_DEFINITIONS(-DROOT_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/\")

FIND_PACKAGE(catkin REQUIRED COMPONENTS
  geometry_msgs
  nav_msgs
  pcl_ros
  roscpp
  sensor_msgs
  tf
  livox_ros_driver
  message_generation
)

FIND_PACKAGE(TBB REQUIRED)

FIND_PACKAGE(Eigen3 REQUIRED)

SET(PCL_DIR /usr/local/share/pcl-1.13)
FIND_PACKAGE(PCL 1.13 REQUIRED)
SET(PCL_INCLUDE_DIRS /usr/local/include/pcl-1.13)
add_definitions(${PCL_DEFINITIONS})

FIND_PATH(LOGGER_INCLUDE_DIR NAMES file_logger.h PATHS /home/neal/usr/include/file_logger)
FIND_LIBRARY(LOGGER_LIBRARIES file_logger HINTS /home/neal/usr/lib)

ADD_MESSAGE_FILES(
  FILES
  Pose6D.msg
)

GENERATE_MESSAGES(
  DEPENDENCIES
  geometry_msgs
)

CATKIN_PACKAGE(
  INCLUDE_DIRS include
  CATKIN_DEPENDS geometry_msgs nav_msgs pcl_ros roscpp sensor_msgs tf livox_ros_driver message_generation
  DEPENDS EIGEN3 PCL
)

INCLUDE_DIRECTORIES(
  include
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${PCL_INCLUDE_DIRS}
  ${LOGGER_INCLUDE_DIR}
)

ADD_EXECUTABLE(lio_node src/laserMapping.cpp src/preprocess.cpp src/IMU_Processing.cpp src/use-ikfom.cpp src/common_lib.cpp)

TARGET_LINK_LIBRARIES(lio_node ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${LOGGER_LIBRARIES} tbb)
