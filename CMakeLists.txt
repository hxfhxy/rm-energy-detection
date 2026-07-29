cmake_minimum_required(VERSION 3.8)
project(energy_unit_pose)
set(OpenGL_GL_PREFERENCE GLVND)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
# C++17 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# OpenVINO RPATH
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH "/home/hzy/.local/lib/python3.10/site-packages/openvino/libs")

# 编译优化
set(CMAKE_BUILD_TYPE Release)
add_compile_options(-O2 -Wall -Wextra)

# 查找依赖
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(OpenCV 4 REQUIRED)
find_package(Eigen3 REQUIRED)
# ================= 新增 M3T 的依赖 =================
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(glfw3 REQUIRED)

# 告诉 CMake 带着 M3T 这个子文件夹一起编译
add_subdirectory(M3T)

# OpenVINO
set(OpenVINO_DIR "/home/hzy/.local/lib/python3.10/site-packages/openvino/cmake")
find_package(OpenVINO REQUIRED)
find_package(PCL REQUIRED COMPONENTS common features filters)

# 包含目录
include_directories(
  include
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
)

# PCL 定义
add_definitions(${PCL_DEFINITIONS})

# 库文件
add_library(energy_unit_lib SHARED
  src/energy_unit_detector.cpp
  src/pose_estimator.cpp
  src/arm_transform.cpp
)
target_link_libraries(energy_unit_lib
  m3t
  OpenGL::GL
  GLEW::GLEW
  glfw
  ${OpenCV_LIBS}
  ${PCL_LIBRARIES}
  Eigen3::Eigen
  openvino::runtime
)
ament_target_dependencies(energy_unit_lib
  rclcpp
  sensor_msgs
  geometry_msgs
  cv_bridge
)

# 可执行文件
add_executable(energy_unit_node src/energy_unit_node.cpp)
target_link_libraries(energy_unit_node
  energy_unit_lib
  ${OpenCV_LIBS}
  ${PCL_LIBRARIES}
  Eigen3::Eigen
)
ament_target_dependencies(energy_unit_node
  rclcpp
  sensor_msgs
  geometry_msgs
  std_msgs
  cv_bridge
  image_transport
  tf2
  tf2_ros
  tf2_geometry_msgs
)

# 安装
install(TARGETS energy_unit_node energy_unit_lib
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)
install(DIRECTORY include/
  DESTINATION include
)
install(DIRECTORY launch config model
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
