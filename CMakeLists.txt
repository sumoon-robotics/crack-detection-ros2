cmake_minimum_required(VERSION 3.8)
project(crack_livo_fusion)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(message_filters REQUIRED)

ament_python_install_package(${PROJECT_NAME})

install(PROGRAMS
  scripts/crack_segmentation_node.py
  scripts/crack_fusion_node.py
  scripts/crack_measurement_node.py
  scripts/crack_map_accumulator_node.py
  scripts/range_limited_cloud_node.py
  door_detection/door_yolo_node.py
  door_detection/door_dimension_node.py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  launch
  config
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
