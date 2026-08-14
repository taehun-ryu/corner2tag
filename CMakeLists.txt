cmake_minimum_required(VERSION 3.8)
project(corner2tag)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(CORNER2TAG_STANDALONE "Build corner2tag without ROS/ament" OFF)

find_package(ament_cmake QUIET)

find_package(OpenCV REQUIRED)
find_package(Ceres REQUIRED)
find_package(HDF5 REQUIRED COMPONENTS C)

if (CORNER2TAG_STANDALONE OR NOT ament_cmake_FOUND)
  set(CORNER2TAG_USE_AMENT OFF)
else()
  set(CORNER2TAG_USE_AMENT ON)
endif()
# ---------------------------
# Library
# ---------------------------
add_library(corner2tag
  src/core/cm/iwe_gaussian.cpp
  src/core/cm/cm_velocity_2d.cpp
  src/core/cm/cm_tracker_2d.cpp
  src/core/detection/patch_extractor.cpp
  src/core/detection/corner_init.cpp
  src/core/detection/corner_refinement.cpp
  src/core/postprocess/checkerboard_filter.cpp
  src/core/postprocess/checkerboard_order.cpp
  src/core/postprocess/checkerboard_validity.cpp
  src/core/calib/checkerboard_calibration.cpp
  src/viz/corner_detection_viz.cpp
  src/viz/calibration_report.cpp
  src/io/h5_events.cpp
  src/io/calibration_config.cpp
  src/io/calibration_output.cpp
  src/io/detection_output.cpp
  src/viz/event_render.cpp
)

target_include_directories(corner2tag PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

target_include_directories(corner2tag PRIVATE
  ${HDF5_INCLUDE_DIRS}
)

target_link_libraries(corner2tag
  ${OpenCV_LIBS}
  Ceres::ceres
  ${HDF5_LIBRARIES}
)

# Export include dirs + deps so downstream packages can link cleanly
if (CORNER2TAG_USE_AMENT)
  ament_export_include_directories(include)
  ament_export_libraries(corner2tag)
  ament_target_dependencies(corner2tag
    OpenCV
    Ceres
  )
endif()

# ---------------------------
# Install
# ---------------------------
install(DIRECTORY include/
  DESTINATION include
)

install(TARGETS corner2tag
  EXPORT export_${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

if (CORNER2TAG_USE_AMENT)
  ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)
endif()

# ---------------------------
# Executables
# ---------------------------
function(add_corner2tag_app target_name source_path)
  add_executable(${target_name}
    ${source_path}
  )

  target_include_directories(${target_name} PRIVATE
    ${HDF5_INCLUDE_DIRS}
  )

  target_link_libraries(${target_name}
    corner2tag
    ${HDF5_LIBRARIES}
    ${OpenCV_LIBS}
  )

  install(TARGETS ${target_name}
    RUNTIME DESTINATION lib/${PROJECT_NAME}
  )
endfunction()

add_corner2tag_app(checkerboard
  src/app/checkerboard.cpp
)

include(CTest)
if(BUILD_TESTING)
  add_executable(fixed_window_tests
    tests/fixed_window_tests.cpp
  )
  target_link_libraries(fixed_window_tests corner2tag)
  add_test(NAME fixed_window_tests COMMAND fixed_window_tests)
endif()

if (CORNER2TAG_USE_AMENT)
  ament_package()
endif()
