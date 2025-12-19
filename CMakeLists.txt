cmake_minimum_required(VERSION 3.15)
project(FaceReconstruction VERSION 1.0.0 LANGUAGES CXX)

# C++17 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)

# Find required packages
# Try to find OpenCV with different methods
find_package(OpenCV QUIET)
if(NOT OpenCV_FOUND)
    # Try alternative method for OpenCV 4.x
    find_package(OpenCV QUIET PATHS
        /opt/homebrew/opt/opencv
        /usr/local/opt/opencv
        /opt/homebrew/Cellar/opencv/*/lib/cmake/opencv4
        /usr/local/Cellar/opencv/*/lib/cmake/opencv4
        NO_DEFAULT_PATH
    )
endif()
if(NOT OpenCV_FOUND)
    message(WARNING "OpenCV not found. Some features will be disabled.")
    message(STATUS "To install: brew install opencv")
else()
    message(STATUS "Found OpenCV: ${OpenCV_VERSION}")
endif()

find_package(Eigen3 REQUIRED)
if(Eigen3_FOUND)
    message(STATUS "Found Eigen3: ${EIGEN3_VERSION_STRING}")
endif()

# Optional: Ceres (for future optimization)
# find_package(Ceres REQUIRED)

# Source files
set(SOURCES
    src/data/RGBDFrame.cpp
    src/camera/CameraIntrinsics.cpp
    src/utils/DepthUtils.cpp
    src/model/MorphableModel.cpp
    src/landmarks/LandmarkData.cpp
    src/alignment/Procrustes.cpp
    src/alignment/LandmarkMapping.cpp
)

# Header files
set(HEADERS
    include/data/RGBDFrame.h
    include/camera/CameraIntrinsics.h
    include/utils/DepthUtils.h
    include/model/MorphableModel.h
    include/landmarks/LandmarkData.h
    include/alignment/Procrustes.h
    include/alignment/LandmarkMapping.h
)

# Main executable
add_executable(${PROJECT_NAME}
    src/main.cpp
    ${SOURCES}
    ${HEADERS}
)

# Test executable for real data
add_executable(test_real_data
    src/test_real_data.cpp
    ${SOURCES}
    ${HEADERS}
)

# Week 2 Test Executables
# STEP A: Depth Lifting Test
add_executable(test_depth_lifting
    src/test_depth_lifting.cpp
    ${SOURCES}
    ${HEADERS}
)

# STEP B: PCA Coefficient Evaluation Test (only needs MorphableModel, no OpenCV)
set(PCA_SOURCES
    src/model/MorphableModel.cpp
)
add_executable(test_pca_coeffs
    src/test_pca_coeffs.cpp
    ${PCA_SOURCES}
)

# STEP C: Landmark I/O Test
add_executable(test_landmarks_io
    src/test_landmarks_io.cpp
    ${SOURCES}
    ${HEADERS}
)

# STEP D: Pose Initialization Test
add_executable(test_pose_init
    src/test_pose_init.cpp
    ${SOURCES}
    ${HEADERS}
)

# Link libraries
target_link_libraries(${PROJECT_NAME}
    Eigen3::Eigen
)

target_link_libraries(test_real_data
    Eigen3::Eigen
)

target_link_libraries(test_depth_lifting
    Eigen3::Eigen
)

target_link_libraries(test_pca_coeffs
    Eigen3::Eigen
)

target_link_libraries(test_landmarks_io
    Eigen3::Eigen
)

target_link_libraries(test_pose_init
    Eigen3::Eigen
)

if(OpenCV_FOUND)
    target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
    target_link_libraries(test_real_data ${OpenCV_LIBS})
    target_link_libraries(test_depth_lifting ${OpenCV_LIBS})
    target_link_libraries(test_landmarks_io ${OpenCV_LIBS})
    target_link_libraries(test_pose_init ${OpenCV_LIBS})
    target_include_directories(${PROJECT_NAME} PUBLIC ${OpenCV_INCLUDE_DIRS})
    target_include_directories(test_real_data PUBLIC ${OpenCV_INCLUDE_DIRS})
    target_include_directories(test_depth_lifting PUBLIC ${OpenCV_INCLUDE_DIRS})
    target_include_directories(test_landmarks_io PUBLIC ${OpenCV_INCLUDE_DIRS})
    target_include_directories(test_pose_init PUBLIC ${OpenCV_INCLUDE_DIRS})
else()
    # Add dummy OpenCV defines to allow code to compile
    target_compile_definitions(${PROJECT_NAME} PRIVATE OPENCV_NOT_FOUND)
    target_compile_definitions(test_real_data PRIVATE OPENCV_NOT_FOUND)
    target_compile_definitions(test_depth_lifting PRIVATE OPENCV_NOT_FOUND)
    target_compile_definitions(test_landmarks_io PRIVATE OPENCV_NOT_FOUND)
    target_compile_definitions(test_pose_init PRIVATE OPENCV_NOT_FOUND)
    message(WARNING "Compiling without OpenCV - RGB-D features will not work")
endif()

# Include directories for targets
target_include_directories(${PROJECT_NAME} PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

target_include_directories(test_real_data PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

target_include_directories(test_depth_lifting PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

target_include_directories(test_pca_coeffs PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)
# test_pca_coeffs doesn't need OpenCV

target_include_directories(test_landmarks_io PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

target_include_directories(test_pose_init PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

# Compiler options
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra)
    target_compile_options(test_real_data PRIVATE -Wall -Wextra)
    target_compile_options(test_depth_lifting PRIVATE -Wall -Wextra)
    target_compile_options(test_pca_coeffs PRIVATE -Wall -Wextra)
    target_compile_options(test_landmarks_io PRIVATE -Wall -Wextra)
    target_compile_options(test_pose_init PRIVATE -Wall -Wextra)
endif()

# Debug info
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_definitions(${PROJECT_NAME} PRIVATE DEBUG=1)
    target_compile_definitions(test_real_data PRIVATE DEBUG=1)
    target_compile_definitions(test_depth_lifting PRIVATE DEBUG=1)
    target_compile_definitions(test_pca_coeffs PRIVATE DEBUG=1)
    target_compile_definitions(test_landmarks_io PRIVATE DEBUG=1)
    target_compile_definitions(test_pose_init PRIVATE DEBUG=1)
endif()
