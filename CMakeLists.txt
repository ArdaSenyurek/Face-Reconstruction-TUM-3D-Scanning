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

# Core source files
set(CORE_SOURCES
    src/data/RGBDFrame.cpp
    src/camera/CameraIntrinsics.cpp
    src/utils/DepthUtils.cpp
    src/model/MorphableModel.cpp
    src/landmarks/LandmarkData.cpp
    src/alignment/Procrustes.cpp
    src/alignment/LandmarkMapping.cpp
    src/alignment/ICP.cpp
    src/rendering/DepthRenderer.cpp
)

# Week 4: Optimization module sources
set(OPTIMIZATION_SOURCES
    src/optimization/Parameters.cpp
    src/optimization/EnergyFunction.cpp
    src/optimization/GaussNewton.cpp
)

# All sources (core + optimization)
set(SOURCES
    ${CORE_SOURCES}
    ${OPTIMIZATION_SOURCES}
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
    include/alignment/ICP.h
    include/rendering/DepthRenderer.h
    include/optimization/Parameters.h
    include/optimization/EnergyFunction.h
    include/optimization/GaussNewton.h
)

# Helper function to create an executable with common settings
function(add_project_executable TARGET_NAME SOURCE_FILE)
    add_executable(${TARGET_NAME}
        ${SOURCE_FILE}
        ${SOURCES}
        ${HEADERS}
    )
    target_link_libraries(${TARGET_NAME} Eigen3::Eigen)
    target_include_directories(${TARGET_NAME} PUBLIC ${CMAKE_SOURCE_DIR}/include)
    
    if(OpenCV_FOUND)
        target_link_libraries(${TARGET_NAME} ${OpenCV_LIBS})
        target_include_directories(${TARGET_NAME} PUBLIC ${OpenCV_INCLUDE_DIRS})
    else()
        target_compile_definitions(${TARGET_NAME} PRIVATE OPENCV_NOT_FOUND)
    endif()
    
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
        target_compile_options(${TARGET_NAME} PRIVATE -Wall -Wextra)
    endif()
    
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        target_compile_definitions(${TARGET_NAME} PRIVATE DEBUG=1)
    endif()
endfunction()

# ======================================
# Pipeline Tools (called by Python pipeline)
# ======================================

# Production face reconstruction tool (Week 4: with optimization)
add_project_executable(face_reconstruction src/tools/face_reconstruction.cpp)

# Pose initialization tool (Procrustes alignment)
add_project_executable(pose_init src/tools/pose_init.cpp)

# Mapping Validation tool
add_executable(validate_mapping
    src/tools/validate_mapping.cpp
    src/model/MorphableModel.cpp
    src/alignment/LandmarkMapping.cpp
    ${HEADERS}
)
target_link_libraries(validate_mapping Eigen3::Eigen)
target_include_directories(validate_mapping PUBLIC ${CMAKE_SOURCE_DIR}/include)

# Analysis tool (RMSE, depth stats, visualization)
add_executable(analysis
    src/tools/analysis.cpp
    src/data/RGBDFrame.cpp
    src/camera/CameraIntrinsics.cpp
    src/utils/DepthUtils.cpp
    ${HEADERS}
)
target_link_libraries(analysis Eigen3::Eigen)
target_include_directories(analysis PUBLIC ${CMAKE_SOURCE_DIR}/include)
if(OpenCV_FOUND)
    target_link_libraries(analysis ${OpenCV_LIBS})
    target_include_directories(analysis PUBLIC ${OpenCV_INCLUDE_DIRS})
else()
    target_compile_definitions(analysis PRIVATE OPENCV_NOT_FOUND)
endif()

# ======================================
# Week 2 Test Executables
# ======================================

# STEP A: Depth Lifting Test
add_project_executable(test_depth_lifting tests/test_depth_lifting.cpp)

# STEP B: PCA Coefficient Evaluation Test (only needs MorphableModel, no OpenCV)
set(PCA_SOURCES
    src/model/MorphableModel.cpp
)
add_executable(test_pca_coeffs
    tests/test_pca_coeffs.cpp
    ${PCA_SOURCES}
)
target_link_libraries(test_pca_coeffs Eigen3::Eigen)
target_include_directories(test_pca_coeffs PUBLIC ${CMAKE_SOURCE_DIR}/include)
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(test_pca_coeffs PRIVATE -Wall -Wextra)
endif()

# STEP C: Landmark I/O Test
add_project_executable(test_landmarks_io tests/test_landmarks_io.cpp)

# STEP D: Pose Initialization Test
add_project_executable(test_pose_init tests/test_pose_init.cpp)

# ======================================
# Week 3 Test Executables
# ======================================

# STEP 1: Landmark Mapping Validation
add_project_executable(test_landmark_mapping tests/test_landmark_mapping.cpp)

# STEP 3: Depth Renderer Test
add_project_executable(test_depth_renderer tests/test_depth_renderer.cpp)

# STEP 4: Depth Residual Computation
add_project_executable(test_depth_residuals tests/test_depth_residuals.cpp)

# STEP 5: ICP Validation
add_project_executable(test_icp_validation tests/test_icp_validation.cpp)

# Print build summary
message(STATUS "")
message(STATUS "=== Build Summary ===")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "OpenCV: ${OpenCV_FOUND}")
message(STATUS "Eigen3: ${Eigen3_FOUND}")
message(STATUS "Output directory: ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
message(STATUS "")
