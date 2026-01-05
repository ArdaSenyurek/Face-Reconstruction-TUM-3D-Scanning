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
    src/alignment/ICP.cpp
    src/rendering/DepthRenderer.cpp
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
)

# Demo executable (component demonstration)
add_executable(${PROJECT_NAME}
    src/demo.cpp
    ${SOURCES}
    ${HEADERS}
)

# Pipeline Tools (called by Python pipeline)
# Production face reconstruction tool
add_executable(face_reconstruction
    src/tools/face_reconstruction.cpp
    ${SOURCES}
    ${HEADERS}
)

# Pose initialization tool (Procrustes alignment)
add_executable(pose_init
    src/tools/pose_init.cpp
    ${SOURCES}
    ${HEADERS}
)

# Week 2 Test Executables
# STEP A: Depth Lifting Test
add_executable(test_depth_lifting
    tests/test_depth_lifting.cpp
    ${SOURCES}
    ${HEADERS}
)

# STEP B: PCA Coefficient Evaluation Test (only needs MorphableModel, no OpenCV)
set(PCA_SOURCES
    src/model/MorphableModel.cpp
)
add_executable(test_pca_coeffs
    tests/test_pca_coeffs.cpp
    ${PCA_SOURCES}
)

# STEP C: Landmark I/O Test
add_executable(test_landmarks_io
    tests/test_landmarks_io.cpp
    ${SOURCES}
    ${HEADERS}
)

# STEP D: Pose Initialization Test
add_executable(test_pose_init
    tests/test_pose_init.cpp
    ${SOURCES}
    ${HEADERS}
)

# Week 3 - STEP 1: Landmark Mapping Validation
add_executable(test_landmark_mapping
    tests/test_landmark_mapping.cpp
    ${SOURCES}
    ${HEADERS}
)

# Week 3 - STEP 3: Depth Renderer Test
add_executable(test_depth_renderer
    tests/test_depth_renderer.cpp
    ${SOURCES}
    ${HEADERS}
)

# Week 3 - STEP 4: Depth Residual Computation
add_executable(test_depth_residuals
    tests/test_depth_residuals.cpp
    ${SOURCES}
    ${HEADERS}
)

# Pipeline Tools: Mapping Validation
add_executable(validate_mapping
    src/tools/validate_mapping.cpp
    src/model/MorphableModel.cpp
    src/alignment/LandmarkMapping.cpp
    ${HEADERS}
)

# Pipeline Tools: Analysis (RMSE, depth stats, visualization)
add_executable(analysis
    src/tools/analysis.cpp
    src/data/RGBDFrame.cpp
    src/camera/CameraIntrinsics.cpp
    src/utils/DepthUtils.cpp
    ${HEADERS}
)

# Week 3 - STEP 5: ICP Validation
add_executable(test_icp_validation
    tests/test_icp_validation.cpp
    ${SOURCES}
    ${HEADERS}
)

# Link libraries
target_link_libraries(${PROJECT_NAME}
    Eigen3::Eigen
)

target_link_libraries(face_reconstruction
    Eigen3::Eigen
)

target_link_libraries(pose_init
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

target_link_libraries(test_landmark_mapping
    Eigen3::Eigen
)

target_link_libraries(test_depth_renderer
    Eigen3::Eigen
)

target_link_libraries(test_depth_residuals
    Eigen3::Eigen
)

target_link_libraries(test_icp_validation
    Eigen3::Eigen
)

target_link_libraries(validate_mapping
    Eigen3::Eigen
)

target_link_libraries(analysis
    Eigen3::Eigen
)

if(OpenCV_FOUND)
    target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
    target_link_libraries(face_reconstruction ${OpenCV_LIBS})
    target_link_libraries(pose_init ${OpenCV_LIBS})
    target_link_libraries(test_depth_lifting ${OpenCV_LIBS})
    target_link_libraries(test_landmarks_io ${OpenCV_LIBS})
    target_link_libraries(test_pose_init ${OpenCV_LIBS})
    target_link_libraries(test_landmark_mapping ${OpenCV_LIBS})
    target_link_libraries(test_depth_renderer ${OpenCV_LIBS})
    target_link_libraries(test_depth_residuals ${OpenCV_LIBS})
    target_link_libraries(test_icp_validation ${OpenCV_LIBS})
    target_link_libraries(analysis ${OpenCV_LIBS})
    target_include_directories(${PROJECT_NAME} PUBLIC ${OpenCV_INCLUDE_DIRS})
    target_include_directories(face_reconstruction PUBLIC ${OpenCV_INCLUDE_DIRS})
    target_include_directories(pose_init PUBLIC ${OpenCV_INCLUDE_DIRS})
    target_include_directories(test_depth_lifting PUBLIC ${OpenCV_INCLUDE_DIRS})
    target_include_directories(test_landmarks_io PUBLIC ${OpenCV_INCLUDE_DIRS})
    target_include_directories(test_pose_init PUBLIC ${OpenCV_INCLUDE_DIRS})
    target_include_directories(test_landmark_mapping PUBLIC ${OpenCV_INCLUDE_DIRS})
    target_include_directories(test_depth_renderer PUBLIC ${OpenCV_INCLUDE_DIRS})
    target_include_directories(test_depth_residuals PUBLIC ${OpenCV_INCLUDE_DIRS})
    target_include_directories(test_icp_validation PUBLIC ${OpenCV_INCLUDE_DIRS})
    target_include_directories(analysis PUBLIC ${OpenCV_INCLUDE_DIRS})
else()
    # Add dummy OpenCV defines to allow code to compile
    target_compile_definitions(${PROJECT_NAME} PRIVATE OPENCV_NOT_FOUND)
    target_compile_definitions(face_reconstruction PRIVATE OPENCV_NOT_FOUND)
    target_compile_definitions(pose_init PRIVATE OPENCV_NOT_FOUND)
    target_compile_definitions(test_depth_lifting PRIVATE OPENCV_NOT_FOUND)
    target_compile_definitions(test_landmarks_io PRIVATE OPENCV_NOT_FOUND)
    target_compile_definitions(test_pose_init PRIVATE OPENCV_NOT_FOUND)
    target_compile_definitions(test_landmark_mapping PRIVATE OPENCV_NOT_FOUND)
    target_compile_definitions(test_depth_renderer PRIVATE OPENCV_NOT_FOUND)
    target_compile_definitions(test_depth_residuals PRIVATE OPENCV_NOT_FOUND)
    target_compile_definitions(test_icp_validation PRIVATE OPENCV_NOT_FOUND)
    target_compile_definitions(analysis PRIVATE OPENCV_NOT_FOUND)
    message(WARNING "Compiling without OpenCV - RGB-D features will not work")
endif()

# Include directories for targets
target_include_directories(${PROJECT_NAME} PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

target_include_directories(face_reconstruction PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

target_include_directories(pose_init PUBLIC
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

target_include_directories(test_landmark_mapping PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

target_include_directories(test_depth_renderer PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

target_include_directories(test_depth_residuals PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

target_include_directories(test_icp_validation PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

# Compiler options
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra)
    target_compile_options(face_reconstruction PRIVATE -Wall -Wextra)
    target_compile_options(pose_init PRIVATE -Wall -Wextra)
    target_compile_options(test_depth_lifting PRIVATE -Wall -Wextra)
    target_compile_options(test_pca_coeffs PRIVATE -Wall -Wextra)
    target_compile_options(test_landmarks_io PRIVATE -Wall -Wextra)
    target_compile_options(test_pose_init PRIVATE -Wall -Wextra)
    target_compile_options(test_landmark_mapping PRIVATE -Wall -Wextra)
    target_compile_options(test_depth_renderer PRIVATE -Wall -Wextra)
    target_compile_options(test_depth_residuals PRIVATE -Wall -Wextra)
    target_compile_options(test_icp_validation PRIVATE -Wall -Wextra)
endif()

# Debug info
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_definitions(${PROJECT_NAME} PRIVATE DEBUG=1)
    target_compile_definitions(face_reconstruction PRIVATE DEBUG=1)
    target_compile_definitions(pose_init PRIVATE DEBUG=1)
    target_compile_definitions(test_depth_lifting PRIVATE DEBUG=1)
    target_compile_definitions(test_pca_coeffs PRIVATE DEBUG=1)
    target_compile_definitions(test_landmarks_io PRIVATE DEBUG=1)
    target_compile_definitions(test_pose_init PRIVATE DEBUG=1)
    target_compile_definitions(test_landmark_mapping PRIVATE DEBUG=1)
    target_compile_definitions(test_depth_renderer PRIVATE DEBUG=1)
    target_compile_definitions(test_depth_residuals PRIVATE DEBUG=1)
    target_compile_definitions(test_icp_validation PRIVATE DEBUG=1)
endif()
