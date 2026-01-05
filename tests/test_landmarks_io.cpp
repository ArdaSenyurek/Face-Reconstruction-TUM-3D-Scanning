/**
 * STEP C: Landmark Detection End-to-End Integration Test
 * 
 * Goal: From a Biwi RGB image -> landmarks file -> load in C++
 * 
 * Tests:
 *   - Load landmarks from file (TXT or JSON)
 *   - Verify count > 0 and within image bounds
 *   - Print summary statistics
 * 
 * Usage:
 *   bin/test_landmarks_io <rgb_path> <landmarks_path>
 * 
 * Example:
 *   bin/test_landmarks_io data/biwi_person01/rgb/frame_00000.png \
 *                          data/biwi_person01/landmarks/frame_00000.txt
 */

#include "data/RGBDFrame.h"
#include "landmarks/LandmarkData.h"
#include <iostream>
#include <iomanip>
#include <cmath>

using namespace face_reconstruction;

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <rgb_path> <landmarks_path>" << std::endl;
        std::cerr << "\nExample:" << std::endl;
        std::cerr << "  " << argv[0] 
                  << " data/biwi_person01/rgb/frame_00000.png \\" << std::endl;
        std::cerr << "     data/biwi_person01/landmarks/frame_00000.txt" << std::endl;
        std::cerr << "\nNote: Generate landmarks first using:" << std::endl;
        std::cerr << "  python scripts/detect_landmarks.py --image <rgb_path> --output <landmarks_path>" << std::endl;
        return 1;
    }
    
    std::string rgb_path = argv[1];
    std::string landmarks_path = argv[2];
    
    std::cout << "=== Landmark I/O Test ===" << std::endl;
    std::cout << "RGB: " << rgb_path << std::endl;
    std::cout << "Landmarks: " << landmarks_path << std::endl;
    std::cout << std::endl;
    
    // Load RGB image
    RGBDFrame frame;
    if (!frame.loadRGB(rgb_path)) {
        std::cerr << "Failed to load RGB image" << std::endl;
        return 1;
    }
    
    std::cout << "--- RGB Image Statistics ---" << std::endl;
    std::cout << "Size: " << frame.width() << " x " << frame.height() << std::endl;
    std::cout << std::endl;
    
    // Load landmarks
    LandmarkData landmarks;
    
    // Determine file type from extension
    std::string ext = landmarks_path.substr(landmarks_path.find_last_of(".") + 1);
    bool loaded = false;
    
    if (ext == "json" || ext == "JSON") {
        std::cout << "Loading landmarks from JSON..." << std::endl;
        loaded = landmarks.loadFromJSON(landmarks_path);
    } else {
        std::cout << "Loading landmarks from TXT..." << std::endl;
        loaded = landmarks.loadFromTXT(landmarks_path);
    }
    
    if (!loaded) {
        std::cerr << "Failed to load landmarks from: " << landmarks_path << std::endl;
        std::cerr << "\nHint: Generate landmarks first using:" << std::endl;
        std::cerr << "  python scripts/detect_landmarks.py --image " << rgb_path 
                  << " --output " << landmarks_path << std::endl;
        return 1;
    }
    
    std::cout << "Successfully loaded " << landmarks.size() << " landmarks" << std::endl;
    std::cout << std::endl;
    
    // Verify landmark count
    if (landmarks.size() == 0) {
        std::cerr << "ERROR: No landmarks found in file!" << std::endl;
        return 1;
    }
    
    std::cout << "--- Landmark Statistics ---" << std::endl;
    std::cout << "Total landmarks: " << landmarks.size() << std::endl;
    
    // Check if landmarks are within image bounds
    int in_bounds = 0;
    int out_of_bounds = 0;
    double min_x = std::numeric_limits<double>::max();
    double max_x = std::numeric_limits<double>::lowest();
    double min_y = std::numeric_limits<double>::max();
    double max_y = std::numeric_limits<double>::lowest();
    
    for (size_t i = 0; i < landmarks.size(); ++i) {
        const auto& lm = landmarks[i];
        
        if (lm.x < min_x) min_x = lm.x;
        if (lm.x > max_x) max_x = lm.x;
        if (lm.y < min_y) min_y = lm.y;
        if (lm.y > max_y) max_y = lm.y;
        
        if (lm.x >= 0 && lm.x < frame.width() && 
            lm.y >= 0 && lm.y < frame.height()) {
            in_bounds++;
        } else {
            out_of_bounds++;
        }
    }
    
    std::cout << "Landmarks in bounds: " << in_bounds << " / " << landmarks.size() << std::endl;
    if (out_of_bounds > 0) {
        std::cout << "WARNING: " << out_of_bounds << " landmarks are out of bounds!" << std::endl;
    }
    
    std::cout << "Bounding box: [" << min_x << ", " << max_x << "] x [" 
              << min_y << ", " << max_y << "]" << std::endl;
    std::cout << std::endl;
    
    // Check model indices
    std::vector<int> model_indices = landmarks.getModelIndices();
    int valid_model_indices = 0;
    for (int idx : model_indices) {
        if (idx >= 0) {
            valid_model_indices++;
        }
    }
    
    std::cout << "--- Model Index Mapping ---" << std::endl;
    std::cout << "Landmarks with valid model indices: " << valid_model_indices 
              << " / " << landmarks.size() << std::endl;
    if (valid_model_indices == 0) {
        std::cout << "NOTE: No model indices mapped. This is OK for initial testing." << std::endl;
        std::cout << "      You'll need to create a landmark mapping file for pose initialization." << std::endl;
    }
    std::cout << std::endl;
    
    // Print first few landmarks as sample
    std::cout << "--- Sample Landmarks (first 5) ---" << std::endl;
    int num_samples = std::min(5, static_cast<int>(landmarks.size()));
    for (int i = 0; i < num_samples; ++i) {
        const auto& lm = landmarks[i];
        std::cout << "  [" << i << "] (" << std::fixed << std::setprecision(2) 
                  << lm.x << ", " << lm.y << ")";
        if (lm.model_index >= 0) {
            std::cout << " -> model vertex " << lm.model_index;
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
    
    // Summary
    std::cout << "=== Test Summary ===" << std::endl;
    bool all_ok = true;
    
    if (landmarks.size() == 0) {
        std::cerr << "✗ FAIL: No landmarks loaded" << std::endl;
        all_ok = false;
    } else {
        std::cout << "✓ PASS: " << landmarks.size() << " landmarks loaded" << std::endl;
    }
    
    if (out_of_bounds > landmarks.size() / 2) {
        std::cerr << "✗ FAIL: Too many landmarks out of bounds (" << out_of_bounds << ")" << std::endl;
        all_ok = false;
    } else if (out_of_bounds > 0) {
        std::cout << "⚠ WARNING: Some landmarks out of bounds, but acceptable" << std::endl;
    } else {
        std::cout << "✓ PASS: All landmarks within image bounds" << std::endl;
    }
    
    std::cout << std::endl;
    
    if (all_ok) {
        std::cout << "✓ Landmark I/O test completed successfully!" << std::endl;
        std::cout << "\nNext steps:" << std::endl;
        std::cout << "  1. Ensure landmarks are detected correctly on Biwi frames" << std::endl;
        std::cout << "  2. Create landmark-to-model mapping file for pose initialization" << std::endl;
        return 0;
    } else {
        std::cerr << "✗ Some checks failed!" << std::endl;
        return 1;
    }
}

