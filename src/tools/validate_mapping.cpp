/**
 * Landmark Mapping Validation Tool
 * 
 * Validates landmark-to-model vertex mapping file.
 * 
 * NOTE: The correct mapping file is data/bfm_landmark_68.txt which contains
 * accurate correspondences extracted from BFM semantic landmarks.
 * Auto-generation has been removed as it produced incorrect mappings.
 * 
 * Usage:
 *   build/bin/validate_mapping --mapping <path> --model-dir <path> [--min-count <n>]
 * 
 * Example:
 *   build/bin/validate_mapping --mapping data/bfm_landmark_68.txt --model-dir data/model_bfm --min-count 15
 */

#include "model/MorphableModel.h"
#include "alignment/LandmarkMapping.h"
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <fstream>
#include <map>

using namespace face_reconstruction;

int main(int argc, char* argv[]) {
    std::string mapping_path;
    std::string model_dir;
    int min_count = 15;
    bool create_default = false;
    
    // Parse arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--mapping" && i + 1 < argc) {
            mapping_path = argv[++i];
        } else if (arg == "--model-dir" && i + 1 < argc) {
            model_dir = argv[++i];
        } else if (arg == "--min-count" && i + 1 < argc) {
            min_count = std::stoi(argv[++i]);
        } else if (arg == "--create-default") {
            create_default = true;  // Deprecated: will show error message
        } else if (arg == "--help") {
            std::cerr << "Usage: " << argv[0] 
                      << " --mapping <path> --model-dir <path> [--min-count <n>]" << std::endl;
            std::cerr << std::endl;
            std::cerr << "NOTE: Use the correct BFM mapping file: data/bfm_landmark_68.txt" << std::endl;
            std::cerr << "      The --create-default option is deprecated." << std::endl;
            return 1;
        }
    }
    
    if (mapping_path.empty() || model_dir.empty()) {
        std::cerr << "Error: --mapping and --model-dir are required" << std::endl;
        return 1;
    }
    
    // Load model to validate vertex indices
    MorphableModel model;
    if (!model.loadFromFiles(model_dir)) {
        std::cerr << "Error: Failed to load model from " << model_dir << std::endl;
        return 1;
    }
    
    // Try to load existing mapping
    LandmarkMapping mapping;
    bool mapping_exists = mapping.loadFromFile(mapping_path);
    
    if (mapping_exists) {
        // Validate existing mapping
        std::vector<int> mapped_landmarks = mapping.getMappedLandmarks();
        int valid_count = 0;
        
        for (int lm_idx : mapped_landmarks) {
            int vtx_idx = mapping.getModelVertex(lm_idx);
            if (vtx_idx >= 0 && vtx_idx < model.num_vertices) {
                valid_count++;
            }
        }
        
        if (valid_count >= min_count) {
            std::cout << "OK " << valid_count << std::endl;
            return 0;
        } else {
            std::cerr << "ERROR: Mapping has only " << valid_count 
                      << " valid entries (minimum: " << min_count << ")" << std::endl;
            return 1;
        }
    } else if (create_default) {
        // Auto-generation has been removed - it produced incorrect mappings
        // based on percentage-based heuristics that don't match actual facial features
        std::cerr << "ERROR: Auto-generation of mapping file is no longer supported." << std::endl;
        std::cerr << std::endl;
        std::cerr << "The correct mapping file with accurate correspondences is:" << std::endl;
        std::cerr << "  data/bfm_landmark_68.txt" << std::endl;
        std::cerr << std::endl;
        std::cerr << "This file was generated from BFM semantic landmarks using:" << std::endl;
        std::cerr << "  python pipeline/utils/create_bfm_landmark_mapping.py" << std::endl;
        std::cerr << std::endl;
        std::cerr << "To use it, either:" << std::endl;
        std::cerr << "  1. Copy data/bfm_landmark_68.txt to " << mapping_path << std::endl;
        std::cerr << "  2. Or run the pipeline with: --landmark-mapping data/bfm_landmark_68.txt" << std::endl;
        return 1;
    } else {
        std::cerr << "ERROR: Mapping file not found: " << mapping_path << std::endl;
        std::cerr << std::endl;
        std::cerr << "Use the correct BFM mapping file instead:" << std::endl;
        std::cerr << "  --mapping data/bfm_landmark_68.txt" << std::endl;
        return 1;
    }
}

