#!/usr/bin/env python3
"""
Landmark detection script for face reconstruction.
Uses dlib or MediaPipe to detect facial landmarks and saves them in a format
compatible with the C++ face reconstruction pipeline.
"""

import argparse
import cv2
import numpy as np
import json
import sys
import os

try:
    import dlib
    DLIB_AVAILABLE = True
except ImportError:
    DLIB_AVAILABLE = False
    print("Warning: dlib not available. Install with: pip install dlib")

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("Warning: mediapipe not available. Install with: pip install mediapipe")


def detect_landmarks_dlib(image_path, predictor_path=None):
    """
    Detect facial landmarks using dlib.
    
    Args:
        image_path: Path to input image
        predictor_path: Path to dlib shape predictor file (e.g., shape_predictor_68_face_landmarks.dat)
                       If None, uses default 68-point model
    
    Returns:
        List of (x, y) tuples for landmark points
    """
    if not DLIB_AVAILABLE:
        raise RuntimeError("dlib is not available")
    
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Initialize face detector and landmark predictor
    detector = dlib.get_frontal_face_detector()
    
    if predictor_path is None:
        # Try to find default predictor file
        default_paths = [
            "shape_predictor_68_face_landmarks.dat",
            "models/shape_predictor_68_face_landmarks.dat",
            "data/models/shape_predictor_68_face_landmarks.dat",
        ]
        predictor_path = None
        for path in default_paths:
            if os.path.exists(path):
                predictor_path = path
                break
        
        if predictor_path is None:
            raise ValueError(
                "Predictor file not found. Download from: "
                "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
            )
    
    predictor = dlib.shape_predictor(predictor_path)
    
    # Detect faces
    faces = detector(gray)
    if len(faces) == 0:
        raise ValueError("No face detected in image")
    
    if len(faces) > 1:
        print(f"Warning: {len(faces)} faces detected, using the largest one")
    
    # Use the largest face
    face = max(faces, key=lambda rect: rect.width() * rect.height())
    
    # Detect landmarks
    landmarks = predictor(gray, face)
    
    # Convert to list of (x, y) tuples
    points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(68)]
    
    return points, (face.left(), face.top(), face.width(), face.height())


def detect_landmarks_mediapipe(image_path):
    """
    Detect facial landmarks using MediaPipe.
    
    Args:
        image_path: Path to input image
    
    Returns:
        List of (x, y) tuples for landmark points
    """
    if not MEDIAPIPE_AVAILABLE:
        raise RuntimeError("MediaPipe is not available")
    
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = rgb_img.shape[:2]
    
    # Initialize MediaPipe Face Mesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    )
    
    results = face_mesh.process(rgb_img)
    
    if not results.multi_face_landmarks:
        raise ValueError("No face detected in image")
    
    # Get landmarks (468 points for full face mesh)
    # Use first 68 points for compatibility with 68-point models
    face_landmarks = results.multi_face_landmarks[0]
    
    # Map to image coordinates
    points = []
    landmark_indices_68 = [
        # Key facial points (approximate mapping from MediaPipe 468 to 68)
        10, 151, 9, 175, 152, 6,  # Jaw line
        33, 7, 163, 144, 145, 153, 154, 155, 133,  # Left eyebrow
        362, 382, 381, 380, 374, 373, 390, 249,  # Right eyebrow
        468, 469, 470, 471, 472,  # Nose
        61, 146, 91, 181, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318,  # Eyes
        78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308,  # More eye points
        0, 11, 12, 13, 14, 15, 16, 17, 18,  # Lower face
    ]
    
    # Use first 68 points from MediaPipe (or map specific points)
    for i in range(min(68, len(face_landmarks.landmark))):
        lm = face_landmarks.landmark[i]
        x = int(lm.x * w)
        y = int(lm.y * h)
        points.append((x, y))
    
    return points, None


def save_landmarks_txt(landmarks, output_path, model_indices=None):
    """Save landmarks to TXT file"""
    with open(output_path, 'w') as f:
        for i, (x, y) in enumerate(landmarks):
            model_idx = model_indices[i] if model_indices else -1
            f.write(f"{x} {y} {model_idx}\n")
    print(f"Saved {len(landmarks)} landmarks to {output_path}")


def save_landmarks_json(landmarks, output_path, model_indices=None):
    """Save landmarks to JSON file"""
    data = {
        "landmarks": [
            {
                "x": float(x),
                "y": float(y),
                "model_index": int(model_indices[i] if model_indices else -1)
            }
            for i, (x, y) in enumerate(landmarks)
        ]
    }
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Saved {len(landmarks)} landmarks to {output_path}")


def map_to_model_indices(landmarks, method='dlib'):
    """
    Map detected landmarks to model vertex indices.
    This is a placeholder - in practice, you need to know the correspondence
    between landmark points and your morphable model vertices.
    
    Args:
        landmarks: List of (x, y) landmark points
        method: Detection method ('dlib' or 'mediapipe')
    
    Returns:
        List of model vertex indices (or -1 if unknown)
    """
    # For now, return -1 for all (unknown correspondence)
    # In practice, you would have a mapping file or predefined correspondence
    return [-1] * len(landmarks)


def main():
    parser = argparse.ArgumentParser(
        description='Detect facial landmarks from RGB image',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using dlib (requires shape_predictor_68_face_landmarks.dat)
  python detect_landmarks.py --image data/rgb.png --method dlib \\
      --predictor shape_predictor_68_face_landmarks.dat \\
      --output data/landmarks.txt
  
  # Using MediaPipe
  python detect_landmarks.py --image data/rgb.png --method mediapipe \\
      --output data/landmarks.json
        """
    )
    
    parser.add_argument('--image', type=str, required=True,
                       help='Path to input RGB image')
    parser.add_argument('--output', type=str, required=True,
                       help='Output landmarks file (TXT or JSON)')
    parser.add_argument('--method', choices=['dlib', 'mediapipe'], default='mediapipe',
                       help='Landmark detection method (default: mediapipe)')
    parser.add_argument('--predictor', type=str,
                       help='Path to dlib shape predictor file (for dlib method)')
    parser.add_argument('--format', choices=['txt', 'json', 'auto'], default='auto',
                       help='Output format (default: auto - determined from extension)')
    parser.add_argument('--visualize', action='store_true',
                       help='Show detected landmarks on image')
    
    args = parser.parse_args()
    
    # Detect landmarks
    try:
        if args.method == 'dlib':
            if not DLIB_AVAILABLE:
                print("Error: dlib is not available")
                print("Install with: pip install dlib")
                sys.exit(1)
            landmarks, bbox = detect_landmarks_dlib(args.image, args.predictor)
        else:  # mediapipe
            if not MEDIAPIPE_AVAILABLE:
                print("Error: MediaPipe is not available")
                print("Install with: pip install mediapipe")
                sys.exit(1)
            landmarks, bbox = detect_landmarks_mediapipe(args.image)
        
        print(f"Detected {len(landmarks)} landmarks")
        
        # Map to model indices (placeholder - returns -1 for all)
        model_indices = map_to_model_indices(landmarks, args.method)
        
        # Determine output format
        if args.format == 'auto':
            ext = os.path.splitext(args.output)[1].lower()
            if ext == '.json':
                format_type = 'json'
            else:
                format_type = 'txt'
        else:
            format_type = args.format
        
        # Save landmarks
        if format_type == 'json':
            save_landmarks_json(landmarks, args.output, model_indices)
        else:
            save_landmarks_txt(landmarks, args.output, model_indices)
        
        # Visualize if requested
        if args.visualize:
            img = cv2.imread(args.image)
            for i, (x, y) in enumerate(landmarks):
                cv2.circle(img, (int(x), int(y)), 2, (0, 255, 0), -1)
                cv2.putText(img, str(i), (int(x)+3, int(y)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
            
            cv2.imshow('Detected Landmarks', img)
            print("Press any key to close the window...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        print(f"Success! Landmarks saved to {args.output}")
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
