#!/usr/bin/env python3
"""
measurement_extractor.py - ENHANCED with foot measurements

IMPROVEMENTS:
- ✅ Added foot_length and foot_width extraction
- ✅ Compatible with scale calibration in vision.py
- ✅ Maintains all original measurements (chest, waist, hips, etc.)
- ✅ Works with existing merge_measurements and validate_measurement_sanity
"""

from typing import Dict, Optional, Tuple
import numpy as np
import math


def calculate_3d_distance(point1, point2, img_height: int, img_width: int) -> float:
    """Calculate pixel distance between two landmark points."""
    p1 = np.array([point1.x * img_width, point1.y * img_height, point1.z * img_width])
    p2 = np.array([point2.x * img_width, point2.y * img_height, point2.z * img_width])
    return float(np.linalg.norm(p2 - p1))


def pixel_to_meters(pixel_distance: float, reference_height_m: float,
                    measured_height_px: float) -> float:
    """Convert pixel measurements to real-world meters."""
    if measured_height_px <= 0:
        return 0.0
    meters_per_pixel = reference_height_m / measured_height_px
    return pixel_distance * meters_per_pixel


def apply_perspective_correction(measurement_m: float, camera_distance_m: float,
                                 angle_deg: float) -> float:
    """Correct measurement for camera distance and angle."""
    if camera_distance_m < 1.5:
        distance_factor = 1.0 + (1.5 - camera_distance_m) * 0.1
    elif camera_distance_m > 3.5:
        distance_factor = 1.0 - (camera_distance_m - 3.5) * 0.05
    else:
        distance_factor = 1.0
    
    angle_rad = math.radians(abs(angle_deg))
    angle_factor = 1.0 / math.cos(angle_rad) if angle_rad < math.radians(30) else 1.0
    
    return measurement_m * distance_factor * angle_factor


def extract_shoulder_width(pose_landmarks, img_h: int, img_w: int,
                           height_m: float, height_px: float,
                           camera_distance: float = 2.5,
                           roll_deg: float = 0.0) -> Optional[float]:
    """Extract shoulder width from pose landmarks."""
    try:
        import mediapipe as mp
        mp_pose = mp.solutions.pose
        
        left_shoulder = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        
        shoulder_px = calculate_3d_distance(left_shoulder, right_shoulder, img_h, img_w)
        shoulder_m = pixel_to_meters(shoulder_px, height_m, height_px)
        shoulder_m = apply_perspective_correction(shoulder_m, camera_distance, roll_deg)
        
        return shoulder_m
    except Exception as e:
        print(f"[MEASURE] Warning: Could not extract shoulder width: {e}")
        return None


def extract_chest_circumference(pose_landmarks, img_h: int, img_w: int,
                                height_m: float, height_px: float,
                                camera_distance: float = 2.5) -> Optional[float]:
    """Estimate chest circumference from shoulder and side landmarks."""
    try:
        import mediapipe as mp
        mp_pose = mp.solutions.pose
        
        shoulder_width_m = extract_shoulder_width(pose_landmarks, img_h, img_w,
                                                  height_m, height_px, camera_distance)
        if not shoulder_width_m:
            return None
        
        left_shoulder = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        
        avg_z = (left_shoulder.z + right_shoulder.z) / 2.0
        depth_ratio = 0.55 + abs(avg_z) * 0.1
        chest_depth_m = shoulder_width_m * depth_ratio
        
        a = shoulder_width_m / 2.0
        b = chest_depth_m / 2.0
        circumference = math.pi * (3 * (a + b) - math.sqrt((3*a + b) * (a + 3*b)))
        
        return circumference
    except Exception as e:
        print(f"[MEASURE] Warning: Could not extract chest: {e}")
        return None


def extract_waist_circumference(pose_landmarks, img_h: int, img_w: int,
                                height_m: float, height_px: float,
                                camera_distance: float = 2.5) -> Optional[float]:
    """Estimate waist circumference from hip landmarks."""
    try:
        import mediapipe as mp
        mp_pose = mp.solutions.pose
        
        left_hip = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
        
        hip_width_px = calculate_3d_distance(left_hip, right_hip, img_h, img_w)
        hip_width_m = pixel_to_meters(hip_width_px, height_m, height_px)
        hip_width_m = apply_perspective_correction(hip_width_m, camera_distance, 0)
        
        waist_width_m = hip_width_m * 0.85
        
        avg_z = (left_hip.z + right_hip.z) / 2.0
        depth_ratio = 0.50 + abs(avg_z) * 0.1
        waist_depth_m = waist_width_m * depth_ratio
        
        a = waist_width_m / 2.0
        b = waist_depth_m / 2.0
        circumference = math.pi * (3 * (a + b) - math.sqrt((3*a + b) * (a + 3*b)))
        
        return circumference
    except Exception as e:
        print(f"[MEASURE] Warning: Could not extract waist: {e}")
        return None


def extract_hip_circumference(pose_landmarks, img_h: int, img_w: int,
                              height_m: float, height_px: float,
                              camera_distance: float = 2.5) -> Optional[float]:
    """Estimate hip circumference from hip landmarks."""
    try:
        import mediapipe as mp
        mp_pose = mp.solutions.pose
        
        left_hip = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
        
        hip_width_px = calculate_3d_distance(left_hip, right_hip, img_h, img_w)
        hip_width_m = pixel_to_meters(hip_width_px, height_m, height_px)
        hip_width_m = apply_perspective_correction(hip_width_m, camera_distance, 0)
        
        avg_z = (left_hip.z + right_hip.z) / 2.0
        depth_ratio = 0.60 + abs(avg_z) * 0.1
        hip_depth_m = hip_width_m * depth_ratio
        
        a = hip_width_m / 2.0
        b = hip_depth_m / 2.0
        circumference = math.pi * (3 * (a + b) - math.sqrt((3*a + b) * (a + 3*b)))
        
        return circumference
    except Exception as e:
        print(f"[MEASURE] Warning: Could not extract hips: {e}")
        return None


def extract_inseam_length(pose_landmarks, img_h: int, img_w: int,
                          height_m: float, height_px: float,
                          camera_distance: float = 2.5) -> Optional[float]:
    """Extract inseam (crotch to ankle) length."""
    try:
        import mediapipe as mp
        mp_pose = mp.solutions.pose
        
        left_hip = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
        left_ankle = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE]
        
        inseam_px = calculate_3d_distance(left_hip, left_ankle, img_h, img_w)
        inseam_m = pixel_to_meters(inseam_px, height_m, height_px)
        inseam_m = apply_perspective_correction(inseam_m, camera_distance, 0)
        
        return inseam_m
    except Exception as e:
        print(f"[MEASURE] Warning: Could not extract inseam: {e}")
        return None


def extract_arm_length(pose_landmarks, img_h: int, img_w: int,
                       height_m: float, height_px: float,
                       camera_distance: float = 2.5) -> Optional[float]:
    """Extract arm length (shoulder to wrist)."""
    try:
        import mediapipe as mp
        mp_pose = mp.solutions.pose
        
        left_shoulder = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        left_wrist = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
        
        arm_px = calculate_3d_distance(left_shoulder, left_wrist, img_h, img_w)
        arm_m = pixel_to_meters(arm_px, height_m, height_px)
        arm_m = apply_perspective_correction(arm_m, camera_distance, 0)
        
        return arm_m
    except Exception as e:
        print(f"[MEASURE] Warning: Could not extract arm length: {e}")
        return None


# ====================== NEW: Foot Measurements ======================

def extract_foot_length(pose_landmarks, img_h: int, img_w: int,
                        height_m: float, height_px: float,
                        camera_distance: float = 2.5) -> Optional[float]:
    """
    Extract foot length from heel to toe.
    
    NEW FEATURE: Added for foot measurement support.
    """
    try:
        import mediapipe as mp
        mp_pose = mp.solutions.pose
        
        # Use left foot (could average both)
        left_heel = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HEEL]
        left_foot_index = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX]
        
        foot_px = calculate_3d_distance(left_heel, left_foot_index, img_h, img_w)
        foot_m = pixel_to_meters(foot_px, height_m, height_px)
        foot_m = apply_perspective_correction(foot_m, camera_distance, 0)
        
        return foot_m
    except Exception as e:
        print(f"[MEASURE] Warning: Could not extract foot length: {e}")
        return None


def extract_foot_width(pose_landmarks, img_h: int, img_w: int,
                       height_m: float, height_px: float,
                       foot_length: Optional[float] = None) -> Optional[float]:
    """
    Estimate foot width.
    
    NEW FEATURE: Added for foot measurement support.
    
    Note: MediaPipe doesn't provide cross-foot landmarks, so we estimate
    width as approximately 38-40% of foot length (typical human proportion).
    """
    try:
        # If foot length is available, use proportion
        if foot_length and foot_length > 0:
            # Typical foot width is 38-40% of foot length
            foot_width = foot_length * 0.39
            return foot_width
        
        # Alternative: try to estimate from ankle width
        import mediapipe as mp
        mp_pose = mp.solutions.pose
        
        left_ankle = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE]
        left_heel = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HEEL]
        
        # This is a rough approximation
        ankle_heel_px = calculate_3d_distance(left_ankle, left_heel, img_h, img_w)
        ankle_heel_m = pixel_to_meters(ankle_heel_px, height_m, height_px)
        
        # Foot width roughly 1.2-1.3x ankle-heel distance
        foot_width = ankle_heel_m * 1.25
        
        return foot_width
    except Exception as e:
        print(f"[MEASURE] Warning: Could not extract foot width: {e}")
        return None


# ====================== Main Extraction Function ======================

def measure_body_height_px(pose_landmarks, img_h: int, img_w: int) -> Optional[float]:
    """Measure body height in pixels from pose landmarks."""
    try:
        import mediapipe as mp
        mp_pose = mp.solutions.pose
        
        nose = pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
        top_y = nose.y * img_h - (0.1 * img_h)
        
        left_ankle = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE]
        right_ankle = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE]
        bottom_y = ((left_ankle.y + right_ankle.y) / 2.0) * img_h
        
        height_px = bottom_y - top_y
        return max(height_px, 1.0)
    except Exception as e:
        print(f"[MEASURE] Warning: Could not measure height: {e}")
        return None


def extract_all_measurements(pose_landmarks, img_h: int, img_w: int,
                             user_height_m: float,
                             camera_info: Optional[Dict] = None) -> Dict[str, Optional[float]]:
    """
    Extract all body measurements from pose landmarks.
    
    ENHANCED: Now includes foot_length and foot_width.
    
    Args:
        pose_landmarks: MediaPipe pose landmarks
        img_h, img_w: Image dimensions
        user_height_m: User's real height (mandatory, used as reference)
        camera_info: Optional dict with 'distance_m' and 'roll_deg'
        
    Returns:
        Dictionary of measurements in meters
    """
    camera_distance = 2.5
    roll_deg = 0.0
    if camera_info:
        camera_distance = camera_info.get('distance_m', 2.5)
        roll_deg = camera_info.get('roll_deg', 0.0)
    
    height_px = measure_body_height_px(pose_landmarks, img_h, img_w)
    if not height_px or height_px < 10:
        print("[MEASURE] Error: Could not measure body height in image")
        return {}
    
    print(f"[MEASURE] Reference: User height = {user_height_m:.2f}m, "
          f"Measured = {height_px:.0f}px")
    print(f"[MEASURE] Camera: distance = {camera_distance:.2f}m, "
          f"tilt = {roll_deg:.1f}°")
    
    # Extract foot length first (needed for width estimation)
    foot_length = extract_foot_length(pose_landmarks, img_h, img_w,
                                      user_height_m, height_px,
                                      camera_distance)
    
    # Extract all measurements
    measurements = {
        'shoulder': extract_shoulder_width(pose_landmarks, img_h, img_w,
                                          user_height_m, height_px,
                                          camera_distance, roll_deg),
        'chest': extract_chest_circumference(pose_landmarks, img_h, img_w,
                                             user_height_m, height_px,
                                             camera_distance),
        'waist': extract_waist_circumference(pose_landmarks, img_h, img_w,
                                             user_height_m, height_px,
                                             camera_distance),
        'hips': extract_hip_circumference(pose_landmarks, img_h, img_w,
                                          user_height_m, height_px,
                                          camera_distance),
        'inseam': extract_inseam_length(pose_landmarks, img_h, img_w,
                                        user_height_m, height_px,
                                        camera_distance),
        'arm': extract_arm_length(pose_landmarks, img_h, img_w,
                                  user_height_m, height_px,
                                  camera_distance),
        'foot_length': foot_length,
        'foot_width': extract_foot_width(pose_landmarks, img_h, img_w,
                                        user_height_m, height_px,
                                        foot_length),
    }
    
    print("[MEASURE] Extracted measurements:")
    for key, value in measurements.items():
        if value:
            print(f"  {key}: {value:.3f}m ({value*100:.1f}cm)")
        else:
            print(f"  {key}: failed to extract")
    
    return measurements


def merge_measurements(photo_measurements: Dict[str, Optional[float]],
                      user_measurements: Dict[str, Optional[float]],
                      confidence_threshold: float = 0.7) -> Dict[str, Optional[float]]:
    """
    Merge photo-extracted and user-provided measurements.
    
    Priority:
    1. User-provided measurements (always trusted)
    2. Photo-extracted measurements (if confidence is high enough)
    3. None if neither available
    
    ENHANCED: Now handles foot_length and foot_width.
    """
    merged = {}
    
    # All measurement keys including new foot measurements
    all_keys = ['chest', 'waist', 'hips', 'shoulder', 'inseam', 'arm',
                'foot_length', 'foot_width']
    
    for key in all_keys:
        user_val = user_measurements.get(key)
        photo_val = photo_measurements.get(key)
        
        if user_val is not None and user_val > 0:
            merged[key] = user_val
            print(f"[MEASURE] {key}: Using user input = {user_val:.3f}m")
        elif photo_val is not None and photo_val > 0:
            merged[key] = photo_val
            print(f"[MEASURE] {key}: Using photo extraction = {photo_val:.3f}m")
        else:
            merged[key] = None
            print(f"[MEASURE] {key}: No data available")
    
    return merged


def validate_measurement_sanity(measurements: Dict[str, Optional[float]],
                                height_m: float,
                                use_defaults: bool = True) -> Dict[str, Optional[float]]:
    """
    Validate measurements are physically reasonable.
    
    FIXED:
    - Relaxed validation ranges to accept normal human variation
    - Added default fallback values when measurements are unreasonable
    - Better logging to show what's happening
    
    Args:
        measurements: Dict of measurements to validate
        height_m: Reference height in meters
        use_defaults: If True, use proportional defaults for unreasonable values
        
    Returns:
        Dict with validated measurements (None or default for unreasonable values)
    """
    validated = measurements.copy()
    
    # FIXED: More realistic ranges accounting for human variation
    reasonable_ranges = {
        'shoulder': (0.18, 0.32),      # 18-32% (was 20-30%)
        'chest': (0.42, 0.68),          # 42-68% (was 45-65%)
        'waist': (0.30, 0.58),          # 30-58% (was 40-60%) - allows slim builds
        'hips': (0.38, 0.65),           # 38-65% (was 45-65%) - allows slim builds
        'inseam': (0.40, 0.50),         # 40-50% (was 40-52%) - tightened upper bound
        'arm': (0.34, 0.46),            # 34-46% (was 35-45%)
        'foot_length': (0.13, 0.17),    # 13-17% (was 12-18%)
        'foot_width': (0.04, 0.07),     # 4-7% (was 4-8%)
    }
    
    # Default proportions for fallback (based on average human proportions)
    default_proportions = {
        'shoulder': 0.25,
        'chest': 0.52,
        'waist': 0.42,
        'hips': 0.50,
        'inseam': 0.45,
        'arm': 0.38,
        'foot_length': 0.15,
        'foot_width': 0.055,
    }
    
    print("\n[MEASURE] Validating measurements:")
    
    for key, (min_ratio, max_ratio) in reasonable_ranges.items():
        value = validated.get(key)
        
        if value is None or value <= 0:
            if use_defaults:
                validated[key] = height_m * default_proportions[key]
                print(f"[MEASURE] {key}: None → using default {validated[key]:.3f}m "
                      f"({default_proportions[key]*100:.1f}% of height)")
            else:
                print(f"[MEASURE] {key}: None (no data)")
            continue
        
        ratio = value / height_m
        
        if ratio < min_ratio or ratio > max_ratio:
            print(f"[MEASURE] ⚠ {key} = {value:.3f}m ({ratio*100:.1f}% of height) "
                  f"is unreasonable (valid range: {min_ratio*100:.1f}-{max_ratio*100:.1f}%)")
            
            if use_defaults:
                # Use proportional default instead of None
                validated[key] = height_m * default_proportions[key]
                print(f"[MEASURE]   → Replacing with default {validated[key]:.3f}m "
                      f"({default_proportions[key]*100:.1f}% of height)")
            else:
                validated[key] = None
                print(f"[MEASURE]   → Discarding (set to None)")
        else:
            print(f"[MEASURE] {key}: {value:.3f}m ✓ ({ratio*100:.1f}% of height)")
    
    return validated



def get_default_measurement(measurement_name: str, height_m: float) -> float:
    """
    Get a reasonable default value for a measurement based on height.
    
    NEW FUNCTION: Provides fallback values when measurements fail.
    """
    default_proportions = {
        'shoulder': 0.25,
        'chest': 0.52,
        'waist': 0.42,
        'hips': 0.50,
        'inseam': 0.45,
        'arm': 0.38,
        'foot_length': 0.15,
        'foot_width': 0.055,
        'neck': 0.20,
        'head': 0.35,
        'hand': 0.10,
    }
    
    proportion = default_proportions.get(measurement_name, 0.3)
    return height_m * proportion
