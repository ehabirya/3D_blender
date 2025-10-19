#!/usr/bin/env python3
"""
measurement_extractor.py - Extract body measurements from photos

This module extracts real-world body measurements from photos using:
1. MediaPipe pose landmarks
2. Camera calibration (distance, angle)
3. User-provided height as reference scale

Combines photo-extracted measurements with user inputs (user takes priority).
"""

from typing import Dict, Optional, Tuple
import numpy as np
import math


def calculate_3d_distance(point1, point2, img_height: int, img_width: int) -> float:
    """
    Calculate pixel distance between two landmark points.
    
    Args:
        point1, point2: MediaPipe landmarks (normalized 0-1 coordinates)
        img_height, img_width: Image dimensions in pixels
        
    Returns:
        Distance in pixels
    """
    p1 = np.array([point1.x * img_width, point1.y * img_height, point1.z * img_width])
    p2 = np.array([point2.x * img_width, point2.y * img_height, point2.z * img_width])
    return float(np.linalg.norm(p2 - p1))


def pixel_to_meters(pixel_distance: float, reference_height_m: float, 
                    measured_height_px: float) -> float:
    """
    Convert pixel measurements to real-world meters.
    
    Uses known height as reference scale.
    
    Args:
        pixel_distance: Distance in pixels to convert
        reference_height_m: User's real height in meters
        measured_height_px: Measured height in pixels from landmarks
        
    Returns:
        Distance in meters
    """
    if measured_height_px <= 0:
        return 0.0
    
    # pixels_per_meter = measured_height_px / reference_height_m
    meters_per_pixel = reference_height_m / measured_height_px
    return pixel_distance * meters_per_pixel


def apply_perspective_correction(measurement_m: float, camera_distance_m: float, 
                                 angle_deg: float) -> float:
    """
    Correct measurement for camera distance and angle.
    
    When camera is far, there's less perspective distortion.
    When camera is tilted, measurements need angular correction.
    
    Args:
        measurement_m: Raw measurement in meters
        camera_distance_m: Estimated camera distance
        angle_deg: Camera tilt angle from vertical
        
    Returns:
        Corrected measurement in meters
    """
    # Distance correction (closer = more distortion)
    # Optimal distance is 2-3 meters
    if camera_distance_m < 1.5:
        distance_factor = 1.0 + (1.5 - camera_distance_m) * 0.1  # Up to 10% correction
    elif camera_distance_m > 3.5:
        distance_factor = 1.0 - (camera_distance_m - 3.5) * 0.05  # Up to 5% correction
    else:
        distance_factor = 1.0
    
    # Angle correction (tilt causes foreshortening)
    angle_rad = math.radians(abs(angle_deg))
    angle_factor = 1.0 / math.cos(angle_rad) if angle_rad < math.radians(30) else 1.0
    
    return measurement_m * distance_factor * angle_factor


def extract_shoulder_width(pose_landmarks, img_h: int, img_w: int, 
                           height_m: float, height_px: float,
                           camera_distance: float = 2.5,
                           roll_deg: float = 0.0) -> Optional[float]:
    """
    Extract shoulder width from pose landmarks.
    
    Returns:
        Shoulder width in meters, or None if can't calculate
    """
    try:
        import mediapipe as mp
        mp_pose = mp.solutions.pose
        
        left_shoulder = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        
        # Calculate pixel distance
        shoulder_px = calculate_3d_distance(left_shoulder, right_shoulder, img_h, img_w)
        
        # Convert to meters using height reference
        shoulder_m = pixel_to_meters(shoulder_px, height_m, height_px)
        
        # Apply perspective correction
        shoulder_m = apply_perspective_correction(shoulder_m, camera_distance, roll_deg)
        
        return shoulder_m
    except Exception as e:
        print(f"[MEASURE] Warning: Could not extract shoulder width: {e}")
        return None


def extract_chest_circumference(pose_landmarks, img_h: int, img_w: int,
                                height_m: float, height_px: float,
                                camera_distance: float = 2.5) -> Optional[float]:
    """
    Estimate chest circumference from shoulder and side landmarks.
    
    Note: Circumference estimation from front view is approximate.
    Uses shoulder width and depth estimation.
    
    Returns:
        Estimated chest circumference in meters
    """
    try:
        import mediapipe as mp
        mp_pose = mp.solutions.pose
        
        # Get shoulder width
        shoulder_width_m = extract_shoulder_width(pose_landmarks, img_h, img_w,
                                                  height_m, height_px, camera_distance)
        if not shoulder_width_m:
            return None
        
        # Estimate chest depth from z-coordinates (rough approximation)
        left_shoulder = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        
        # Average z gives depth estimate (normalized)
        avg_z = (left_shoulder.z + right_shoulder.z) / 2.0
        
        # Chest depth is typically 0.5-0.6 of shoulder width
        # Adjust based on z-coordinate (more negative = more depth visible)
        depth_ratio = 0.55 + abs(avg_z) * 0.1  # 0.55 to 0.65 typically
        chest_depth_m = shoulder_width_m * depth_ratio
        
        # Circumference approximation: ellipse formula
        # C ≈ π * (3(a+b) - sqrt((3a+b)(a+3b)))
        # where a = width/2, b = depth/2
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
    """
    Estimate waist circumference from hip landmarks.
    
    Returns:
        Estimated waist circumference in meters
    """
    try:
        import mediapipe as mp
        mp_pose = mp.solutions.pose
        
        left_hip = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
        
        # Calculate hip width (proxy for waist)
        hip_width_px = calculate_3d_distance(left_hip, right_hip, img_h, img_w)
        hip_width_m = pixel_to_meters(hip_width_px, height_m, height_px)
        hip_width_m = apply_perspective_correction(hip_width_m, camera_distance, 0)
        
        # Waist is typically 0.8-0.9 of hip width
        waist_width_m = hip_width_m * 0.85
        
        # Estimate depth (similar to chest)
        avg_z = (left_hip.z + right_hip.z) / 2.0
        depth_ratio = 0.50 + abs(avg_z) * 0.1
        waist_depth_m = waist_width_m * depth_ratio
        
        # Circumference
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
    """
    Estimate hip circumference from hip landmarks.
    
    Returns:
        Estimated hip circumference in meters
    """
    try:
        import mediapipe as mp
        mp_pose = mp.solutions.pose
        
        left_hip = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
        
        hip_width_px = calculate_3d_distance(left_hip, right_hip, img_h, img_w)
        hip_width_m = pixel_to_meters(hip_width_px, height_m, height_px)
        hip_width_m = apply_perspective_correction(hip_width_m, camera_distance, 0)
        
        # Estimate depth
        avg_z = (left_hip.z + right_hip.z) / 2.0
        depth_ratio = 0.60 + abs(avg_z) * 0.1  # Hips have more depth
        hip_depth_m = hip_width_m * depth_ratio
        
        # Circumference
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
    """
    Extract inseam (crotch to ankle) length.
    
    Returns:
        Inseam length in meters
    """
    try:
        import mediapipe as mp
        mp_pose = mp.solutions.pose
        
        # Use left leg (average of both would be better with both legs visible)
        left_hip = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
        left_ankle = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE]
        
        inseam_px = calculate_3d_distance(left_hip, left_ankle, img_h, img_w)
        inseam_m = pixel_to_meters(inseam_px, height_m, height_px)
        inseam_m = apply_perspective_correction(inseam_m, camera_distance, 0)
        
        # Inseam is typically hip to ankle
        return inseam_m
    except Exception as e:
        print(f"[MEASURE] Warning: Could not extract inseam: {e}")
        return None


def extract_arm_length(pose_landmarks, img_h: int, img_w: int,
                       height_m: float, height_px: float,
                       camera_distance: float = 2.5) -> Optional[float]:
    """
    Extract arm length (shoulder to wrist).
    
    Returns:
        Arm length in meters
    """
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


def measure_body_height_px(pose_landmarks, img_h: int, img_w: int) -> Optional[float]:
    """
    Measure body height in pixels from pose landmarks.
    
    Returns:
        Height in pixels (top of head to feet)
    """
    try:
        import mediapipe as mp
        mp_pose = mp.solutions.pose
        
        # Top: nose or top of head estimate
        nose = pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
        top_y = nose.y * img_h - (0.1 * img_h)  # Add ~10% for head top
        
        # Bottom: average of ankles
        left_ankle = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE]
        right_ankle = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE]
        bottom_y = ((left_ankle.y + right_ankle.y) / 2.0) * img_h
        
        height_px = bottom_y - top_y
        return max(height_px, 1.0)  # Prevent division by zero
    except Exception as e:
        print(f"[MEASURE] Warning: Could not measure height: {e}")
        return None


def extract_all_measurements(pose_landmarks, img_h: int, img_w: int,
                             user_height_m: float,
                             camera_info: Optional[Dict] = None) -> Dict[str, Optional[float]]:
    """
    Extract all body measurements from pose landmarks.
    
    Args:
        pose_landmarks: MediaPipe pose landmarks
        img_h, img_w: Image dimensions
        user_height_m: User's real height (mandatory, used as reference)
        camera_info: Optional dict with 'distance_m' and 'roll_deg'
        
    Returns:
        Dictionary of measurements in meters
    """
    # Get camera parameters
    camera_distance = 2.5  # Default
    roll_deg = 0.0
    if camera_info:
        camera_distance = camera_info.get('distance_m', 2.5)
        roll_deg = camera_info.get('roll_deg', 0.0)
    
    # Measure height in pixels (for scaling reference)
    height_px = measure_body_height_px(pose_landmarks, img_h, img_w)
    if not height_px or height_px < 10:
        print("[MEASURE] Error: Could not measure body height in image")
        return {}
    
    print(f"[MEASURE] Reference: User height = {user_height_m:.2f}m, "
          f"Measured = {height_px:.0f}px")
    print(f"[MEASURE] Camera: distance = {camera_distance:.2f}m, "
          f"tilt = {roll_deg:.1f}°")
    
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
    }
    
    # Log extracted measurements
    print("[MEASURE] Extracted measurements:")
    for key, value in measurements.items():
        if value:
            print(f"  {key}: {value:.3f}m ({value*100:.1f}cm)")
        else:
            print(f"  {key}: failed to extract")
    
    return measurements


def merge_measurements(photo_measurements: Dict[str, Optional[float]],
                      user_measurements: Dict[str, Optional[float]],
                      confidence_threshold: float = 0.7) -> Dict[str, float]:
    """
    Merge photo-extracted and user-provided measurements.
    
    Priority:
    1. User-provided measurements (always trusted)
    2. Photo-extracted measurements (if confidence is high enough)
    3. Default/fallback values
    
    Args:
        photo_measurements: Extracted from photos
        user_measurements: Provided by user (can be None/empty)
        confidence_threshold: Minimum confidence to use photo measurements
        
    Returns:
        Final merged measurements
    """
    merged = {}
    
    for key in ['chest', 'waist', 'hips', 'shoulder', 'inseam', 'arm']:
        user_val = user_measurements.get(key)
        photo_val = photo_measurements.get(key)
        
        if user_val is not None and user_val > 0:
            # User provided - always use this
            merged[key] = user_val
            print(f"[MEASURE] {key}: Using user input = {user_val:.3f}m")
        elif photo_val is not None and photo_val > 0:
            # Use photo extraction
            merged[key] = photo_val
            print(f"[MEASURE] {key}: Using photo extraction = {photo_val:.3f}m")
        else:
            # No data available
            merged[key] = None
            print(f"[MEASURE] {key}: No data available")
    
    return merged


def validate_measurement_sanity(measurements: Dict[str, Optional[float]],
                                height_m: float) -> Dict[str, Optional[float]]:
    """
    Validate measurements are physically reasonable.
    
    Filters out clearly wrong measurements based on human proportions.
    
    Returns:
        Validated measurements (invalid ones set to None)
    """
    validated = measurements.copy()
    
    # Typical human proportions (as ratio of height)
    reasonable_ranges = {
        'shoulder': (0.20, 0.30),    # 20-30% of height
        'chest': (0.45, 0.65),       # 45-65% of height
        'waist': (0.40, 0.60),       # 40-60% of height
        'hips': (0.45, 0.65),        # 45-65% of height
        'inseam': (0.40, 0.52),      # 40-52% of height
        'arm': (0.35, 0.45),         # 35-45% of height
    }
    
    for key, (min_ratio, max_ratio) in reasonable_ranges.items():
        value = validated.get(key)
        if value is not None and value > 0:
            ratio = value / height_m
            if ratio < min_ratio or ratio > max_ratio:
                print(f"[MEASURE] ⚠ {key} = {value:.3f}m "
                      f"({ratio*100:.1f}% of height) is unreasonable, discarding")
                validated[key] = None
    
    return validated
