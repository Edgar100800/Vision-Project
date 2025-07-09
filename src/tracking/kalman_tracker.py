"""
Kalman filter tracker for person tracking.
"""

import numpy as np
import cv2
from typing import Tuple, Optional
import logging

from ..utils.data_structures import Detection


class KalmanTracker:
    """Kalman filter for tracking person positions."""
    
    def __init__(self, process_noise: float = 0.01, measurement_noise: float = 0.1):
        """
        Initialize Kalman tracker.
        
        Args:
            process_noise: Process noise covariance
            measurement_noise: Measurement noise covariance
        """
        # State: [x, y, vx, vy] (position and velocity)
        self.kalman = cv2.KalmanFilter(4, 2)
        
        # State transition matrix
        self.kalman.transitionMatrix = np.array([
            [1, 0, 1, 0],  # x = x + vx
            [0, 1, 0, 1],  # y = y + vy
            [0, 0, 1, 0],  # vx = vx
            [0, 0, 0, 1]   # vy = vy
        ], np.float32)
        
        # Measurement matrix (we only measure position)
        self.kalman.measurementMatrix = np.array([
            [1, 0, 0, 0],  # measure x
            [0, 1, 0, 0]   # measure y
        ], np.float32)
        
        # Process noise covariance
        self.kalman.processNoiseCov = np.array([
            [process_noise, 0, 0, 0],
            [0, process_noise, 0, 0],
            [0, 0, process_noise, 0],
            [0, 0, 0, process_noise]
        ], np.float32)
        
        # Measurement noise covariance
        self.kalman.measurementNoiseCov = np.array([
            [measurement_noise, 0],
            [0, measurement_noise]
        ], np.float32)
        
        # Initial state covariance
        self.kalman.errorCovPost = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], np.float32)
        
        self.is_initialized = False
        logging.info("KalmanTracker initialized")
    
    def predict(self) -> Tuple[float, float]:
        """
        Predict next position.
        
        Returns:
            Predicted position (x, y)
        """
        if not self.is_initialized:
            return (0.0, 0.0)
        
        prediction = self.kalman.predict()
        return (prediction[0], prediction[1])
    
    def update(self, measurement: Tuple[float, float]) -> Tuple[float, float]:
        """
        Update with new measurement.
        
        Args:
            measurement: Measured position (x, y)
            
        Returns:
            Updated position (x, y)
        """
        measurement_array = np.array([[measurement[0]], [measurement[1]]], np.float32)
        
        if not self.is_initialized:
            # Initialize with first measurement
            self.kalman.statePost = np.array([
                [measurement[0]],  # x
                [measurement[1]],  # y
                [0.0],            # vx
                [0.0]             # vy
            ], np.float32)
            self.is_initialized = True
        
        # Update
        correction = self.kalman.correct(measurement_array)
        return (correction[0], correction[1])
    
    def get_velocity(self) -> Tuple[float, float]:
        """
        Get current velocity estimate.
        
        Returns:
            Velocity (vx, vy)
        """
        if not self.is_initialized:
            return (0.0, 0.0)
        
        state = self.kalman.statePost
        return (state[2], state[3])
    
    def predict_future_position(self, steps: int = 5) -> Tuple[float, float]:
        """
        Predict position several steps ahead.
        
        Args:
            steps: Number of steps to predict ahead
            
        Returns:
            Predicted position (x, y)
        """
        if not self.is_initialized:
            return (0.0, 0.0)
        
        # Create transition matrix for multiple steps
        transition = np.linalg.matrix_power(self.kalman.transitionMatrix, steps)
        
        # Predict
        current_state = self.kalman.statePost
        future_state = transition @ current_state
        
        return (future_state[0], future_state[1])
    
    def reset(self):
        """Reset the Kalman filter."""
        self.is_initialized = False
        self.kalman.statePost = np.zeros((4, 1), np.float32)
        self.kalman.statePre = np.zeros((4, 1), np.float32) 