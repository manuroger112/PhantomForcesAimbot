import winsound
import win32api
import win32con
import win32gui
import numpy as np
import random
import time 
import cv2
import mss
from collections import deque

#Config class to store important info about program capture
class Config:
    def __init__(self):
        
        self.width = 1920
        self.height = 1080
        self.center_x = self.width // 2
        self.center_y = self.height // 2
        self.uniformCaptureSize = 240
        self.crosshairUniform = self.uniformCaptureSize // 2
        self.capture_left = self.center_x - self.crosshairUniform
        self.capture_top = self.center_y - self.crosshairUniform
        
        self.region = {"top": self.capture_top,"left": self.capture_left,"width": self.uniformCaptureSize,"height": self.uniformCaptureSize}
        
config = Config()
screenCapture = mss.mss()

# Load template and create scaled versions for multi-scale matching
template = cv2.imread("enemyIndic3.png", cv2.IMREAD_UNCHANGED)
template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
w, h = template_gray.shape[::-1]

# Create scaled templates for multi-scale matching
scale_factors = [0.9, 1.0, 1.1]  # 90%, 100%, 110% scales
templates = []
for scale in scale_factors:
    if scale == 1.0:
        templates.append(template_gray)
    else:
        width = int(template_gray.shape[1] * scale)
        height = int(template_gray.shape[0] * scale)
        scaled_template = cv2.resize(template_gray, (width, height), interpolation=cv2.INTER_AREA)
        templates.append(scaled_template)

# Template matching will give us top left corner coords which is not what we
# want as we must hit the center of the rhombus, so we get half of template size
# to offset coords towards the center of template (rhombus)
centerW = w//2
centerH = h//2

# Copy values of these 2 vars as to make access faster (accessing attribute from fn is slower than direct variable)
crosshairU = config.crosshairUniform
regionC = config.region

# Change sensitivity here
robloxSensitivity = 0.84
PF_MouseSensitivity = 0.456
PF_AimSensitivity = 0.648

PF_sensitivity = PF_MouseSensitivity*PF_AimSensitivity
movementCompensation = 0.2 #keep it in 0 to 1 range
finalComputerSensitivityMultiplier = ((robloxSensitivity*PF_sensitivity)/0.55) + movementCompensation

# Target prediction variables
target_positions = deque(maxlen=3)  # Reduced from 5 to 3 for more responsive tracking
last_time = time.time()
prediction_enabled = False  # Disabled prediction temporarily to fix stuttering

# Threshold for template matching - increased to reduce false positives
MATCH_THRESHOLD = 0.85  # Higher threshold to avoid false positives

# Performance optimization
last_frame_time = 0
target_fps = 60  # Reduced from 240 to 60 for stability
frame_time = 1.0 / target_fps

# Consecutive matches required before shooting (reduces false positives)
min_consecutive_matches = 2
consecutive_matches = 0

def multi_scale_template_match(frame, templates, scale_factors):
    """Perform template matching at multiple scales and return the best match"""
    best_val = -1
    best_loc = None
    best_scale_idx = 0
    
    # Only check the original scale (index 1) first for performance
    template = templates[1]  # Original scale (1.0)
    result = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    
    # If we have a good match at original scale, don't check other scales
    if max_val >= MATCH_THRESHOLD:
        best_val = max_val
        best_loc = max_loc
        best_scale_idx = 1
    else:
        # Check other scales only if needed
        for i, (template, scale) in enumerate(zip(templates, scale_factors)):
            if i == 1:  # Skip the original scale as we already checked it
                continue
                
            result = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            
            if max_val > best_val:
                best_val = max_val
                best_loc = max_loc
                best_scale_idx = i
    
    # Calculate center offset based on the template size
    w_scaled = int(w * scale_factors[best_scale_idx])
    h_scaled = int(h * scale_factors[best_scale_idx])
    center_w_scaled = w_scaled // 2
    center_h_scaled = h_scaled // 2
    
    return best_val, best_loc, center_w_scaled, center_h_scaled

def predict_target_position(positions, time_delta):
    """Predict the next position based on recent movement"""
    if len(positions) < 2:
        return None
    
    # Calculate velocity from the last two positions
    last_pos = positions[-1]
    prev_pos = positions[-2]
    
    # Calculate velocity (pixels per second)
    vel_x = (last_pos[0] - prev_pos[0]) / time_delta
    vel_y = (last_pos[1] - prev_pos[1]) / time_delta
    
    # Predict next position (assuming constant velocity)
    # Adjust the prediction factor (0.05) to control how far ahead to predict
    pred_x = int(last_pos[0] + vel_x * 0.05)
    pred_y = int(last_pos[1] + vel_y * 0.05)
    
    return (pred_x, pred_y)

while True:
    # Simple sleep instead of complex frame timing to reduce stuttering
    time.sleep(0.001)  # Small sleep to prevent CPU overload
    
    # Capture screen and convert to grayscale
    GameFrame = np.array(screenCapture.grab(regionC))
    GameFrame = cv2.cvtColor(GameFrame, cv2.COLOR_BGRA2GRAY)

    # Exit condition - change to 0x12 if you want to close program with ALT
    if win32api.GetAsyncKeyState(0x6) < 0:
        winsound.Beep(1000, 10)
        break
    
    # Aiming condition - right mouse button
    elif win32api.GetAsyncKeyState(0x02) < 0:
        # Multi-scale template matching
        max_val, max_loc, center_w, center_h = multi_scale_template_match(GameFrame, templates, scale_factors)

        # If a good match is found
        if max_val >= MATCH_THRESHOLD:
            # Calculate target position
            X = max_loc[0] + center_w
            Y = max_loc[1] + center_h
            
            # Store position for prediction
            target_positions.append((X, Y))
            
            # Calculate time delta for prediction
            time_delta = time.time() - last_time
            last_time = time.time()
            
            # Apply prediction if enabled and we have enough data
            if prediction_enabled and len(target_positions) >= 2:
                predicted_pos = predict_target_position(target_positions, time_delta)
                if predicted_pos:
                    X, Y = predicted_pos
            
            # Calculate mouse movement
            nX = (-(crosshairU - X)) * finalComputerSensitivityMultiplier
            nY = (-(crosshairU - Y)) * finalComputerSensitivityMultiplier
            
            # Apply mouse movement
            win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, int(nX), int(nY), 0, 0)
            
            # Increment consecutive matches counter
            consecutive_matches += 1
            
            # Only shoot if we've had multiple consecutive matches (reduces false positives)
            if consecutive_matches >= min_consecutive_matches:
                # Click (shoot)
                win32api.mouse_event(0x0002, 0, 0, 0, 0)  # Mouse down
                time.sleep(random.uniform(0.01, 0.02))  # Slightly faster click
                win32api.mouse_event(0x0004, 0, 0, 0, 0)  # Mouse up
        else:
            # Clear prediction data if no target found
            target_positions.clear()
            # Reset consecutive matches counter
            consecutive_matches = 0
    else:
        # Clear prediction data when not aiming
        target_positions.clear()
        # Reset consecutive matches counter
        consecutive_matches = 0
            
    # Uncomment for debugging
    # cv2.imshow("test", GameFrame)

print("bye")
