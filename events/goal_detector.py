import os
import numpy as np

# Configurable thresholds
GOAL_ZONE_WIDTH_PCT = float(os.getenv('GOAL_ZONE_WIDTH_PCT', '0.15'))
GOAL_COOLDOWN_SECONDS = float(os.getenv('GOAL_COOLDOWN_SECONDS', '3.0'))
CONSECUTIVE_FRAMES_REQ = int(os.getenv('CONSECUTIVE_FRAMES_REQ', '3'))
GOAL_Y_BAND_TOP = float(os.getenv('GOAL_Y_BAND_TOP', '0.3'))
GOAL_Y_BAND_BOTTOM = float(os.getenv('GOAL_Y_BAND_BOTTOM', '0.8'))

def detect_goals(ball_track, team_possession_history, frame_size, fps):
    """
    Detect goals based on ball entering goal zones.
    
    Args:
        ball_track: List of ball positions per frame
        team_possession_history: List of team possession per frame
        frame_size: Tuple of (height, width) for frame dimensions
        fps: Frames per second
        
    Returns:
        List of goal events: [{"frame": int, "timestamp": float, "team": int}, ...]
    """
    goals = []
    
    if not ball_track or not team_possession_history:
        return goals
    
    height, width = frame_size[:2]  # Handle (H, W, C) format
    goal_zone_width = int(width * GOAL_ZONE_WIDTH_PCT)
    y_top = int(height * GOAL_Y_BAND_TOP)
    y_bottom = int(height * GOAL_Y_BAND_BOTTOM)
    
    # Define goal zones (left and right)
    left_goal_zone = (0, goal_zone_width, y_top, y_bottom)
    right_goal_zone = (width - goal_zone_width, width, y_top, y_bottom)
    
    consecutive_in_goal = 0
    current_goal_zone = None
    last_goal_frame = -1
    cooldown_frames = int(GOAL_COOLDOWN_SECONDS * fps)
    
    for frame_num, ball_frame_data in enumerate(ball_track):
        if not ball_frame_data or 1 not in ball_frame_data:
            consecutive_in_goal = 0
            current_goal_zone = None
            continue
            
        ball_bbox = ball_frame_data[1]['bbox']
        ball_center_x = (ball_bbox[0] + ball_bbox[2]) / 2
        ball_center_y = (ball_bbox[1] + ball_bbox[3]) / 2
        
        # Check if ball is in goal zone
        in_left_goal = (left_goal_zone[0] <= ball_center_x <= left_goal_zone[1] and 
                       left_goal_zone[2] <= ball_center_y <= left_goal_zone[3])
        in_right_goal = (right_goal_zone[0] <= ball_center_x <= right_goal_zone[1] and 
                        right_goal_zone[2] <= ball_center_y <= right_goal_zone[3])
        
        if in_left_goal or in_right_goal:
            goal_zone = 'left' if in_left_goal else 'right'
            
            if goal_zone == current_goal_zone:
                consecutive_in_goal += 1
            else:
                consecutive_in_goal = 1
                current_goal_zone = goal_zone
            
            # Goal detected if consecutive frames requirement met and cooldown passed
            if (consecutive_in_goal >= CONSECUTIVE_FRAMES_REQ and 
                frame_num - last_goal_frame > cooldown_frames):
                
                # Determine scoring team from recent possession history
                possession_lookback_frames = min(int(2.0 * fps), frame_num)
                recent_possession = team_possession_history[max(0, frame_num - possession_lookback_frames):frame_num]
                
                if recent_possession:
                    # Get most common team from recent possession (excluding 0/unknown)
                    possession_counts = {}
                    for team in recent_possession:
                        if team > 0:
                            possession_counts[team] = possession_counts.get(team, 0) + 1
                    
                    if possession_counts:
                        scoring_team = max(possession_counts, key=possession_counts.get)
                        
                        # For left goal, scoring team is typically team 2 (attacking left to right)
                        # For right goal, scoring team is typically team 1 (attacking right to left)
                        # But we use possession history as the source of truth
                        
                        goals.append({
                            "frame": frame_num,
                            "timestamp": round(frame_num / fps, 1),
                            "team": scoring_team
                        })
                        
                        last_goal_frame = frame_num
        else:
            consecutive_in_goal = 0
            current_goal_zone = None
    
    return goals