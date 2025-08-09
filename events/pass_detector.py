import os
import sys
sys.path.append('../')
from utils import measure_distance, get_center_of_bbox

# Configurable thresholds
MIN_PASS_DISTANCE = float(os.getenv('MIN_PASS_DISTANCE', '5.0'))
MAX_PASS_TIME = float(os.getenv('MAX_PASS_TIME', '1.5'))
POSSESSION_DISTANCE = float(os.getenv('POSSESSION_DISTANCE', '70'))

def detect_passes(tracks, team_assignments, fps):
    """
    Detect completed passes between players of the same team.
    
    Args:
        tracks: Dictionary with 'players' and 'ball' tracks
        team_assignments: Dict mapping player_id to team number
        fps: Frames per second
        
    Returns:
        Dict: {"team1": count, "team2": count}
    """
    pass_counts = {"team1": 0, "team2": 0}
    
    if not tracks.get('players') or not tracks.get('ball') or not team_assignments:
        return pass_counts
    
    max_pass_frames = int(MAX_PASS_TIME * fps)
    last_possessor = None
    last_possession_frame = None
    last_ball_position = None
    
    for frame_num, (player_frame, ball_frame) in enumerate(zip(tracks['players'], tracks['ball'])):
        if not ball_frame or 1 not in ball_frame:
            continue
            
        ball_bbox = ball_frame[1]['bbox']
        ball_position = get_center_of_bbox(ball_bbox)
        
        # Find current ball possessor
        current_possessor = None
        min_distance = float('inf')
        
        for player_id, player_data in player_frame.items():
            if player_id not in team_assignments:
                continue
                
            player_bbox = player_data['bbox']
            # Use foot position for players
            player_pos = (int((player_bbox[0] + player_bbox[2]) / 2), int(player_bbox[3]))
            
            distance = measure_distance(player_pos, ball_position)
            
            if distance < POSSESSION_DISTANCE and distance < min_distance:
                min_distance = distance
                current_possessor = player_id
        
        # Check for pass completion
        if (current_possessor and last_possessor and 
            current_possessor != last_possessor and
            last_possession_frame is not None):
            
            # Check timing constraint
            time_diff = frame_num - last_possession_frame
            if time_diff <= max_pass_frames:
                
                # Check if same team
                current_team = team_assignments.get(current_possessor)
                last_team = team_assignments.get(last_possessor)
                
                if current_team and last_team and current_team == last_team:
                    
                    # Check distance constraint
                    if last_ball_position:
                        ball_travel_distance = measure_distance(last_ball_position, ball_position)
                        
                        if ball_travel_distance >= MIN_PASS_DISTANCE:
                            # Valid pass completed
                            if current_team == 1:
                                pass_counts["team1"] += 1
                            elif current_team == 2:
                                pass_counts["team2"] += 1
        
        # Update possession tracking
        if current_possessor:
            last_possessor = current_possessor
            last_possession_frame = frame_num
            last_ball_position = ball_position
    
    return pass_counts