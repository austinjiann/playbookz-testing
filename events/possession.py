import os
import sys
sys.path.append('../')
from utils import measure_distance, get_center_of_bbox

POSSESSION_DISTANCE = float(os.getenv('POSSESSION_DISTANCE', '40'))  # Match pass detector

def calculate_possession(tracks, ball_track, team_assignments):
    """
    Calculate ball possession percentages for each team.
    
    Args:
        tracks: Dictionary with 'players' and 'ball' tracks
        ball_track: Explicit ball track data (same as tracks['ball'])
        team_assignments: Dict mapping player_id to team number
        
    Returns:
        Dict: {"team1_pct": float, "team2_pct": float, "unknown_pct": float}
    """
    possession_counts = {"team1": 0, "team2": 0, "unknown": 0}
    total_frames = 0
    
    if not tracks.get('players') or not ball_track or not team_assignments:
        return {"team1_pct": 0.0, "team2_pct": 0.0, "unknown_pct": 100.0}
    
    for frame_num, (player_frame, ball_frame) in enumerate(zip(tracks['players'], ball_track)):
        if not ball_frame or 1 not in ball_frame:
            possession_counts["unknown"] += 1
            total_frames += 1
            continue
            
        ball_bbox = ball_frame[1]['bbox']
        ball_position = get_center_of_bbox(ball_bbox)
        
        # Find nearest player to ball
        nearest_player = None
        min_distance = float('inf')
        
        for player_id, player_data in player_frame.items():
            if player_id not in team_assignments:
                continue
                
            player_bbox = player_data['bbox']
            # Use foot position for players
            player_pos = (int((player_bbox[0] + player_bbox[2]) / 2), int(player_bbox[3]))
            
            distance = measure_distance(player_pos, ball_position)
            
            if distance < min_distance:
                min_distance = distance
                nearest_player = player_id
        
        # Assign possession based on distance threshold
        if nearest_player and min_distance < POSSESSION_DISTANCE:
            team = team_assignments.get(nearest_player)
            if team == 1:
                possession_counts["team1"] += 1
            elif team == 2:
                possession_counts["team2"] += 1
            else:
                possession_counts["unknown"] += 1
        else:
            possession_counts["unknown"] += 1
            
        total_frames += 1
    
    # Calculate percentages
    if total_frames == 0:
        return {"team1_pct": 0.0, "team2_pct": 0.0, "unknown_pct": 100.0}
    
    team1_pct = round((possession_counts["team1"] / total_frames) * 100, 1)
    team2_pct = round((possession_counts["team2"] / total_frames) * 100, 1)
    unknown_pct = round((possession_counts["unknown"] / total_frames) * 100, 1)
    
    return {
        "team1_pct": team1_pct,
        "team2_pct": team2_pct, 
        "unknown_pct": unknown_pct
    }