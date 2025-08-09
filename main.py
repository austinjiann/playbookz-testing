from utils import read_video
from trackers import Tracker
import cv2
import numpy as np
import json
import os
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from events.goal_detector import detect_goals
from events.pass_detector import detect_passes
from events.possession import calculate_possession


def main():
    # Read Video and get FPS
    video_path = 'input_videos/no_audio_5min.mp4'
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    
    video_frames = read_video(video_path)

    # Initialize Tracker
    tracker = Tracker('models/best.pt')

    tracks = tracker.get_object_tracks(video_frames,
                                       read_from_stub=False,
                                       stub_path='stubs/track_stubs_5min.pkl')
    # Get object positions 
    tracker.add_position_to_tracks(tracks)

    # camera movement estimator
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames,
                                                                                read_from_stub=False,
                                                                                stub_path='stubs/camera_movement_stub_5min.pkl')
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks,camera_movement_per_frame)


    # View Trasnformer
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)

    # Interpolate Ball Positions
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])


    # Assign Player Teams
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], 
                                    tracks['players'][0])
    
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num],   
                                                 track['bbox'],
                                                 player_id)
            tracks['players'][frame_num][player_id]['team'] = team 
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

    
    # Assign Ball Aquisition
    player_assigner = PlayerBallAssigner()
    team_ball_control = []
    for frame_num, player_track in enumerate(tracks['players']):
        if tracks['ball'][frame_num] and 1 in tracks['ball'][frame_num]:
            ball_bbox = tracks['ball'][frame_num][1]['bbox']
            assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

            if assigned_player != -1:
                tracks['players'][frame_num][assigned_player]['has_ball'] = True
                team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
            else:
                # Use last known possession if available
                team_ball_control.append(team_ball_control[-1] if team_ball_control else 0)
        else:
            # No ball detected, use unknown possession
            team_ball_control.append(team_ball_control[-1] if team_ball_control else 0)
    team_ball_control = np.array(team_ball_control)


    # Ensure output directory exists
    os.makedirs('output_videos', exist_ok=True)
    
    # Standardize team assignments
    team_assignments = {}
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track_info in player_track.items():
            if 'team' in track_info:
                team_assignments[player_id] = track_info['team']
    
    # Event Detection (after stub processing for deterministic results)
    goals = detect_goals(tracks["ball"], team_ball_control.tolist(), video_frames[0].shape, fps)
    passes = detect_passes(tracks, team_assignments, fps)
    possession = calculate_possession(tracks, tracks["ball"], team_assignments)
    
    # Output JSON with version
    summary = {
        "version": "1.0",
        "goals": {
            "team1": len([g for g in goals if g["team"] == 1]),
            "team2": len([g for g in goals if g["team"] == 2]),
            "events": goals
        },
        "passes": passes,
        "possession": possession,
        "video": {
            "frames": len(video_frames),
            "fps": round(fps, 1),
            "duration_sec": round(len(video_frames) / fps, 1)
        }
    }
    
    with open('output_videos/summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Analysis complete. Results saved to output_videos/summary.json")
    print(f"Goals: Team 1: {summary['goals']['team1']}, Team 2: {summary['goals']['team2']}")
    print(f"Passes: Team 1: {summary['passes']['team1']}, Team 2: {summary['passes']['team2']}")
    print(f"Possession: Team 1: {summary['possession']['team1_pct']}%, Team 2: {summary['possession']['team2_pct']}%")

if __name__ == '__main__':
    main()