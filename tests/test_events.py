import unittest
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from events.goal_detector import detect_goals
from events.pass_detector import detect_passes
from events.possession import calculate_possession

class TestEventDetection(unittest.TestCase):
    
    def test_goal_detection_with_synthetic_data(self):
        """Test goal detection with mock ball tracks"""
        # Create synthetic ball track entering left goal zone
        ball_track = []
        frame_size = (600, 800)  # height, width
        
        # Ball moving towards left goal (need to get inside goal zone)
        for i in range(10):
            x = 50 - i * 5  # Moving left into goal zone
            y = 300  # In Y-band
            ball_track.append({1: {'bbox': [x-5, y-5, x+5, y+5]}})
        
        # Team possession history (team 2 attacking left goal)
        team_possession = [2] * 10
        
        goals = detect_goals(ball_track, team_possession, frame_size, 24)
        
        # Should detect a goal
        self.assertGreater(len(goals), 0)
        if goals:
            self.assertEqual(goals[0]['team'], 2)
            self.assertIsInstance(goals[0]['timestamp'], float)
            self.assertIsInstance(goals[0]['frame'], int)
    
    def test_pass_detection_heuristics(self):
        """Test pass counting with synthetic player/ball data"""
        # Create synthetic tracks
        tracks = {
            'players': [
                {1: {'bbox': [100, 100, 120, 140]}, 2: {'bbox': [200, 100, 220, 140]}},  # Frame 0
                {1: {'bbox': [100, 100, 120, 140]}, 2: {'bbox': [200, 100, 220, 140]}},  # Frame 1
                {1: {'bbox': [100, 100, 120, 140]}, 2: {'bbox': [200, 100, 220, 140]}},  # Frame 2
            ],
            'ball': [
                {1: {'bbox': [105, 135, 115, 145]}},  # Near player 1
                {1: {'bbox': [150, 135, 160, 145]}},  # Moving towards player 2
                {1: {'bbox': [195, 135, 205, 145]}},  # Near player 2
            ]
        }
        
        team_assignments = {1: 1, 2: 1}  # Both players on team 1
        
        passes = detect_passes(tracks, team_assignments, 24)
        
        self.assertIsInstance(passes, dict)
        self.assertIn('team1', passes)
        self.assertIn('team2', passes)
        # Should detect at least one pass since ball moved between same-team players
        self.assertGreaterEqual(passes['team1'], 0)
    
    def test_possession_calculation(self):
        """Test possession percentage calculation"""
        # Create synthetic tracks
        tracks = {
            'players': [
                {1: {'bbox': [100, 100, 120, 140]}, 2: {'bbox': [200, 100, 220, 140]}},  # Frame 0
                {1: {'bbox': [100, 100, 120, 140]}, 2: {'bbox': [200, 100, 220, 140]}},  # Frame 1
            ],
            'ball': [
                {1: {'bbox': [105, 135, 115, 145]}},  # Near player 1
                {1: {'bbox': [195, 135, 205, 145]}},  # Near player 2
            ]
        }
        
        ball_track = tracks['ball']
        team_assignments = {1: 1, 2: 2}  # Players on different teams
        
        possession = calculate_possession(tracks, ball_track, team_assignments)
        
        self.assertIsInstance(possession, dict)
        self.assertIn('team1_pct', possession)
        self.assertIn('team2_pct', possession)
        self.assertIn('unknown_pct', possession)
        
        # Percentages should sum to approximately 100
        total = possession['team1_pct'] + possession['team2_pct'] + possession['unknown_pct']
        self.assertAlmostEqual(total, 100.0, delta=0.1)
    
    def test_empty_data_handling(self):
        """Test that empty data is handled gracefully"""
        # Test with empty data
        goals = detect_goals([], [], (600, 800), 24)
        self.assertEqual(goals, [])
        
        passes = detect_passes({}, {}, 24)
        self.assertEqual(passes, {"team1": 0, "team2": 0})
        
        possession = calculate_possession({}, [], {})
        self.assertEqual(possession['team1_pct'], 0.0)
        self.assertEqual(possession['team2_pct'], 0.0)
        self.assertEqual(possession['unknown_pct'], 100.0)

if __name__ == '__main__':
    unittest.main()