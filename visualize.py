import cv2
import json
import numpy as np
from collections import defaultdict
import os

def visualize_matches(tacticam_video, broadcast_video, tacticam_tracks_file, broadcast_tracks_file, mappings_file):
    # 1. Load all data files
    print("Loading data files...")
    if not all(os.path.exists(f) for f in [tacticam_video, broadcast_video, tacticam_tracks_file, broadcast_tracks_file, mappings_file]):
        print("Error: One or more required files are missing.")
        print(f"Tacticam Video: {'Found' if os.path.exists(tacticam_video) else 'Missing'}")
        print(f"Broadcast Video: {'Found' if os.path.exists(broadcast_video) else 'Missing'}")
        print(f"Tacticam Tracks: {'Found' if os.path.exists(tacticam_tracks_file) else 'Missing'}")
        print(f"Broadcast Tracks: {'Found' if os.path.exists(broadcast_tracks_file) else 'Missing'}")
        print(f"Mappings: {'Found' if os.path.exists(mappings_file) else 'Missing'}")
        return
    with open(tacticam_tracks_file, 'r') as f:
        tacticam_tracks = json.load(f)
    with open(broadcast_tracks_file, 'r') as f:
        broadcast_tracks = json.load(f)
    with open(mappings_file, 'r') as f:
        mappings = json.load(f)
    # Group tracks by frame for efficient access
    tacticam_frames = defaultdict(list)
    for track in tacticam_tracks:
        tacticam_frames[track['frame']].append(track)

    broadcast_frames = defaultdict(list)
    for track in broadcast_tracks:
        broadcast_frames[track['frame']].append(track)
        
    # Create mapping
    tacticam_to_global = {m['tacticam_track_id']: m['global_player_id'] for m in mappings}
    broadcast_to_global = {m['broadcast_track_id']: m['global_player_id'] for m in mappings}

    tacticam_cap = cv2.VideoCapture(tacticam_video)
    broadcast_cap = cv2.VideoCapture(broadcast_video)
    
    frame_id = 0
    colors = {}
    for mapping in mappings:
        gid = mapping['global_player_id']
        if gid not in colors:
            colors[gid] = tuple(np.random.randint(100, 256, 3).tolist())

    print("Starting visualization... Press 'q' to quit.")

    while tacticam_cap.isOpened() and broadcast_cap.isOpened():
        ret_tac, frame_tac = tacticam_cap.read()
        ret_bc, frame_bc = broadcast_cap.read()
        if not ret_tac or not ret_bc:
            break
        # Draw on Tacticam Frame
        if frame_id in tacticam_frames:
            for track in tacticam_frames[frame_id]:
                track_id = track['track_id']
                if track_id in tacticam_to_global:
                    gid = tacticam_to_global[track_id]
                    color = colors.get(gid, (0, 0, 255)) # Default to red if color not found
                    x1, y1, x2, y2 = map(int, track['bbox'])
                    cv2.rectangle(frame_tac, (x1, y1), (x2, y2), color, 3)
                    cv2.putText(frame_tac, f"P-{gid}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        # Draw on Broadcast Frame
        if frame_id in broadcast_frames:
            for track in broadcast_frames[frame_id]:
                track_id = track['track_id']
                if track_id in broadcast_to_global:
                    gid = broadcast_to_global[track_id]
                    color = colors.get(gid, (0, 0, 255))
                    x1, y1, x2, y2 = map(int, track['bbox'])
                    cv2.rectangle(frame_bc, (x1, y1), (x2, y2), color, 3)
                    cv2.putText(frame_bc, f"P-{gid}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

        # 5. Resize and combine frames for display
        target_height = 720
        # Resize tacticam frame
        tac_h, tac_w, _ = frame_tac.shape
        tac_ratio = target_height / tac_h
        frame_tac_resized = cv2.resize(frame_tac, (int(tac_w * tac_ratio), target_height))
        
        # Resize broadcast frame
        bc_h, bc_w, _ = frame_bc.shape
        bc_ratio = target_height / bc_h
        frame_bc_resized = cv2.resize(frame_bc, (int(bc_w * bc_ratio), target_height))
        # Combine frames
        combined_frame = np.hstack((frame_tac_resized, frame_bc_resized))
        cv2.imshow('Cross-Camera Player Mapping Visualization', combined_frame)

        frame_id += 1

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    # 6. Cleanup
    tacticam_cap.release()
    broadcast_cap.release()
    cv2.destroyAllWindows()
    print("Visualization finished.")

def main():
    visualize_matches(
        tacticam_video="tacticam.mp4",
        broadcast_video="broadcast.mp4",
        tacticam_tracks_file="tracks_tacticam.json",
        broadcast_tracks_file="tracks_broadcast.json",
        mappings_file="player_mappings.json"
    )

if __name__ == "__main__":
    main() 