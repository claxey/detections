import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import json
import os

def generate_tracks(video_path, output_file, video_name):
    print(f"Processing {video_name} video...")
    model = YOLO("best.pt")
    tracker = DeepSort()
    cap = cv2.VideoCapture(video_path)
    
    all_tracks = []
    frame_id = 0
    
    def extract_embeddings(frame, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame.shape[1], x2)
        y2 = min(frame.shape[0], y2)
        if x2 <= x1 or y2 <= y1:
            return np.zeros(128)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return np.zeros(128)
        crop_resized = cv2.resize(crop, (64, 128))
        gray = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [128], [0, 256]).flatten()
        hist = hist / (hist.sum() + 1e-8)
        if len(hist) < 128:
            hist = np.pad(hist, (0, 128 - len(hist)))
        else:
            hist = hist[:128]
        return hist
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame)
        input_dets = []
        embeddings = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = box.conf[0].item()
            cls_id = int(box.cls[0].item())
            class_name = model.names[cls_id]
            
            if class_name not in ["player", "ball"]:
                continue
            input_dets.append(([x1, y1, x2 - x1, y2 - y1], conf, class_name))
            embedding = extract_embeddings(frame, [x1, y1, x2, y2])
            embeddings.append(embedding)
        tracks = tracker.update_tracks(input_dets, frame=frame)
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            ltrb = track.to_ltrb()
            embedding = embeddings[track_id]
            all_tracks.append({
                "track_id": int(track_id),
                "bbox": [int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])],
                "frame": frame_id,
                "embedding": embedding.tolist(),
                "class": track.get_det_class()
            })
        
        frame_id += 1
        if frame_id % 30 == 0:  # Print progress every 30 frames
            print(f"  Processed {frame_id} frames...")
    
    cap.release()
    # Save tracks
    with open(output_file, "w") as f:
        json.dump(all_tracks, f, indent=2)
    print(f"Completed {video_name}. {len(all_tracks)} track detections saved to {output_file}")
    return all_tracks

def main():
    # tracks for both videos
    if os.path.exists("tacticam.mp4"):
        generate_tracks("tacticam.mp4", "tracks_tacticam.json", "tacticam")
    else:
        print("tacticam.mp4 not found!")
    
    if os.path.exists("broadcast.mp4"):
        generate_tracks("broadcast.mp4", "tracks_broadcast.json", "broadcast")
    else:
        print("broadcast.mp4 not found!")

if __name__ == "__main__":
    main() 