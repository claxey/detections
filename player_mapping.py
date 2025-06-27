import cv2
import numpy as np
import json
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from scipy.optimize import linear_sum_assignment
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
import os

class PlayerMapper:
    def __init__(self, sample_interval=10):
        self.sample_interval = sample_interval
        self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    def extract_embedding(self, frame, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame.shape[1], x2)
        y2 = min(frame.shape[0], y2)
        if x2 <= x1 or y2 <= y1:
            return None
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(crop_rgb)
        input_tensor = self.transform(pil_image).unsqueeze(0)
        with torch.no_grad():
            embedding = self.model(input_tensor)
            embedding = embedding.squeeze().numpy()
        return embedding
    def get_track_embeddings(self, video_path, tracks_data):
        print(f"Extracting embeddings from {video_path}...")
        tracks_by_id = defaultdict(list)
        for track in tracks_data:
            if track["class"] == "player":
                tracks_by_id[track["track_id"]].append(track)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return {}
        track_embeddings = {}
        for track_id, track_frames in tracks_by_id.items():
            print(f"  Processing track {track_id} with {len(track_frames)} detections...")
            embeddings = []
            for i in range(0, len(track_frames), self.sample_interval):
                track = track_frames[i]
                frame_num = track["frame"]
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, frame = cap.read()
                if ret:
                    embedding = self.extract_embedding(frame, track["bbox"])
                    if embedding is not None:
                        embeddings.append(embedding)
            if embeddings:
                avg_embedding = np.mean(embeddings, axis=0)
                track_embeddings[track_id] = avg_embedding
                print(f"    Extracted {len(embeddings)} embeddings for track {track_id}")
            else:
                print(f"    No valid embeddings for track {track_id}")
        cap.release()
        return track_embeddings
    def match_players(self, tacticam_embeddings, broadcast_embeddings):
        print("Matching players across cameras...") 
        tacticam_ids = list(tacticam_embeddings.keys())
        broadcast_ids = list(broadcast_embeddings.keys()) 
        if not tacticam_ids or not broadcast_ids:
            print("No embeddings found for matching")
            return []
        similarity_matrix = np.zeros((len(tacticam_ids), len(broadcast_ids)))
        for i, tacticam_id in enumerate(tacticam_ids):
            for j, broadcast_id in enumerate(broadcast_ids):
                tacticam_emb = tacticam_embeddings[tacticam_id]
                broadcast_emb = broadcast_embeddings[broadcast_id]    
                similarity = cosine_similarity(
                    tacticam_emb.reshape(1, -1), 
                    broadcast_emb.reshape(1, -1)
                )[0, 0]
                similarity_matrix[i, j] = similarity
        row_indices, col_indices = linear_sum_assignment(-similarity_matrix)
        mappings = []
        for i, (row_idx, col_idx) in enumerate(zip(row_indices, col_indices)):
            tacticam_track_id = tacticam_ids[row_idx]
            broadcast_track_id = broadcast_ids[col_idx]
            similarity_score = similarity_matrix[row_idx, col_idx]
            if similarity_score > 0.3:
                mappings.append({
                    "global_player_id": i + 1,
                    "tacticam_track_id": tacticam_track_id,
                    "broadcast_track_id": broadcast_track_id,
                    "similarity_score": float(similarity_score)
                })
                print(f"  Matched: Tacticam track {tacticam_track_id} -> Broadcast track {broadcast_track_id} (similarity: {similarity_score:.3f})")
            else:
                print(f"  Low similarity: Tacticam track {tacticam_track_id} -> Broadcast track {broadcast_track_id} (similarity: {similarity_score:.3f})")
        return mappings  
    def run_mapping(self, tacticam_video, broadcast_video, tacticam_tracks, broadcast_tracks):
        print("Loading track data...")
        with open(tacticam_tracks, 'r') as f:
            tacticam_data = json.load(f)       
        with open(broadcast_tracks, 'r') as f:
            broadcast_data = json.load(f)
        tacticam_embeddings = self.get_track_embeddings(tacticam_video, tacticam_data)
        broadcast_embeddings = self.get_track_embeddings(broadcast_video, broadcast_data)
        mappings = self.match_players(tacticam_embeddings, broadcast_embeddings)
        output_file = "player_mappings.json"
        with open(output_file, 'w') as f:
            json.dump(mappings, f, indent=2)
        print(f"\nMapping complete! Found {len(mappings)} player matches.")
        print(f"Results saved to {output_file}")
        return mappings

def main():
    if not os.path.exists("tracks_tacticam.json"):
        print("tracks_tacticam.json not found. Please run generate_tracks.py first.")
        return
    if not os.path.exists("tracks_broadcast.json"):
        print("tracks_broadcast.json not found. Please run generate_tracks.py first.")
        return
    mapper = PlayerMapper(sample_interval=10)
    mappings = mapper.run_mapping(
        tacticam_video="tacticam.mp4",
        broadcast_video="broadcast.mp4",
        tacticam_tracks="tracks_tacticam.json",
        broadcast_tracks="tracks_broadcast.json"
    )
    print("\n=== PLAYER MAPPING SUMMARY ===")
    for mapping in mappings:
        print(f"Player {mapping['global_player_id']}: "
              f"Tacticam Track {mapping['tacticam_track_id']} -> "
              f"Broadcast Track {mapping['broadcast_track_id']} "
              f"(Similarity: {mapping['similarity_score']:.3f})")

if __name__ == "__main__":
    main() 