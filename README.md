# Player Mapping Pipeline

This project matches and maps player tracks between two different video sources (e.g., tacticam and broadcast) of the same sports event using deep learning and computer vision.

## Features
- Detects and tracks players in video using YOLO and DeepSort.
- Extracts visual embeddings for each player track using ResNet50.
- Matches players across two camera views using cosine similarity and the Hungarian algorithm.
- Outputs a mapping of players between the two videos.

## Requirements
- Python 3.8+
- See `requirements.txt` for all dependencies.
- Pre-trained YOLO weights file (`best.pt`)
- Input videos: `tacticam.mp4` and `broadcast.mp4`

## Usage

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Place your input videos** (`tacticam.mp4`, `broadcast.mp4`) and YOLO weights (`best.pt`) in the project directory.

3. **Run the full pipeline:**
   ```bash
   python run_all.py
   ```
   This will:
   - Generate player tracks for the tacticam video.
   - Map players between tacticam and broadcast videos.
   - Save results to `player_mappings.json`.

## Output
- `tracks_tacticam.json` and `tracks_broadcast.json`: Player tracks with embeddings.
- `player_mappings.json`: List of matched players between the two videos.

## File Descriptions
- `generate_tracks.py`: Detects and tracks players, saves track data.
- `player_mapping.py`: Extracts embeddings, matches players, and saves mapping.
- `run_all.py`: Runs the full pipeline automatically.
- `requirements.txt`: Python dependencies.

## Notes
- Make sure your input videos and weights are named as above or update the scripts accordingly.
- For best results, use high-quality, synchronized videos. 