import os
from scripts.process_videos import process

VIDEO_ROOT = "videos"

for folder in ["men", "women"]:
    path = os.path.join(VIDEO_ROOT, folder)

    for file in os.listdir(path):
        if file.endswith(".mp4") or file.endswith(".mov"):

            full_path = os.path.join(path, file)

            print("Processing:", full_path)
            process(full_path)