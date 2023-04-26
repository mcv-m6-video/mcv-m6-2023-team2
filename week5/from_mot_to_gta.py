import os

# Convert dir structure from: /datadir/sXX_cYYY.txt
# to: /tracking_results/train/SXX/cYYY/predictions.txt

MOT_DIR = "./data/trackers/mot_challenge/parabellum-train/overlap_filtAreaTrue_filtParkedTrue/data/"
TARGET_DIR = "tracking_results/overlap_filtAreaTrue_filtParkedTrue/"

for seq_cam in os.listdir(MOT_DIR):
    seq, cam = seq_cam.split(".")[0].split("_")
    seq_dir = os.path.join(TARGET_DIR, seq)
    cam_dir = os.path.join(seq_dir, cam)
    os.makedirs(cam_dir, exist_ok=True)
    # copy sXX_cYYY.txt to sXX/cYYY/predictions.txt
    with open(os.path.join(MOT_DIR, seq_cam), "r") as f:
        lines = f.readlines()
    with open(os.path.join(cam_dir, "predictions.txt"), "w") as f:
        f.writelines(lines)