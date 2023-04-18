import argparse
import sys
import os

from utils import load_config, store_frames


def generate_data(data_dir, seqs, phase, output_dir):
    # Loop through the sequences
    for seq in seqs:
        print(f"Processing SEQ: {seq}:")
        seq_dir = data_dir + seq
        cams_list = os.listdir(seq_dir)
        # Loop through the cameras
        for i, cam in enumerate(cams_list):
            print(f"Processing CAM: {cam} {i+1}/{len(cams_list)}")
            gt_path, avi_path = f"{seq_dir}/{cam}/gt/gt.txt", f"{seq_dir}/{cam}/vdo.avi"
            # Store the frames and the ground truth for the current sequence and camera
            store_frames(f"{output_dir}/{phase}", seq, cam, avi_path, gt_path)


def prepare_detection_data(cfg):
    output_dir = cfg["output_dir"]

    os.makedirs(output_dir, exist_ok=True)
    print(f"Creating dirs in: {output_dir}")
    os.makedirs(f"{output_dir}/train", exist_ok=True)
    os.makedirs(f"{output_dir}/val", exist_ok=True)

    generate_data(cfg["data_dir"], cfg["train_S"], "train", cfg["output_dir"])
    generate_data(cfg["data_dir"], cfg["val_S"], "val", cfg["output_dir"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="../config/prepare_detection_data.yaml")
    args = parser.parse_args(sys.argv[1:])

    cfg = load_config(args.config)

    prepare_detection_data(cfg)
