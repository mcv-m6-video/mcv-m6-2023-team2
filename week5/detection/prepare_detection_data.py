import argparse
import sys
import os
import cv2

sys.path.append('../')
from utils import load_config, to_yolo_format, load_gt_aicity


def store_frames(output_dir, seq, cam, video_path, gt_path):

    video = cv2.VideoCapture(video_path)

    gt = load_gt_aicity(gt_path)  # Load the ground truth file

    os.makedirs(output_dir, exist_ok=True)

    count = 0
    while True:
        ret, frame = video.read()

        if not ret:
            break

        # Store the frame in the output directory
        filename = f"{output_dir}/{seq}_{cam}_{str(count).zfill(6)}.jpg"
        cv2.imwrite(filename, frame)

        # Store the ground truth in the output directory
        if str(count) in gt.keys():
            store_gt(f"{output_dir}/{seq}_{cam}_{str(count).zfill(6)}.txt", frame.shape[1], frame.shape[0], gt[str(count)])
        else:
            store_gt(f"{output_dir}/{seq}_{cam}_{str(count).zfill(6)}.txt")

        count += 1

    video.release()


def store_gt(out_txt_path, w=None, h=None, gt=None):
    if gt is None:
        open(out_txt_path, 'w')
    else:
        with open(out_txt_path, 'w') as file:
            for line in gt:
                # Convert the list to the YOLO format
                annotation = to_yolo_format(line, w, h)

                # Convert the list to a string and write it to the file
                row_string = ' '.join(map(str, annotation))
                file.write(row_string + '\n')


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
