import os
import cv2
import yaml


def load_config(cfg_path):
    with open(cfg_path, 'r') as f:
        config_yaml = yaml.safe_load(f)
    return config_yaml


def to_yolo_format(ann, img_w, img_h):
    x1 = int(ann[1])
    y1 = int(ann[2])
    w = int(ann[3])
    h = int(ann[4])

    # Get the center coordinates
    x_center, y_center = float(x1 + (w / 2)), float(y1 + (h / 2))

    # Normalize the coordinates
    x_center /= img_w
    y_center /= img_h
    box_width = float(w) / img_w
    box_height = float(h) / img_h

    return [0, x_center, y_center, box_width, box_height]


def load_gt_aicity(txt_path):

    with open(txt_path, 'r') as file:
        # Create an empty dictionary for the ground truth
        my_dict = {}
        for row in file:
            # Split the row into columns
            columns = row.strip().split(",")
            # Get the key and the values
            key = columns[0]
            values = columns[1:]
            if key not in my_dict.keys():
                my_dict[key] = [values]
            else:
                my_dict[key].append(values)

    return my_dict


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