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

