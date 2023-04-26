import os
import sys
import glob
import argparse

from tracking_utils import store_trackers_list, load_predictions


def main(args):
    for filepath in glob.glob(os.path.join(f'{args.tracking_data_dir}/*.txt')):
        print(filepath)
        preds = load_predictions(filepath)

        new_file = filepath.replace(".txt", "_old.txt")
        os.rename(filepath, new_file)

        store_trackers_list([preds], filepath, file_mode="w")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--tracking_data_dir", default="data/trackers/mot_challenge/parabellum-train/metric_learning/data/")
    args = parser.parse_args(sys.argv[1:])

    main(args)