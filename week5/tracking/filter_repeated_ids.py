import sys
import argparse

from tracking_utils import store_trackers_list, load_predictions

def main(args):
    preds = load_predictions(args.tracking_path)

    store_trackers_list(preds, args.tracking_path, file_mode="w")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--tracking_path", required=True)
    args = parser.parse_args(sys.argv[1:])

    main(args)