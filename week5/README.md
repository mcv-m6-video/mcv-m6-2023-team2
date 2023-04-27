# Week 5


## 1. Prepare detection data and finetune detector on it

```
python detection/prepare_detection_data.py --config=config/prepare_detection_data.yaml
```

This will create directory containing the training images in the format expected by `Ultralytic's` library.

Then, finetune a `YOLOv8` detector:

```
python detection/finetune_detector.py --config=config/finetune_detector.yaml
```

## 2. Perform inference on the data to extract detections

To perform detection on the video sequences:
```
python detection/inference.py
```
The results will be stored in `<path_results>/<seq>/<camera>`.

## 3. Perform individual tracking.
In `config/tracking_single.yaml` the tracking method (`kalman`, `kalman_of`, `overlap`) and the `<detections_dir>`, as well as `filter_by_area` and `filter_parked` can be specified. The thresholds `filter_area_threshold` and `filter_parked_threshold` have been empirically determined but can be adapted to new sequences.

## 4. De-Duplicate detections on th MTMC setup

### Metric Learning Re-ID

### SfM Re-ID
You can use `python gta_track.py`specifying the `--sequence_path`, the `--detections_path` from the output of the previous step, `--timestamps_path` for the start timestamps of each camera, and `--path_tracking_data` for the output. This output may be used by the same parameters to visualize in the GPS projection with `gta_map.py`. 

The expected directory structure for the sequence path and timestaps_path is the official one:
``` 
--timestamps_path ./aic19/cam_timestamp/SXX.txt
--sequenc_path ./aic19/train/SXX
```
Regarding the detections_path, we expect a structure such as `tracking_restults_root/SXX/cYYY/detections.txt`.
```
--detections_path ./tracking_restults_root/SXX
```

## 5. Evalaute individual tracking.
The ground truth data must be in the appropiate format expected by the `TrackEval` library. See the [docs](https://github.com/JonathonLuiten/TrackEval/tree/master/docs/MOTChallenge-Official) for the MOTChallenge format that we follow here.

Once the data directories are in the appropiate format, simply run
```
python run_mot_challenge.py --BENCHMARK parabellum --DO_PREPROC False
```
## 6. Visualizations

### GTA Map (GPS Tracking Awesome Map)
You can use `python gta_map.py` specifying the `--sequence_path`, the `--detections_path` from the output of any sequence tracking and `--timestamps_path` for the start timestamps of each camera. The expected directory structure is the same as in `python gta_track.py`.
