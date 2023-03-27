# import cv2

from utils import (
    iou_over_time,
    load_annotations,
    load_predictions,
)


# def run_inference(args):
#     gt = read_annotations(args.gt_path, grouped=True, use_parked=True)
#     det = read_detections(det_path, grouped=False)

#     det = group_by_frame(filter_by_conf(det, conf_thr=args.min_conf))

#     vidcap = cv2.VideoCapture(args.video_path)
#     # vidcap.set(cv2.CAP_PROP_POS_FRAMES, initial_frame)  # to start from frame #frame_id
#     num_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

#     for frame_id in range(num_frames):
#         _, frame = vidcap.read()

#         if frame_id >= 1755 and frame_id <= 1835:
#             frame = draw_boxes(frame, gt[frame_id], color='g')
#             frame = draw_boxes(frame, det[frame_id], color='b', det=True)

#             cv2.imshow('frame', frame)
#             if cv2.waitKey() == 113:  # press q to quit
#                 break

#     cv2.destroyAllWindows()


def task_1_1_visualization(args):

    gt =  load_annotations(args.path_GT, grouped=False, use_parked=True)
    det = load_predictions(args.path_det, grouped=False)

    mean_iou = iou_over_time(
        video_path=args.path_video,
        annotations=gt,
        predictions=det,
        max_frames=args.num_frames,
        frame_sampling_each=4,
        save_path=args.path_results,
    )

    print("Mean IoU: ", mean_iou)
    print("Results saved to: ", args.path_results)
    print("-" * 50)
