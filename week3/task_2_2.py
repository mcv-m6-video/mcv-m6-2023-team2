import os
import argparse
import numpy as np
import time
import matplotlib.pyplot as plt  
import matplotlib.patches as patches

from skimage import io
from __future__ import print_function
from IPython import display as dp

from utils import (
    load_predictions,
    load_annotations,
)

def __parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Road Traffic Monitoring Analysis for Video Surveillance. MCV-M6-Project, week 3, task 2.2. Team 2'
    )
    
    parser.add_argument('--path_annotations', type=str, default="data/AICity_data/train/S03/c010/ai_challenge_s03_c010-full_annotation.xml",
                    help='Path to the directory where the annotations are stored.')

    parser.add_argument('--path_results', type=str, default="./results/",
                    help='The path to the directory where the results will be stored.')
    
    parser.add_argument('--use_ground_truth', action='store_true', default=True,
                    help='Whether to use the ground truth for evaluation.')
    
    args = parser.parse_args()
    return args


def main(args: argparse.Namespace):
    if args.use_ground_truth:
        annotations = load_annotations(args.path_annotations, grouped=True) 
    else:
        raise NotImplementedError
    
    # Reference: https://github.com/telecombcn-dl/2017-persontyle/blob/master/sessions/tracking/tracking_kalman.ipynb
    
    display = True
    total_time = 0.0
    total_frames = 0
    out = []

    if display:
        plt.ion() # for iterative display
        fig, ax = plt.subplots(1, 2,figsize=(20,20))

    for frame in range(int(seq_dets[:,0].max())): # all frames in the sequence
        frame += 1 #detection and frame numbers begin at 1
        dets = seq_dets[seq_dets[:,0]==frame,2:7]   
        
        if display:
            fn = '../../../data/2DMOT2015/%s/%s/img1/%06d.jpg'%(phase,seq,frame) # read the frame
            im =io.imread(fn)
            ax[0].imshow(im)
            ax[0].axis('off')
            ax[0].set_title('Original Faster R-CNN detections')
            for j in range(np.shape(dets)[0]):
                color = colours[j]
                coords = (dets[j,0],dets[j,1]), dets[j,2], dets[j,3]
                ax[0].add_patch(plt.Rectangle(*coords,fill=False,edgecolor=color,lw=3))
                
        dets[:,2:4] += dets[:,0:2] #convert to [x1,y1,w,h] to [x1,y1,x2,y2] for the tracker input
        total_frames += 1

        if display:
            ax[1].imshow(im)
            ax[1].axis('off')
            ax[1].set_title('Tracked Targets')

        start_time = time.time()
        trackers = mot_tracker.update(dets)
        cycle_time = time.time() - start_time
        total_time += cycle_time
        
        out.append(trackers)
        for d in trackers:
            if display:
                d = d.astype(np.uint32)
                ax[1].add_patch(patches.Rectangle((d[0],d[1]),d[2]-d[0],d[3]-d[1],fill=False,lw=3,ec=colours[d[4]%32,:]))
                ax[1].set_adjustable('box-forced')

        if display:
            dp.clear_output(wait=True)
            dp.display(plt.gcf())
            time.sleep(0.000001)
            ax[0].cla()
            ax[1].cla()

    print("Total Tracking took: %.3f for %d frames or %.1f FPS"%(total_time,total_frames,total_frames/total_time))

if __name__ == "__main__":
    args = __parse_args()
    main(args)
