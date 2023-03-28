import os 
import cv2
from tqdm import tqdm
from utils import load_annotations, group_annotations_by_frame


annotations = load_annotations("./data/AICity_data/train/S03/c010/ai_challenge_s03_c010-full_annotation.xml", use_parked=True)
annotations = group_annotations_by_frame(annotations)

base_dir = "./week3/data"
benchmark = "parabellum"
# Sequence path will be like this: ./week3/data/gt/mot_challenge/parabellum-train/s03/gt
sequence_path = os.path.join(base_dir, "gt/mot_challenge", f"{benchmark}-train", "s03")
seqmaps_path = os.path.join(base_dir, "gt/mot_challenge", "seqmaps")
os.makedirs(os.path.join(sequence_path, "gt"), exist_ok=True)
os.makedirs(seqmaps_path, exist_ok=True)
# Path will be like this: ./week3/data/gt/mot_challenge/parabellum-train/MODEL_NAME/data/s03.txt
os.makedirs(os.path.join(base_dir, "trackers/mot_challenge", f"{benchmark}-train"), exist_ok=True)

# Write the ground truth file annotations
print("Writing ground truth file")
gt_file = open(os.path.join(sequence_path, "gt", "gt.txt"), "w")

for frame_anns in tqdm(annotations):
    for ann in frame_anns:
        x1, y1, w, h = ann.coordinates_dim
        gt_file.write(f"{ann.frame+1},{ann.track_id+1},{x1},{y1},{w},{h},1,-1,-1,-1\n")

gt_file.close()

# Write the seqinfo.ini file
print("Writing seqinfo.ini file")
video = cv2.VideoCapture("./data/AICity_data/train/S03/c010/vdo.avi")
seqinfo_file = open(os.path.join(sequence_path, "seqinfo.ini"), "w")
seqinfo_file.write(f"[Sequence]\n")
seqinfo_file.write(f"name = s03\n")
seqinfo_file.write(f"imDir = .\n")  
seqinfo_file.write(f"frameRate = {int(video.get(cv2.CAP_PROP_FPS))}\n")
seqinfo_file.write(f"seqLength = {int(video.get(cv2.CAP_PROP_FRAME_COUNT))}\n")
seqinfo_file.write(f"imWidth = {int(video.get(cv2.CAP_PROP_FRAME_WIDTH))}\n")
seqinfo_file.write(f"imHeight = {int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))}\n")
seqinfo_file.write(f"imExt = .jpg\n")   
seqinfo_file.close()

# Create seqmaps
print("Creating seqmaps")
seqmaps_file = open(os.path.join(seqmaps_path, f"{benchmark}-all.txt"), "w") 
seqmaps_file.write(f"name\n")
seqmaps_file.write(f"s03\n")
seqmaps_file.close()
seqmaps_file = open(os.path.join(seqmaps_path, f"{benchmark}-train.txt"), "w") 
seqmaps_file.write(f"name\n")
seqmaps_file.write(f"s03\n")
seqmaps_file.close()
seqmaps_file = open(os.path.join(seqmaps_path, f"{benchmark}-test.txt"), "w") 
seqmaps_file.write(f"name\n")
seqmaps_file.write(f"s03\n")
seqmaps_file.close()

print("Done!")
