import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from tqdm import tqdm
import os

def standarize(x): 
    return (x - x.min()) / (x.max() - x.min())

def norm(x): 
    return (1 + ((x - x.mean()) / x.std())) / 2


def read_flow(path:str):
    """

    I went crazy with the flow format.
        Credits: https://github.com/mcv-m6-video/mcv-m6-2021-team1/blob/main/week1/task_04_flow.ipynb
    Used for the quiver plot; the other one is more low level then has no problems.

    Optical flow maps are saved as 3-channel uint16 PNG images: The first channel
    contains the u-component, the second channel the v-component and the third
    channel denotes if a valid ground truth optical flow value exists for that
    pixel (1 if true, 0 otherwise)
    """
    # cv2 flips the order of reading channels
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.double)
    
    # valid channel
    valid = img[:,:,0]
    
    # get flow vectors
    u_flow = (img[:,:,2] - 2**15)/64
    v_flow = (img[:,:,1] - 2**15)/64
    
    # remove invalid flow values
    u_flow[valid == 0] = 0
    v_flow[valid == 0] = 0
    
    # return image in correct order
    return np.dstack((u_flow, v_flow, valid))


def plot_optical_flow_hsv(path, normalize = 1, hide_unlabeled = 1, use_whole_range = 0, value = 1, onlyphase = False, onlymagnitude = False):

    assert (not onlyphase or not onlymagnitude) or (onlymagnitude != onlyphase), 'Do you want phase or magnitude??'

    ###### FOR FUTURE STUDENTS ######
    # channel 3: i component.       #
    # channel 1: j component.       #
    # channel 0: labeled (binary)   #
    #                               #
    # no problem                    #
    # main idea for this plot       #
    #                               #
    #  We have:                     #
    #     1. direction data         #
    #     3. modulus data           #
    #  We will impose an HSV mask   #
    #    direction: H   (arctan(b/a))/360             
    #    modulus: S                 #
    #                               #
    #################################
            
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.double)
    image, labeled = image[:, :, 1:], image[:, :, 0] # v u
    
    phase = np.rad2deg(np.arctan(image[:, :, 1] / image[:, :, 0])) / 360
    magnitude = (image[:, :, 1] ** 2  + image[:, :, 0] ** 2) ** .5
    
    norm = lambda x: (1 + ((x - x.mean()) / x.std())) / 2
    standarize = lambda x: (x - x.min()) / (x.max() - x.min())
    if normalize:

        phase = np.clip(norm(phase), 0, 1)
        magnitude = np.clip(norm(magnitude), 0, 1)

    else: 

        phase = standarize(phase)
        magnitude = standarize(magnitude)
    
    if use_whole_range:
        phase = cv2.equalizeHist((255 * phase).astype(np.uint8)) / 255

    hsv = np.stack([phase, magnitude, np.ones_like(phase) * value]).transpose(1, 2, 0).astype(np.float32)

    if onlymagnitude: hsv[:, :, 0] = .5
    elif onlyphase: hsv[:, :, 1:] = 1
    hsv[:, :, 0] *= 179
    hsv[:, :, 1:] *= 255

    rgb = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    if hide_unlabeled: rgb = rgb * labeled.astype(np.uint8)[:, :, None]
    
    plt.imshow(rgb)
    plt.axis('off')
    plt.savefig('out3.png')

    return None

def ofmedian(magnitude, size):
    
    maxmag = magnitude.max()
    minmag = magnitude.min()

    magnitude = (magnitude - minmag) / (maxmag - minmag)
    magnitude = cv2.blur((magnitude * 255).astype(np.uint8), (size, size)).astype(np.float32)
    magnitude /= 255
    magnitude = magnitude * (maxmag) + minmag 
    return magnitude

def plot_optical_flow_quiver(path, original_image_path = None, step = 30, scale = 0.05, flow_with_camera = False, median = 7):
    ofimg = read_flow(path)
    magnitude = np.hypot(ofimg[:, :, 0], ofimg[:, :, 1])
    mask = magnitude != magnitude.min()

    if flow_with_camera: ofimg *= -1 

    x, y = np.meshgrid(np.arange(0, magnitude.shape[1]), np.arange(0, magnitude.shape[0]))
    plt.quiver(x[::step, ::step], y[::step, ::step], ofimg[::step, ::step, 0], ofimg[::step, ::step, 1], magnitude[::step, ::step], scale_units='xy', angles='xy', scale=scale)
    
    if not original_image_path is None:
        if isinstance(original_image_path, str): im = cv2.imread(original_image_path, cv2.IMREAD_GRAYSCALE)
        else: im = original_image_path
        plt.imshow(im,  cmap='gray')
    plt.axis('off')
    plt.savefig('out2.png')
    plt.clf()

    return  cv2.cvtColor(cv2.imread('out2.png', cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

def plot_optical_flow_surface(path,  original_image_path = None):
    plt.close()

    flow = read_flow(path)

    image = cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.double)
    image, labeled = image[:, :, 1:], image[:, :, 0]

    flow_norm = flow / (np.abs(flow).max() + 1e-8)
    u, v = flow_norm[:, :, 0], flow_norm[:, :, 1]

    phase =  np.arctan2(v, u) / (2 * np.pi) % 1
    magnitude = np.sqrt(u ** 2 + v ** 2)

    norm = lambda x: (1 + ((x - x.mean()) / x.std())) / 2
    phase = np.clip(norm(phase), 0, 1)
    
    hsv = np.stack([phase, np.ones_like(phase), np.ones_like(phase)]).transpose(1, 2, 0).astype(np.float32)
    hsv[:, :, 0] *= 179
    hsv[:, :, 1:] *= 255

    labeled = magnitude != magnitude.min()
    rgb = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB) / 255
    rgba = np.dstack( ( rgb, labeled) )

    fig, ax = plt.subplots( subplot_kw={"projection": "3d"})

    x, y = np.meshgrid(np.arange(0, magnitude.shape[1]), np.arange(0, magnitude.shape[0]))
    surf = ax.plot_surface(x, -5 *magnitude - .5, -y, facecolors = rgba,
                       linewidth=0, antialiased=False)
    ax.set_ylim(-14, None)
    
    IMAGE_Z = np.ones_like(magnitude) * 0  
    if not original_image_path is None:
        if isinstance(original_image_path, str): im = cv2.imread(original_image_path, cv2.IMREAD_COLOR)
        else: im = original_image_path
        rgbimage = cv2.cvtColor(im, cv2.COLOR_BGR2RGB) / 255
        rgbaimg = np.dstack( ( rgb, np.ones_like(magnitude) * 0.5) )

        img = ax.plot_surface(x, IMAGE_Z, -y, facecolors = rgbimage,
                       linewidth=0, antialiased=True)

    fig.savefig('out1.png')
    return cv2.imread('out1.png', cv2.IMREAD_COLOR)
    


def record_3d_video(seq_video_path, results_path, vidout = 'outpy.avi'):

    results = sorted([results_path + r for r in os.listdir(results_path) if 'png' in r],)
    video_handler = cv2.VideoCapture(seq_video_path)

    W = 480
    H = 640

    out = cv2.VideoWriter('3d_'+vidout, cv2.VideoWriter_fourcc('M','J','P','G'), video_handler.get(cv2.CAP_PROP_FPS), (H, W))

    num_frames = int(video_handler.get(cv2.CAP_PROP_FRAME_COUNT))
    for frame_id in tqdm(range(num_frames - 1)):
        _, image = video_handler.read()
        plot_3d = plot_optical_flow_surface(results[frame_id], image)
        out.write(plot_3d)


    out.release()
    video_handler.release()

def plot_optical_flow_quiver_with_centroids(path, original_image_path = None, centroids = [(128, 128)], step = 30, scale = 1, flow_with_camera = False):
    plt.close()
    # THIS IS A TODO; CENTROIDS ARE NOT PROPERLY WORKING I THINK

    if flow_with_camera: ofimg *= -1

    Xs = [x[0] for x in centroids]
    Ys = [y[1] for y in centroids]
    plt.quiver(Ys, Xs, ofimg[Xs, Ys, 0], ofimg[Xs, Ys, 1], magnitude[Xs, Ys], scale_units='xy', angles='xy', scale=scale)
    
    if not original_image_path is None:

        if isinstance(original_image_path, str): im = cv2.imread(original_image_path, cv2.IMREAD_GRAYSCALE)
        else: im = original_image_path
        plt.imshow(im,  cmap='gray')
    plt.axis('off')
    plt.savefig('out2.png')

    return cv2.cvtColor(cv2.imread('out2.png', cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

def video_optical_quiver(seq_video_path, results_path, detections = None, vidout = 'outquiver.avi'):

    results = sorted([results_path + r for r in os.listdir(results_path) if 'png' in r],)
    if isinstance(detections, str):
        detections = open(detections, 'r').readlines()
        detections_lut = {}

        for line in detections:

            line = line.strip().split(',')
            if not line[0] in detections_lut: detections_lut[int(line[0])] = []

            centroid = [float(line[2]) + .5 * float(line[4]), float(line[3]) + .5* float(line[5])]
            detections_lut[int(line[0])].append([int(i) for i in centroid][::-1])

    video_handler = cv2.VideoCapture(seq_video_path)

    W = 480
    H = 640

    out = cv2.VideoWriter('quiver_'+vidout, cv2.VideoWriter_fourcc('M','J','P','G'), video_handler.get(cv2.CAP_PROP_FPS), (H, W))

    num_frames = int(video_handler.get(cv2.CAP_PROP_FRAME_COUNT))
    for frame_id in tqdm(range(num_frames - 1)):
        _, image = video_handler.read()
        plot_3d = plot_optical_flow_quiver(results[frame_id], image,) #  centroids=detections_lut[frame_id + 1]
        out.write(plot_3d)
        
    out.release()
    video_handler.release()   

if __name__ == '__main__':

    data = '../data/GT_OF/000157_10.png'
    original = '../data/OR_IMG/157.png'

    video = 'OFs/c010/vdo.avi'
    results = 'OFs/video_of_blockmatching_S03_c010/'
    #detections = 'OFs/c010/det/det_mask_rcnn.txt'

    videos = [
        '../data/aic19/train/S01/c003/vdo.avi',
        '../data/aic19/train/S01/c003/vdo.avi',
        
        '../data/aic19/train/S03/c010/vdo.avi',
        '../data/aic19/train/S03/c010/vdo.avi',

        '../data/aic19/train/S03/c013/vdo.avi',
        '../data/aic19/train/S03/c013/vdo.avi',

        '../data/aic19/train/S04/c016/vdo.avi',
        '../data/aic19/train/S04/c016/vdo.avi',
        
        ]
    results = [
        './results/video_of_blockmatching_S01_c003/',
        './results/video_of_unimatch_S01_c003/',

        './results/video_of_blockmatching_S03_c010/',
        './results/video_of_unimatch_S03_c010/',

        './results/video_of_blockmatching_S03_c013/',
        './results/video_of_unimatch_S03_c013/',

        './results/video_of_blockmatching_S04_c016/',
        './results/video_of_unimatch_S04_c016/'
        
    ]
    outnames = [
        'S01_c003_BLOCKMATCH',
        'S01_c003_UNIMATCH',

        'S03_c010_BLOCKMATCH',
        'S03_c010_UNIMATCH',

        'S03_c013_BLOCKMATCH',
        'S03_c013_UNIMATCH',


        'S04_c016_BLOCKMATCH',
        'S04_c016_UNIMATCH',
    ]
    

    for video, result, outname in zip(videos, results, ['videos/'+x for x in outnames]):
        print(f"\nProcessing:\n\t Video: {video}\n\t Result: {result}\n\tSaving at: {outname}")
        video_optical_quiver(video, result, vidout =outname)
        record_3d_video(video, result, vidout = outname)
    