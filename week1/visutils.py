import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

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
    image, labeled = image[:, :, 1:], image[:, :, 0]
    
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

def plot_optical_flow_quiver(path, original_image_path = None, step = 30, scale = 0.05, flow_with_camera = False):
    ofimg = read_flow(path)
    magnitude = np.hypot(ofimg[:, :, 0], ofimg[:, :, 1])
    if flow_with_camera: ofimg *= -1

    x, y = np.meshgrid(np.arange(0, magnitude.shape[1]), np.arange(0, magnitude.shape[0]))
    plt.quiver(x[::step, ::step], y[::step, ::step], ofimg[::step, ::step, 0], ofimg[::step, ::step, 1], magnitude[::step, ::step], scale_units='xy', angles='xy', scale=scale)
    
    if not original_image_path is None:
        plt.imshow(cv2.imread(original_image_path, cv2.IMREAD_GRAYSCALE), cmap='gray')
    plt.axis('off')
    plt.savefig('out2.png')

    return None

def plot_optical_flow_surface(path,  original_image_path = None):

    ofimg = read_flow(path)
    magnitude = np.hypot(ofimg[:, :, 0], ofimg[:, :, 1])

    image = cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.double)
    image, labeled = image[:, :, 1:], image[:, :, 0]

    phase = np.rad2deg(np.arctan(image[:, :, 1] / image[:, :, 0])) / 360

    norm = lambda x: (1 + ((x - x.mean()) / x.std())) / 2
    phase = np.clip(norm(phase), 0, 1)
    
    hsv = np.stack([phase, np.ones_like(phase), np.ones_like(phase)]).transpose(1, 2, 0).astype(np.float32)
    hsv[:, :, 0] *= 179
    hsv[:, :, 1:] *= 255
    rgb = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB) / 255
    rgba = np.dstack( ( rgb, labeled) )

    fig, ax = plt.subplots( subplot_kw={"projection": "3d"})

    x, y = np.meshgrid(np.arange(0, magnitude.shape[1]), np.arange(0, magnitude.shape[0]))
    surf = ax.plot_surface(x, -magnitude, -y, facecolors = rgba,
                       linewidth=0, antialiased=False)
    
    IMAGE_Z = np.ones_like(magnitude) * 0 
    if not original_image_path is None:
        rgbimage = cv2.cvtColor(cv2.imread(original_image_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB) / 255
        rgbaimg = np.dstack( ( rgb, np.ones_like(magnitude) * 0.5) )

        img = ax.plot_surface(x, IMAGE_Z, -y, facecolors = rgbimage,
                       linewidth=0, antialiased=True)

    fig.savefig('out1.png')
    





if __name__ == '__main__':

    data = '../data/GT_OF/000157_10.png'
    original = '../data/OR_IMG/157.png'

    plot_optical_flow_hsv(data,)
    plot_optical_flow_quiver(data, original_image_path=original, flow_with_camera=False)
    plot_optical_flow_surface(data, original_image_path= original)
