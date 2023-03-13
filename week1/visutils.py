import cv2
import numpy as np
import matplotlib.pyplot as plt

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
    plt.savefig('out.png')

    return rgb

if __name__ == '__main__':

    plot_optical_flow_hsv('../data/GT_OF/000157_10.png')