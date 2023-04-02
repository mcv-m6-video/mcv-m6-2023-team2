import os
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt


def mse(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean()


def mae(y_true, y_pred):
    return np.abs(y_true - y_pred).mean()


def visualize_optical_flow_error(GT, OF_pred, frame):
    error_dist = u_diff, v_diff = GT[:, :, 0] - \
        OF_pred[:, :, 0], GT[:, :, 1] - OF_pred[:, :, 1]
    error_dist = np.sqrt(u_diff ** 2 + v_diff ** 2)

    max_range = int(math.ceil(np.amax(error_dist)))

    plt.figure(figsize=(8, 5))
    plt.hist(error_dist[GT[..., 2] == 1].ravel(),
             bins=30, range=(0.0, max_range))
    plt.title('MSEN Distribution')
    plt.ylabel('Count')
    plt.xlabel('Mean Square Error in Non-Occluded Areas')
    plt.savefig(f'./results/MSEN_hist_{frame}.png')


def norm(x): 
    return (1 + ((x - x.mean()) / x.std())) / 2


def standarize(x): 
    return (x - x.min()) / (x.max() - x.min())


def plot_optical_flow_hsv(flow, 
                          labelled=None, 
                          normalize=True, 
                          hide_unlabeled=True, 
                          use_whole_range=False, 
                          value=1, 
                          onlyphase=False, 
                          onlymagnitude=False, 
                          output_dir = "./results/"
                          ):
    phase = np.rad2deg(np.arctan(flow[:, :, 0] / flow[:, :, 1])) / 360
    magnitude = (flow[:, :, 0] ** 2 + flow[:, :, 1] ** 2) ** .5

    if normalize:
        phase = np.clip(norm(phase), 0, 1)
        magnitude = np.clip(norm(magnitude), 0, 1)
    else:
        phase = standarize(phase)
        magnitude = standarize(magnitude)

    if use_whole_range:
        phase = cv2.equalizeHist((255 * phase).astype(np.uint8)) / 255

    hsv = np.stack([phase, magnitude, np.ones_like(phase) * value]
                   ).transpose(1, 2, 0).astype(np.float32)

    if onlymagnitude:
        hsv[:, :, 0] = .5
    elif onlyphase:
        hsv[:, :, 1:] = 1

    hsv[:, :, 0] *= 179
    hsv[:, :, 1:] *= 255

    rgb = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    if hide_unlabeled:
        rgb = rgb * labelled.astype(np.uint8)[:, :, None]

    plt.axis('off')
    plt.imshow(rgb)
    plt.savefig(os.path.join(output_dir, 'optical_flow_hsv.png'))


def plot_optical_flow_quiver(ofimg, original_image, output_dir = "./results", step=30, scale=0.05, flow_with_camera=False):
    magnitude = np.hypot(ofimg[:, :, 0], ofimg[:, :, 1])

    if flow_with_camera:
        ofimg *= -1

    x, y = np.meshgrid(
        np.arange(0, magnitude.shape[1]), np.arange(0, magnitude.shape[0]))
    plt.quiver(x[::step, ::step], y[::step, ::step], ofimg[::step, ::step, 0], ofimg[::step,
               ::step, 1], magnitude[::step, ::step], scale_units='xy', angles='xy', scale=scale)

    if not original_image is None:
        plt.imshow(original_image, cmap='gray')

    plt.axis('off')
    plt.savefig('./results/optical_flow_quiver.png')


def plot_optical_flow_surface(path,  original_image_path=None):
    ofimg = read_flow(path)
    magnitude = np.hypot(ofimg[:, :, 0], ofimg[:, :, 1])

    image = cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.double)
    image, labeled = image[:, :, 1:], image[:, :, 0]

    phase = np.rad2deg(np.arctan(image[:, :, 1] / image[:, :, 0])) / 360

    def norm(x): 
        return (1 + ((x - x.mean()) / x.std())) / 2
    
    phase = np.clip(norm(phase), 0, 1)

    hsv = np.stack([phase, np.ones_like(phase), np.ones_like(phase)]).transpose(
        1, 2, 0).astype(np.float32)
    hsv[:, :, 0] *= 179
    hsv[:, :, 1:] *= 255
    rgb = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB) / 255
    rgba = np.dstack((rgb, labeled))

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    x, y = np.meshgrid(
        np.arange(0, magnitude.shape[1]), np.arange(0, magnitude.shape[0]))
    
    surf = ax.plot_surface(x, -magnitude, -y, facecolors=rgba,
                           linewidth=0, antialiased=False)

    IMAGE_Z = np.ones_like(magnitude) * 0

    if not original_image_path is None:
        rgbimage = cv2.cvtColor(cv2.imread(
            original_image_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB) / 255
        rgbaimg = np.dstack((rgb, np.ones_like(magnitude) * 0.5))

        img = ax.plot_surface(x, IMAGE_Z, -y, facecolors=rgbimage,
                              linewidth=0, antialiased=True)

    fig.savefig('out1.png')
