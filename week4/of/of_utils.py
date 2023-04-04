import os
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt


def mse(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean()


def mae(y_true, y_pred):
    return np.abs(y_true - y_pred).mean()


def norm(x): 
    return (1 + ((x - x.mean()) / x.std())) / 2


def standarize(x): 
    return (x - x.min()) / (x.max() - x.min())


def visualize_optical_flow_error(GT, OF_pred, output_dir = "./results/"):
    error_dist = u_diff, v_diff = GT[:, :, 0] - \
        OF_pred[:, :, 0], GT[:, :, 1] - OF_pred[:, :, 1]
    error_dist = np.sqrt(u_diff ** 2 + v_diff ** 2)

    max_range = int(math.ceil(np.amax(error_dist)))

    plt.clf()
    plt.figure(figsize=(8, 5))
    plt.title('MSEN Distribution')
    plt.hist(error_dist[GT[..., 2] == 1].ravel(),
             bins=30, range=(0.0, max_range))
    plt.ylabel('Count')
    plt.xlabel('Mean Square Error in Non-Occluded Areas')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "MSEN_hist.png"))


def plot_optical_flow_hsv(flow, 
                          labelled=None, 
                          hide_unlabeled=True, 
                          use_whole_range=False, 
                          value=1, 
                          onlyphase=False, 
                          onlymagnitude=False, 
                          output_dir="./results/"):
    # Normalize flow to range [0, 1]
    flow_norm = flow / (np.abs(flow).max() + 1e-8)
    u, v = flow_norm[:, :, 0], flow_norm[:, :, 1]

    # Calculate phase angle using arctan2
    phase = np.arctan2(v, u) / (2 * np.pi) % 1
    magnitude = np.sqrt(u ** 2 + v ** 2)

    # Standarize phase
    phase = standarize(phase)
    
    # Normalize magnitude
    # Credit for thr to Team1
    clip_th = np.quantile(magnitude, 0.95)
    magnitude = np.clip(magnitude, 0, clip_th)
    magnitude = magnitude / magnitude.max()

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

    h, w = flow.shape[:2]
    plt.clf()
    fig, ax = plt.subplots(figsize=(w/100, h/100))
    ax.imshow(rgb)
    ax.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'optical_flow_hsv.png'), bbox_inches='tight', pad_inches=0)


def plot_optical_flow_quiver(ofimg, original_image, output_dir = "./results", step=20, scale=.15, flow_with_camera=False):
    magnitude = np.hypot(ofimg[:, :, 0], ofimg[:, :, 1])

    if flow_with_camera:
        ofimg = np.array(ofimg)
        ofimg *= -1

    x, y = np.meshgrid(
        np.arange(0, magnitude.shape[1]), np.arange(0, magnitude.shape[0]))
    plt.clf()

    h, w = ofimg.shape[:2]
    fig, ax = plt.subplots(figsize=(w/100, h/100))
    ax.quiver(x[::step, ::step], y[::step, ::step], ofimg[::step, ::step, 0], ofimg[::step,
               ::step, 1], magnitude[::step, ::step], scale_units='xy', angles='xy', scale=scale)
    ax.imshow(original_image, cmap='gray')
    ax.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f'optical_flow_quiver{"_camera" if flow_with_camera else ""}.png'), bbox_inches='tight', pad_inches=0)


def plot_optical_flow_surface(flow, original_image, output_dir = "./results"):
    plt.clf()
    labeled = flow[..., 2]

    # Normalize flow to range [0, 1]
    flow_norm = flow / (np.abs(flow).max() + 1e-8)
    u, v = flow_norm[:, :, 0], flow_norm[:, :, 1]

    # Calculate phase angle using arctan2
    phase = np.arctan2(v, u) / (2 * np.pi) % 1
    magnitude = np.sqrt(u ** 2 + v ** 2)

    # Standarize phase
    phase = standarize(phase)

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

    rgbimage = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB) / 255
    rgbaimg = np.dstack((rgb, np.ones_like(magnitude) * 0.5))
    ax.plot_surface(x, IMAGE_Z, -y, facecolors=rgbimage,
                            linewidth=0, antialiased=True)
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, 'optical_flow_surface.png'), bbox_inches='tight', pad_inches=0)
