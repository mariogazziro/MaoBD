# -*- coding: utf-8 -*-
"""
Image I/O and simple operations.

@author: Artem Sevastopolsky, 2016
"""

import os
import glob
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.pyplot import imshow, figure
import cv2


class ImLibException(Exception):
    pass


def _resizeImg(img, s):
    # Preservers the aspect ratio
    r = s/img.shape[1]
    dim = (int(s), int(img.shape[0]*r))
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    return resized

def _skinColorSegmentation(image):
    ycbcr = cv2.cvtColor(image, cv2.COLOR_RGB2YCR_CB)
    thr = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    #ycbcr = cv2.GaussianBlur(ycbcr, (9,9), 0)

    cb = ycbcr[:, :, 2]
    cr = ycbcr[:, :, 1]
    thr[:, :] = np.where(((cb >= 77) & (cb <= 127) &
                          (cr >= 133) & (cr <= 174)), 255, 0)

    # TODO: Histogram equalization??

    # Filling holes
    #thr = cv2.medianBlur(thr, 5)
    #fill = thr.copy()
    #h, w = thr.shape[:2]
    #mask = np.zeros((h+2, w+2), np.uint8)
    #cv2.floodFill(fill, mask, (0,0), 255);
    #nfill = cv2.bitwise_not(fill)
    #ret = thr | nfill

    ke = np.ones((3,3), np.uint8)
    kc = np.ones((5,5), np.uint8)

    ret = cv2.medianBlur(thr, 5)
    ret = cv2.erode(ret, ke)
    ret = cv2.morphologyEx(ret, cv2.MORPH_CLOSE, kc, iterations=1)

    return ret


def load_image(path):
    return np.asarray(Image.open(path)) / 255.0


def save_image(path, img):
    tmp = np.asarray(img * 255.0, dtype=np.uint8)
    Image.fromarray(tmp).save(path)


def show_image(img, fig_size=(10, 10)):
    figure(figsize=fig_size)
    #imshow(img, cmap=cm.Greys_r)
    imshow(img)
    

def intensity(img):
    return img.mean(axis=2)



def crop_black_border(img):
    row_sums = img.sum(axis=(1, 2))
    col_sums = img.sum(axis=(0, 2))
    top, bottom, left, right = -1, -1, -1, -1
    for i in range(len(row_sums)):
        if row_sums[i] > 0:
            if top == -1:
                top = i
            bottom = i
    for i in range(len(col_sums)):
        if col_sums[i] > 0:
            if left == -1:
                left = i
            right = i
    if top == -1 or left == -1:
        raise ImLibException('Image contains only black pixels')
    return img[max(top - 10, 0):(bottom + 1) + 10, max(left - 10, 0):(right + 1) + 10, :]
    


def load_set(names, shuffle=False):
    if shuffle:
        np.random.shuffle(names)
    data = []
    for img_fn in names:
        img = load_image(img_fn)
        data.append(img)
    return data, names


def image_names_in_folder(folder):
    fn = []
    for ending in ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif'):
        fn.extend(glob.glob(os.path.join(folder, ending)))
    fn.sort()
    return fn


def plot_subfigures(imgs, title=None, fig_size=None, contrast_normalize=False):
    if isinstance(imgs, list):
        # Multiple pictures in one row
        if fig_size is None:
            fig, axes = plt.subplots(nrows=1, ncols=len(imgs))
                                     #figsize=(20, 20))
        else:
            fig, axes = plt.subplots(nrows=1, ncols=len(imgs),
                                     figsize=fig_size)
        plt.gray()
        if title is not None:
            fig.suptitle(title, fontsize=12)
        for i in range(len(imgs)):
            axes[i].axis('off')
            axes[i].set_xticks([])
            axes[i].set_yticks([])
            if contrast_normalize:
                # Normalizing contrast for each image
                vmin, vmax = imgs[i].min(), imgs[i].max()
                axes[i].imshow(imgs[i], vmin=vmin, vmax=vmax)
            else:
                axes[i].imshow(imgs[i])
    elif len(imgs.shape) == 4 and imgs.shape[0] == 1:
        plot_subfigures(imgs.reshape((imgs.shape[1], imgs.shape[2], imgs.shape[3])),
                        title, fig_size, contrast_normalize)
    elif len(imgs.shape) == 2:
        # One picture
        if title is not None:
            plt.title(title)
        show_image(imgs)
    
    elif len(imgs.shape) == 3:
        # Multiple pictures in one row
        if fig_size is None:
            fig, axes = plt.subplots(nrows=1, ncols=imgs.shape[0])
                                     #figsize=(20, 20))
        else:
            fig, axes = plt.subplots(nrows=1, ncols=imgs.shape[0],
                                     figsize=fig_size)
        plt.gray()
        if title is not None:
            fig.suptitle(title, fontsize=12)
        for i in range(imgs.shape[0]):
            axes[i].axis('off')
            axes[i].set_xticks([])
            axes[i].set_yticks([])
            if contrast_normalize:
                # Normalizing contrast for each image
                vmin, vmax = imgs[i].min(), imgs[i].max()
                axes[i].imshow(imgs[i], vmin=vmin, vmax=vmax)
            else:
                axes[i].imshow(imgs[i])
            
    elif len(imgs.shape) == 4:
        # Multiple pictures in a few rows
        if fig_size is None:
            fig, axes = plt.subplots(nrows=imgs.shape[0], ncols=imgs.shape[1])
                                     #figsize=(20, 20))
        else:
            fig, axes = plt.subplots(nrows=imgs.shape[0], ncols=imgs.shape[1],
                                     figsize=fig_size)
        plt.gray()
        if title is not None:
            fig.suptitle(title, fontsize=12)
        for i in range(imgs.shape[0]):
            for j in range(imgs.shape[1]):
                axes[i][j].axis('off') 
                axes[i][j].set_xticks([])
                axes[i][j].set_yticks([])
                if contrast_normalize:
                    # Normalizing contrast for each image
                    vmin, vmax = imgs[i][j].min(), imgs[i][j].max()
                    axes[i][j].imshow(imgs[i][j], vmin=vmin, vmax=vmax)
                else:
                    axes[i][j].imshow(imgs[i][j])
    else:
        raise ImLibException("imgs array contains 3D set of images or deeper")
