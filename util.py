import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

def import_image(path):
    return cv2.imread(path, cv2.IMREAD_COLOR)

def import_label(path):
    if '.txt' in os.path.splitext(path):
        labels = []
        with open(path, "r") as f:
            while True:
                label = f.readline()
                if label == "":
                    return labels
                else:
                    label_list = label.split()
                    f_pos = [float(s) for s in label_list[1:]]
                    labels.append([int(label_list[0]), *f_pos])


def rotate_90(img, labels):
    new_labels = []
    for label in labels:
        t, x, y, w, h = label
        new_labels.append([t, y, 1.0 - x, h, w])
    return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE), convert_label_str(new_labels)


def rotate_180(img, labels):
    new_labels = []
    for label in labels:
        t, x, y, w, h = label
        new_labels.append([t, 1.0 - x, 1.0 - y, w, h])
    return cv2.rotate(img, cv2.ROTATE_180), convert_label_str(new_labels)


def rotate_270(img, labels):
    new_labels = []
    for label in labels:
        t, x, y, w, h = label
        new_labels.append([t, 1.0 - y, x, h, w])
    return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE), convert_label_str(new_labels)


def gaussian_noise(img, labels, mean=0, sigma=5):
    gauss = np.random.normal(mean, sigma, img.shape)
    new_img = img + np.floor(gauss)
    new_img[new_img > 255] = 255
    new_img[new_img < 0] = 0
    return new_img.astype(np.uint8), convert_label_str(labels)


def reverse_horizontal(img, labels):
    new_labels = []
    for label in labels:
        t, x, y, w, h = label
        new_labels.append([t, 1 - x, y, w, h])
    return cv2.flip(img, 1), convert_label_str(new_labels)


def gamma_correction(img, labels, gamma=0.5):
    x = np.arange(256)
    y = np.floor((x / 255) ** gamma * 255)
    new_img = cv2.LUT(img, y)
    new_img[new_img > 255] = 255
    new_img[new_img < 0] = 0
    return new_img.astype(np.uint8), convert_label_str(labels)


def plot_boundingbox(img, labels):
    for l in labels:
        y, x = int(img.shape[0]), int(img.shape[1])
        s_x = int(l[1] * x - ((l[3] * x)/2))
        s_y = int(l[2] * y - ((l[4] * y)/2))
        e_x = int(l[1] * x + ((l[3] * x)/2))
        e_y = int(l[2] * y + ((l[4] * y)/2))
        cv2.rectangle(img, pt1=(s_x, s_y), pt2=(e_x, e_y), color=(
            255, 0, 0), thickness=3, lineType=cv2.LINE_4)
    plt.imshow(img)

def convert_label_str(labels):
    new_label = []
    for l in labels:
        _label_str = ' '.join([str(x) for x in l])
        new_label.append(_label_str + "\n")
    return new_label
