
import numpy as np
import tensorflow as tf
import cv2
import math

class KeyPointPair:
    def __init__(self, x0, y0, alpha_norm, c = None , classes = None):
        self.x0 = x0
        self.y0 = y0
        self.alpha_norm = alpha_norm

        self.c     = c
        self.classes   = classes

        self.label = -1
        self.score = -1

    def get_label(self):
        if self.label == -1:
            self.label = np.argmax(self.classes)

        return self.label

    def get_score(self):
        if self.score == -1:
            self.score = self.classes[self.get_label()]

        return self.score

def draw_kpp(image, g_wdt, g_hgt, kpps, labels,cls_threshold=0.8):
    image_h, image_w, _ = image.shape

    for kpp in kpps:

        x0 = int(kpp.x0)
        y0 = int(kpp.y0)

        alpha = (kpp.alpha_norm - 0.1)/0.8*math.pi
        #alpha = kpp.alpha_norm
        x1 = int(kpp.x0 + math.cos( alpha )*20)
        y1 = int(kpp.y0 + math.sin( alpha )*20)

        cv2.circle(image, (x0,y0), 4, (0,0,255), 1)
        cv2.line( image, (x0,y0), (x1,y1), (0,0,255), 1 )

    return image

def decode_netout(netout, img_w, img_h, nb_class, obj_threshold=0.70, nms_threshold=0.25):
    grid_h, grid_w, nb_kpp = netout.shape[:3]

    kpps = []

    # decode output from network
    #netout[..., :2]  = _sigmoid(netout[..., :2])
    #netout[..., 2]  = _sigmoid(netout[..., 2])
    netout[..., 3]  = _sigmoid(netout[..., 3])
    netout[..., 4:] = netout[..., 3][..., np.newaxis] * _softmax(netout[..., 4:])
    netout[..., 4:] *= netout[..., 4:] > obj_threshold

    for row in range(grid_h):
        for col in range(grid_w):
            for ikpp in range(nb_kpp):
                # from 4th element onwards are confidence and class classes
                classes = netout[row, col, ikpp, 4:]

                #if np.sum(classes)>0:

                conf = netout[row,col,ikpp,3]


                #if (conf >= obj_threshold):
                #print( "conf=", conf )


                x0, y0, alpha_norm = netout[row,col,ikpp,:3]

                x0 = ((col + _sigmoid(x0)) / grid_w) * img_w
                y0 = ((row + _sigmoid(y0)) / grid_h) * img_h

                kpp = KeyPointPair(x0, y0, alpha_norm, conf , classes)

                kpps.append(kpp)

    # remove the kepoints which are less likely than a obj_threshold
    kpps = [kpp for kpp in kpps if kpp.get_score() > obj_threshold]

    return kpps


def _sigmoid(x):
    return 1. / (1. + np.exp(-x))

def _softmax(x, axis=-1, t=-100.):
    x = x - np.max(x)

    if np.min(x) < t:
        x = x/np.min(x)*t

    e_x = np.exp(x)

    return e_x / e_x.sum(axis, keepdims=True)
