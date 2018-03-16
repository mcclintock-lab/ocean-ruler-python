import cv2
import numpy as np

import utils

def get_lobster_template_contours():

    #by default, using the big abalone template
    up_shape = get_template_shape("lobster_templates/full_lobster_up.png")

    left_shape = get_template_shape("lobster_templates/full_lobster_left.png")
    right_shape = get_template_shape("lobster_templates/full_lobster_right.png")
    down_shape = get_template_shape("lobster_templates/full_lobster_down.png")

    return up_shape, left_shape, right_shape, down_shape

def get_finfish_template_contours():

    #by default, using the big abalone template
    up_shape = get_template_shape("finfish_templates/sablefish_up.png")

    left_shape = get_template_shape("finfish_templates/sablefish_left.png")
    right_shape = get_template_shape("finfish_templates/sablefish_right.png")
    down_shape = get_template_shape("finfish_templates/sablefish_down.png")

    return up_shape, left_shape, right_shape, down_shape


def get_template_shape(img_name):
    template = cv2.imread(img_name)
    template_edged = cv2.Canny(template, 10, 250)
    edged_img = cv2.dilate(template_edged, None, iterations=1)
    im2, shapes, hierarchy = cv2.findContours(edged_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    
    if len(shapes) > 0:
        tgt_shape = shapes[1]
        cv2.drawContours(im2, shapes, -1, (255,0,0), 3)
        #utils.show_img("shapes", im2)
        return tgt_shape
    else:
        return None

def get_quarter_contours(rescaled_image):
    quarter_only = cv2.imread("images/quarter_template_1280.png")
    quarter_only = quarter_only[30:len(quarter_only),30:len(quarter_only[0])-30]
    quarter_only_edged = cv2.Canny(quarter_only, 15,100)
    quarter_edged_img = cv2.dilate(quarter_only_edged, np.ones((1,1), np.uint8), iterations=2)
    quarter_e, quarter_shapes, hierarchy2 = cv2.findContours(quarter_edged_img,  cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    quarter_shape = quarter_shapes[0] 
    return quarter_shape

