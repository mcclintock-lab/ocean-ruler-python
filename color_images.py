import cv2
import utils
import numpy as np
from imutils import perspective
from imutils import contours
import imutils
import matching

def get_color_image(orig_image, hue_offset):

    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2HSV)
    if len(image) > 500:
        mask = cv2.inRange(image, np.array([0,0,255]), np.array([300,7,255]))
        notmask = cv2.bitwise_not(mask)
        image = cv2.bitwise_and(orig_image,orig_image,mask=notmask)
        
    hues = []
    sats = []
    vals = []
    sat_offset =  10
    val_offset = 50


    for i in range(100,120):
        for j in range(84,86):
            val = orig_image[i,j]
            huemin = get_min(val[0]-hue_offset)
            satmin = get_min(val[1]-(hue_offset+sat_offset))
            valmin = get_min(val[2]-(hue_offset+val_offset))
            
            huemax = get_max(int(val[0])+hue_offset)
            satmax = get_max(int(val[1])+(hue_offset+sat_offset))
            valmax = get_max(int(val[2])+(hue_offset+val_offset))
            bl = np.array([huemin, satmin, valmin])
            bu = np.array([huemax, satmax, valmax])

            #use original image so we get the non-masked values
            mask = cv2.inRange(orig_image, bl, bu)
            notmask = cv2.bitwise_not(mask)

            image = cv2.bitwise_and(image,image,mask=notmask)
    
    bgr = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
    return bgr


def get_image_with_color_mask(input_image, thresh_val, blur_window, show_img):

    rows = len(input_image)
    cols = len(input_image[0])
    image = input_image
    res = get_color_image(image, thresh_val+blur_window)
    
    #maybe use this to see if we should threshold?
    gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (blur_window, blur_window), 0)
    
    retval, threshold = cv2.threshold(gray.copy(),1,255,cv2.THRESH_BINARY)
    if show_img:
        utils.show_img("color mask thresh {};{}".format(thresh_val, blur_window),gray)
    return image, None, res, rows

def do_color_image_match(input_image, template_contour, thresh_val, blur_window, showImg, contour_color, is_ruler, use_gray_threshold):
    #try the color image
    color_image, gray, thresh1, mid_row = get_image_with_color_mask(input_image, thresh_val, blur_window, showImg)
    edged = utils.find_edges(gray, thresh1, use_gray_threshold, showImg)

    # find contours in the edge map
    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST,
        cv2.CHAIN_APPROX_NONE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]

    contours, size = utils.get_largest_edge(cnts)
    if contours is None:
        return None, None, None,None,None

    smallest_combined = 10000000.0
    target_contour = None
    rval = 1000000
    aperc = 1000000
    adist = 1000000
    cdiff = 1000000
    if showImg:
        print "num contours: {}".format(len(contours))
    for contour in contours:

        the_contour, result_val, area_perc, area_dist, centroid_diff, shape_dist= matching.sort_by_matching_shape(contour, template_contour, False,input_image)
        
        comb = result_val*area_dist

        if comb < smallest_combined:
            smallest_combined = comb
            rval = result_val
            aperc = area_perc
            adist = area_dist
            target_contour = the_contour
            cdiff = centroid_diff

            if showImg:
                utils.show_img_and_contour("big color img {}x{}; shape dist: {}, haus_dist:{}".format(thresh_val, blur_window, shape_dist, area_dist), input_image, target_contour, template_contour,0)

    #epsilon = 0.003*cv2.arcLength(target_contour,True)
    #approx = cv2.approxPolyDP(target_contour,epsilon,True)
    return target_contour, rval, aperc, adist, cdiff



def get_min(val):
    minval = np.min(val)
    if minval < 0:
        return 0
    else:
        return minval

def get_max(val):
    maxval = np.amax(val)
    if maxval > 255:
        return 255
    else:
        return maxval