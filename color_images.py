import cv2
import utils
import numpy as np
from imutils import perspective
from imutils import contours
import imutils
import matching

def get_color_image(orig_image, hue_offset, first_pass=True):

    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2HSV)
    if len(image) > 500:
        mask = cv2.inRange(image, np.array([0,0,255]), np.array([300,7,255]))
        notmask = cv2.bitwise_not(mask)
        image = cv2.bitwise_and(orig_image,orig_image,mask=notmask)
        
    sat_offset =  10
    val_offset = 50

    #make this adjust to look for background with color?
    rows = len(orig_image)
    cols = len(orig_image[0])
    row_starts = [100, rows-105]
    row_ends = [115, rows-95]
    if first_pass:
        col_starts = [100, cols-75]
        col_ends = [103, cols-70]
    else:
        col_starts = [75, cols-75]
        col_ends = [85, cols-70]

    #fix this - figure out how to make it a mask of ones and pull out the right bits...
    for x in range(0,2):
        row_start = row_starts[x]
        col_start = col_starts[x]
        row_end = row_ends[x]
        col_end = col_ends[x]
        for i in range(row_start,row_end):
            for j in range(col_start,col_end):
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


def get_image_with_color_mask(input_image, thresh_val, blur_window, show_img,first_pass=True):

    rows = len(input_image)
    cols = len(input_image[0])
    image = input_image
    res = get_color_image(image, thresh_val+blur_window, first_pass=first_pass)
    
    #maybe use this to see if we should threshold?
    gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (blur_window, blur_window), 0)
    
    retval, threshold = cv2.threshold(gray.copy(),1,255,cv2.THRESH_BINARY)
    if show_img:
        utils.show_img("color img, one thats returned {};{}".format(thresh_val, blur_window),res)
        utils.show_img("thresh {};{}".format(thresh_val, blur_window),threshold)
    return image, threshold, res, rows

def do_color_image_match(input_image, template_contour, thresh_val, blur_window, showImg=False, 
    contour_color=None, is_ruler=False, use_gray_threshold=False, enclosing_contour=None, first_pass=True):
    #try the color image
    color_image, gray_img, threshold_img, mid_row = get_image_with_color_mask(input_image, thresh_val, 
        blur_window, showImg, first_pass=first_pass)
    erode_iterations = 1 if is_ruler else 3
    edged = utils.find_edges(img=gray_img, thresh_img=threshold_img, use_gray=use_gray_threshold, showImg=showImg, erode_iterations=erode_iterations)

    # find contours in the edge map
    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]

    if is_ruler:
        contours= utils.get_large_edges(cnts)
    else:
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
        print "number of contours: {}".format(len(contours))
        cv2.drawContours(input_image, contours, -1, (255,255,200), 2)
        cv2.imshow("all contours", input_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    for contour in contours:

        the_contour, result_val, area_perc, area_dist, centroid_diff, shape_dist= matching.sort_by_matching_shape(contour, template_contour, False,input_image)
        #ditch the outliers -- this is fine tuned later
        if area_perc < 0.25 or area_perc > 2.0:
            continue

        if is_ruler:
            print "here..."
            
            x,y,w,h = cv2.boundingRect(contour)
            tgt_ratio = 0.50
            ratio = float(w)/float(h)
            if ratio < tgt_ratio or 1.0/ratio < tgt_ratio:
                print "skipping non-circular contour, ratio is {}".format(ratio)
                continue
            else:
                print "this ratio is ok: {}".format(ratio)
            

            #if its the ruler (quarter), check to see if its enclosed
            is_enclosed = utils.is_contour_enclosed(the_contour, enclosing_contour)
            if is_enclosed:
                continue

        comb = result_val*area_dist

        if comb < smallest_combined:

            if (not is_ruler) or (is_ruler and area_perc > 0.25):
                smallest_combined = comb
                rval = result_val
                aperc = area_perc
                adist = area_dist
                target_contour = the_contour
                cdiff = centroid_diff

    if showImg:
        if is_ruler:
            utils.show_img_and_contour("big color img {}x{}; shape dist: {}, val:{}".format(thresh_val, blur_window, shape_dist, rval), input_image, target_contour, enclosing_contour,0)

        else:
            utils.show_img_and_contour("big color img {}x{}; shape dist: {}, val:{}".format(thresh_val, blur_window, shape_dist, rval), input_image, target_contour, template_contour,0)

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