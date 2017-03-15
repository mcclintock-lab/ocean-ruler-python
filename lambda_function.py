# import the necessary packages
import numpy as np
import argparse
import cv2
import sys
import base64
import time
import threading
import logging

#my files
import matching 
import utils
import color_images as ci
import file_utils
import boto3
import time
from boto3.dynamodb.types import Binary
from boto3.dynamodb.conditions import Key, Attr
from botocore.exceptions import ClientError
import decimal
import json
'''
logging.basicConfig(level=logging.DEBUG,
                    format='(%(threadName)-10s) %(message)s',
                    )
logger = logging.getLogger('abalone_length')

fh = logging.FileHandler('output.log')
fh.setLevel(logging.DEBUG)
logger.addHandler(fh)
'''

ABALONE = "abalone"
RULER = "ruler"
QUARTER = "_quarter"
_start_time = time.time()

class DecimalEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, decimal.Decimal):
            if o % 1 > 0:
                return float(o)
            else:
                return int(o)
        return super(DecimalEncoder, self).default(o)

def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

def rescale(orig_cols, orig_rows, template_img):
    template_cols = len(template_img[0])
    template_rows = len(template_img)

    fx = float(orig_cols)/float(template_cols)
    fy = float(orig_rows)/float(template_rows)

    scaled_image = cv2.resize(template_img, (0,0), fx = fx, fy = fy)
    return scaled_image

def get_template_contours(rescaled_image):
    row_offset = 30
    col_offset = 30

    orig_cols = len(rescaled_image[0]) 
    orig_rows = len(rescaled_image)

    #by default, using the big abalone template
    abalone_template = cv2.imread("images/big_abalone_only_2x.png")
    rescaled_ab_template = rescale(orig_cols, orig_rows, abalone_template)
    abalone_template = rescaled_ab_template[30:len(rescaled_ab_template),30:len(rescaled_ab_template[0])-30]

    small_abalone_template = cv2.imread("images/abalone_only_2x.png")
    rescaled_small_ab_template = rescale(orig_cols, orig_rows, small_abalone_template)
    small_abalone_template = rescaled_small_ab_template[30:len(rescaled_small_ab_template),30:len(rescaled_small_ab_template[0])-30]

    quarter_only = cv2.imread("images/quarter_template_1280.png")
    quarter_only = quarter_only[30:len(quarter_only),30:len(quarter_only[0])-30]

    template_edged = cv2.Canny(abalone_template, 15, 100)
    small_template_edged = cv2.Canny(small_abalone_template, 15, 100)
    quarter_only_edged = cv2.Canny(quarter_only, 15,100)

    edged_img = cv2.dilate(template_edged, None, iterations=1)
    small_edged_img = cv2.dilate(small_template_edged, None, iterations=1)
    quarter_edged_img = cv2.dilate(quarter_only_edged, None, iterations=1)

    im2, abalone_shapes, hierarchy = cv2.findContours(edged_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    abalone_shape = abalone_shapes[1]

    small_im, small_abalone_shapes, small_hierarchy = cv2.findContours(small_edged_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    small_abalone_shape = small_abalone_shapes[1]

    quarter_e, quarter_shapes, hierarchy2 = cv2.findContours(quarter_edged_img,  cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    quarter_shape = quarter_shapes[0] 

    return abalone_shape, small_abalone_shape,quarter_shape

def get_width_from_ruler(dB, rulerWidth):
    return dB/float(rulerWidth)

def get_scaled_image(image_full):
    target_cols = 1280.0
    #target_rows = 960.0

    orig_cols = len(image_full[0]) 
    orig_rows = len(image_full)

    target_rows = (float(orig_rows)/(float(orig_cols))*1280.0)
    fx = float(target_cols/orig_cols)
    fy = float(target_rows/orig_rows)

    scaled_image = cv2.resize( image_full, (0,0), fx = fx, fy = fy)
    
    rows = int(len(scaled_image))
    cols = int(len(scaled_image[0]))

    #scaled_image = scaled_image[30:rows,50:cols-50]

    #return image_full, orig_rows-30, orig_cols-100
    return scaled_image, rows, cols


def get_bw_quarter_image(input_image, thresh_val, blur,use_adaptive=True, showImg=False):

    rows = len(input_image)
    cols = len(input_image[0])      

    #maybe use this to see if we should threshold?
    gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (blur, blur), 0)    
   
    threshold = gray.copy()
    val = thresh_val*4+blur

    #pts = utils.get_points(rows, cols, True)

    #retval, threshold = cv2.threshold(threshold,val,255,cv2.THRESH_BINARY)
    if use_adaptive:
        thresh2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,23,3)
    else:
        retval, thresh2 = cv2.threshold(threshold,val,255,cv2.THRESH_BINARY)
    if showImg:
        utils.show_img("thresh 2 {}x{}".format(thresh_val, blur), thresh2)

    return input_image, gray, thresh2, int(rows/2)


def get_ruler_image(input_image, thresh_val, blur_window,showImg=False):

    rows = len(input_image)
    cols = len(input_image[0])      

    #maybe use this to see if we should threshold?
    gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

    gray = cv2.GaussianBlur(gray, (blur_window, blur_window), 0)    

    #assumes the abalone is centered
    mid_row_start = int(rows/3)*2 - thresh_val
    mid_col_start = int(cols/2) - thresh_val

    mid_row_end = mid_row_start+thresh_val
    mid_col_end = mid_col_start+thresh_val

    mid_patch = gray[mid_row_start:mid_row_end, mid_col_start:mid_col_end]
    mn = np.mean(mid_patch) 

    num_x = len(gray[0]) 
    num_y = len(gray)
    x_chunk = num_x/2
    y_chunk = num_y
   
    threshold = gray.copy()

    corner = 0
    for x in range(0,2):
        for y in range(0,1):
            start_x = x*x_chunk
            start_y = y*y_chunk
            tile = gray[start_y:start_y+y_chunk,start_x:start_x+x_chunk]
            retval, tile_thresh = cv2.threshold(tile,50,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            
            #show_img("chunk {}".format(r+c), tile_thresh)
            threshold[start_y:start_y+y_chunk,start_x:start_x+x_chunk] = tile_thresh
            corner+=1
    
    if showImg:
        utils.show_img("{}:{}".format(thresh_val, blur_window), threshold)
    return input_image, gray, threshold, int(rows/2)


def get_bw_image(input_image, thresh_val, blur_window, use_gray):

    rows = len(input_image)
    cols = len(input_image[0])    
    image = input_image

    #maybe use this to see if we should threshold?
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    gray = cv2.GaussianBlur(gray, (blur_window, blur_window), 0)
 

    is_bright = utils.is_color(input_image)
    if not is_bright:
        retval, threshold = cv2.threshold(gray,140,255,cv2.THRESH_BINARY)
    else:
        threshold = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                cv2.THRESH_BINARY,11,3)

    if False:
        utils.show_img("threshold {}x{}".format(thresh_val, blur_window),threshold)
    return image, gray, threshold, int(rows/2)



def read_args():
    ap = argparse.ArgumentParser()
    
    ap.add_argument("--image", required=False,
        help="path to the input image")
    ap.add_argument("--show", required=False,
        help="show the results. if not set, the results will write to a csv file")
    ap.add_argument("--output_file", required=False,
        help="file to read/write results from")

    try:
        args = vars(ap.parse_args())
        if args['image'] is None:
            ap.add_argument('allimages', metavar='fp', nargs='+', help='file names')
            args = vars(ap.parse_known_args())
    except SystemExit, e:
        ap.add_argument('allimages', metavar='fp', nargs='+', help='file names')
        args = vars(ap.parse_args())  
    
    showResults = args["show"]
    showResults = bool(showResults)

    out_file = args['output_file']
    if not out_file:
        out_file ="data.csv"


    imageName = args["image"]
    if imageName is None or len(imageName) == 0:
        showResults = False
        out_file ="data.csv"
        #out_file = "data.csv"
        allImageNames = args['allimages'][0]

        imageParts = allImageNames.split()

        if(len(imageParts) > 1):
            imageName = "{} {}".format(imageParts[0], imageParts[1])
        else:
            imageName = imageParts[0]

    rulerWidth = 8.5

    return imageName, showResults, rulerWidth, out_file


def draw_contour(base_img, con, pixelsPerMetric, pre, top_offset, rulerWidth,is_quarter,draw_text):
    if pre == "Ellipse":

        pts = cv2.boxPoints(con)
        center_x = int(con[0][0])
        center_y = int(con[0][1])
        size = con[1]
        width = int(size[1]/2)
        height = int(size[0])
        angle = con[2]
        left_edge = (center_x-width, center_y)
        right_edge = (center_x+width, center_y)
        cv2.circle(base_img, (center_x, center_y), 10, (255, 255,255), -1)

        cv2.line(base_img,left_edge,right_edge,(50, 50,50),4)

        dB = abs(right_edge[0] - left_edge[0])
        dimB = dB / pixelsPerMetric

        return pixelsPerMetric,dimB
    else:
        brect = cv2.boundingRect(con)

    if is_quarter:
        rulerWidth = 0.955

    # unpack the ordered bounding box, then compute the midpoint
    # between the top-left and top-right coordinates, followed by
    # the midpoint between bottom-left and bottom-right coordinates
    x = brect[0]
    y=brect[1]
    y = brect[1] + top_offset
    width=brect[2]
    height=brect[3]
    tl = (x, y+height)
    tr = (x+width, y+height)
    bl = (x,y)
    br = (x+width, y)
    corners = [tl, tr, br, bl]
    #the abalone is rotated vertically in the image

    #print "height is: ", height, " width is ", width, "ratio is ", float(height)/float(width)
    #getting rid of this for now, causes more problems than its worth
    if False:
        # compute the midpoint between the top-left and top-right points,
        # followed by the midpoint between the top-righ and bottom-right
        startLinePoint = midpoint(tl, tr)
        startLinePoint = (int(startLinePoint[0]), int(startLinePoint[1]))
        endLinePoint = midpoint(bl, br)
        endLinePoint = (int(endLinePoint[0]), int(endLinePoint[1]))
        dB = abs(startLinePoint[1] - endLinePoint[1])
    else:
        # compute the midpoint between the top-left and top-right points,
        # followed by the midpoint between the top-righ and bottom-right
        startLinePoint = midpoint(tl, bl)
        startLinePoint = (int(startLinePoint[0]), int(startLinePoint[1]))
        endLinePoint = midpoint(tr, br)
        endLinePoint = (int(endLinePoint[0]), int(endLinePoint[1]))
        # compute the Euclidean distance between the midpoints

        dB = abs(startLinePoint[0] - endLinePoint[0])
  
    # draw the midpoints on the image
    cv2.circle(base_img, (int(startLinePoint[0]), int(startLinePoint[1])), 2, (255, 0, 0), -1)
    cv2.circle(base_img, (int(endLinePoint[0]), int(endLinePoint[1])), 2, (255, 0, 0), -1)

    # draw lines between the midpoints


    cv2.line(base_img, startLinePoint, endLinePoint,
        (255, 0, 255), 4)

    if False:
        firstHatchStart = (int(startLinePoint[0]-50), int(startLinePoint[1]))
        firstHatchEnd = (int(startLinePoint[0]+50), int(startLinePoint[1]))
        secondHatchStart = (int(endLinePoint[0]-50), int(endLinePoint[1]))
        secondHatchEnd = (int(endLinePoint[0]+50), int(endLinePoint[1]))
    else:
        firstHatchStart = (int(startLinePoint[0]), int(startLinePoint[1]-50))
        firstHatchEnd = (int(startLinePoint[0]), int(startLinePoint[1]+50))
        secondHatchStart = (int(endLinePoint[0]), int(endLinePoint[1]-50))
        secondHatchEnd = (int(endLinePoint[0]), int(endLinePoint[1]+50))

    cv2.line(base_img, firstHatchStart, firstHatchEnd,
        (255, 0, 255), 5)

    cv2.line(base_img, secondHatchStart, secondHatchEnd,
        (255, 0, 255), 5)


    # if the pixels per metric has not been initialized, then
    # compute it as the ratio of pixels to supplied metric
    # (in this case, inches)
    if pixelsPerMetric is None:
        pixelsPerMetric = get_width_from_ruler(dB, rulerWidth)
        


    dimB = dB / pixelsPerMetric
    if draw_text:
        if pre == "Ruler":
                # draw the object sizes on the image
            cv2.putText(base_img, "{}: {}in".format("U.S. Quarter",dimB),
                (endLinePoint[0]+10, endLinePoint[1]), cv2.FONT_HERSHEY_TRIPLEX,
                1, (255, 255, 255), 1,lineType=cv2.LINE_AA)
        else:
            # draw the object sizes on the image
            cv2.putText(base_img, "{}".format(pre),
                (endLinePoint[0]+10, endLinePoint[1]), cv2.FONT_HERSHEY_TRIPLEX,
                1, (255, 255, 255), 1, lineType=cv2.LINE_AA)

            cv2.putText(base_img, "{:.1f}in".format(dimB),
                (endLinePoint[0]+10, endLinePoint[1]+50), cv2.FONT_HERSHEY_TRIPLEX,
                1, (255, 255, 255), 1, lineType=cv2.LINE_AA)

    return pixelsPerMetric, dimB, startLinePoint, endLinePoint




#get the set of largest contours, then find the one that has the best shape match
#note: have to cycle through the whole set because there can be contours of the same size that are diff
def get_bw_abalone_contour(input_image, template_contour, thresh_val, blur, use_gray):
    #segmentation and edge finding
    orig_image, gray, thresh1, mid_row = get_bw_image(input_image, thresh_val, blur, use_gray)
    orig = orig_image.copy()
    
    edged = utils.find_edges(img=gray, thresh_img=thresh1, use_gray=use_gray, showImg=False, erode_iterations=1)
    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    cnts = cnts[1] #if imutils.is_cv2() else cnts[1]


    #imageName, input_image, contour, template_contour

    #get all the largest edges
    contours, size = utils.get_largest_edge(cnts)
    smallest_combined = 10000000.0
    target_contour = None
    rval = 1000000
    aperc = 1000000
    adist = 1000000
    #cdiff = 1000000



    for contour in contours:
        the_contour, result_val, area_perc, area_dist, centroid_diff = matching.sort_by_matching_shape(contour, 
            template_contour, False, input_image, False, False)
        if the_contour is None:
            continue

        comb = result_val*area_dist*centroid_diff
        if comb < smallest_combined:
            smallest_combined = comb
            rval = result_val
            aperc = area_perc
            adist = area_dist
            target_contour = the_contour
            #cdiff = centroid_diff

    if False:
        cv2.drawContours(input_image, contours, -1, (0,255,0), 1)
        #if target_contour is not None:
        #    cv2.fillPoly(input_image, [target_contour], (255,0,0))

        cv2.imshow("-->>>  bw abalone image {}x{}".format(thresh_val, blur), input_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return target_contour


def is_roundish(contour, thresh_val, blur):
    x,y,w,h = cv2.boundingRect(contour)
    
    w_v_h = float(w)/float(h)

    lim = 0.70
    is_round = (w_v_h >= lim and w_v_h <= (1.0/lim))
    return is_round

def get_bw_ruler_contour(input_image, template_contour, enclosing_contour, thresh_val, blur, showImg, top_offset, use_quarter, use_gray, use_hull):
    #if use_quarter:
    ruler_image, ruler_gray, ruler_thresh, yoffset = get_bw_quarter_image(input_image, thresh_val, blur, use_gray, False)
    ruler_edged = utils.find_edges(img=ruler_gray, thresh_img=ruler_thresh, use_gray=use_gray, showImg=showImg, erode_iterations=1)
    
    #else:
    #    ruler_image, ruler_gray, ruler_thresh, yoffset = get_ruler_image(input_image, thresh_val, blur, showImg)
    #    ruler_edged = utils.find_edges(ruler_gray, ruler_thresh, True, showImg, 1)
    
    
    #find the contours in the ruler half image
    #use chain approx none to return all the points so enclosed contour works
    ruler_contours = cv2.findContours(ruler_edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    ruler_contours = ruler_contours[1] #if imutils.is_cv2() else ruler_contours[1]
    
    smallest_combined = 10000000.0
    target_contour = None

    ok_contours = []

    for contour in ruler_contours:
        the_contour, result_val, area_perc, area_dist,centroid_diff = matching.sort_by_matching_shape(contour, 
            template_contour, showImg,input_image, use_quarter, False)
        if the_contour is None:
            continue

        comb = result_val*area_dist*centroid_diff

        if comb < smallest_combined and area_perc > 0.25 and area_perc < 2.0 and is_roundish(the_contour,thresh_val,blur):
            if enclosing_contour is not None:
                #if its the ruler (quarter), check to see if its enclosed
                is_enclosed = utils.is_contour_enclosed(the_contour, enclosing_contour, use_hull)
                if is_enclosed:
                    continue
            smallest_combined = comb
            target_contour = the_contour
            ok_contours.append(the_contour)

    if False:
        cv2.drawContours(input_image, ruler_contours, -1, (0,255,0), 1)
        if target_contour is not None:
            cv2.fillPoly(input_image, [target_contour], (255,0,0))

        cv2.imshow("-->>>  bw ruler image {}x{}".format(thresh_val, blur), input_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return target_contour


def add_shape_with_color(shapes, input_image, template_contour, thresh_val, blur, key, showImg=False,
    contour_color=None, is_ruler=False, use_gray_threshold=False, enclosing_contour=None, 
    first_pass=True,is_small=False, use_adaptive=False):
    
    #find the matching shape using the color images with blue background
    #showImg=False, contour_color=None, is_ruler=False, use_gray_threshold=False, enclosing_contour=None
    col_contour, val, area_perc, dist, centroid_diff = ci.do_color_image_match(input_image, template_contour, thresh_val, blur, 
        showImg, contour_color, is_ruler, use_gray_threshold, enclosing_contour, first_pass=first_pass, use_adaptive=use_adaptive,small_img=is_small)

    if val is not None and dist is not None:
        shapes.append((val, area_perc, dist, col_contour, key,val*dist*centroid_diff,centroid_diff))
    return shapes

def add_shape_by_match(shapes, input_image, target_contour, template_contour, thresh_val, blur, key,
    showImg, top_offset, is_quarter, first_pass):
    ii = input_image.copy()
    if target_contour is None:
        return shapes

    #find the matching ruler shape with the alt ruler 
    s_contour, val, area_perc, dist,centroid_diff = matching.sort_by_matching_shape(target_contour, 
        template_contour, False, input_image, is_quarter, first_pass)
    if showImg:
        #cv2.fillPoly(ii, [template_contour], (0,0,255), offset=(0,top_offset))
        utils.show_img("color shape {}x{}".format(thresh_val, blur), ii)
    if s_contour is not None:
        shapes.append((val, area_perc, dist, s_contour, key, val*dist*centroid_diff,centroid_diff))
    return shapes

def match_template(input_image, template, showImg):
    template_image = cv2.imread(template)

    h,w = template_image.shape[:2]
    h=h*2
    w=w*2
    res = cv2.matchTemplate(input_image,template_image,cv2.TM_SQDIFF)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    top_left = max_loc
    x = top_left[0]
    y = top_left[1]
    bottom_right = (x+w, y+h)
    if showImg:
        print "max val is {}; max loc x: {}, y:{}, bottom:{}".format(max_val, x, y, bottom_right)
    #cv2.rectangle(input_image,top_left, bottom_right, 255, 2)
    y,x = np.unravel_index(res.argmax(), res.shape)
    if showImg:
        print "x: {}, y:{}".format(x, y)
    mask = np.zeros(input_image.shape,np.uint8)
    mask[x:x+w, y:y+h] = input_image[x:x+w, y:y+h]
    if showImg:
        utils.show_img("template ", mask)
    return mask, x,y

def trim_abalone_contour(abalone_contour):
    cX, cY = utils.get_centroid(abalone_contour)
    ab_ellipse = cv2.fitEllipse(abalone_contour)
    pts = cv2.boxPoints(ab_ellipse)
    size = ab_ellipse[1]
    width = int(size[1]/2)
    height = int(size[0])
    w = int(size[0])
    h = int(size[1])
    center = np.array([cX, cY])
    radius = width-2
    acon = np.squeeze(abalone_contour)

    xmin = center[0]-((int(w/2)+50))
    xmax = center[0]+((int(w/2))+50)
    for pt in acon:
        if pt[0] < xmin:
            pt[0] = xmin
        elif pt[0] > xmax:
            pt[0] = xmax


    return acon, ab_ellipse

def get_quarter_contour_and_center(quarter_contour):
    cX, cY = utils.get_centroid(quarter_contour)
    quarter_ellipse = cv2.fitEllipse(quarter_contour)
    pts = cv2.boxPoints(quarter_ellipse)
    size = quarter_ellipse[1]
    width = int(size[1]/2)
    height = int(size[0])
    w = int(size[0])
    h = int(size[1])
    center = np.array([cX, cY])
    radius = width-2
    qcon = np.squeeze(quarter_contour)

    xmin = center[0]-((int(w/2)+2))
    xmax = center[0]+((int(w/2))+2)
    for pt in qcon:
        if pt[0] < xmin:
            pt[0] = xmin
        elif pt[0] > xmax:
            pt[0] = xmax

    #mask = (qcon[:,0] - center[0])**2 + (qcon[:,1] - center[1])**2 < (radius**2)+2
    
    #contourWithinQuarterCircle = qcon[mask,:]
    return cX, cY, qcon, quarter_ellipse

def get_color_abalone(thresholds, blurs, abalone_shapes, abalone_template, rescaled_image,first_pass,is_small, 
    use_gray_threshold=False, use_adaptive=False, description='x'):
    for thresh_val in thresholds:
        for blur in blurs:
            key = "{}th_{}bl_{}".format(thresh_val, blur,description)
            show_img = False
            abalone_shapes = add_shape_with_color(abalone_shapes, rescaled_image.copy(), 
                abalone_template.copy(), thresh_val, blur, (key+"_color_ab"),
                showImg=show_img, contour_color=(0,0,255), is_ruler=False, use_gray_threshold=use_gray_threshold, 
                enclosing_contour=None,first_pass=first_pass,is_small=is_small,use_adaptive=use_adaptive)
    return abalone_shapes

def get_color_ruler(thresholds, blurs, ruler_shapes, ruler_template, ruler_image, abalone_contour, 
    use_gray_threshold, first_pass=True, use_adaptive=False, description='x',is_small=False):
    otherinfo = ""
    if first_pass:
        otherinfo = "firstpass"
    else:
        otherinfo = "secondpass"

    if use_adaptive:
        otherinfo = "{}_{}".format(otherinfo, "adaptive")
    else:
        otherinfo = "{}_{}".format(otherinfo, "thresh")

    otherinfo = "{}_{}".format(otherinfo, description)
    #do the quarter selection
    for thresh_val in thresholds:
        for blur in blurs:
            if use_gray_threshold:
                key = "{}th_{}bl_{}".format(thresh_val, blur, "gray_quarter_{}".format(otherinfo))
            else:
                key = "{}th_{}bl_{}".format(thresh_val, blur, "color_quarter_{}".format(otherinfo))

            si = not use_gray_threshold
            ruler_shapes = add_shape_with_color(ruler_shapes,ruler_image.copy(), 
                ruler_template.copy(), thresh_val, blur, key,
                showImg=False, contour_color=(255,0,0), is_ruler=True, use_gray_threshold=use_gray_threshold, 
                enclosing_contour=abalone_contour, first_pass=first_pass, use_adaptive = use_adaptive,is_small=is_small)

    return ruler_shapes

def get_bw_abalone(thresholds, blurs, abalone_shapes, abalone_template, rescaled_image, use_gray,description='x'):
    for thresh_val in thresholds:
        for blur in blurs:
            key = "{}th_{}bl".format(thresh_val, blur,description)
            #read the image for the abalone contour with these settings
            tgt_contour = get_bw_abalone_contour(rescaled_image.copy(), abalone_template, 
                thresh_val, blur, use_gray)
            abalone_shapes = add_shape_by_match(abalone_shapes, rescaled_image.copy(), tgt_contour, 
                abalone_template.copy(),thresh_val, blur, (key+"_bw_ab"),False, 0, False, True)
    return abalone_shapes

def get_bw_ruler(thresholds, blurs, ruler_shapes, ruler_template, enclosing_contour, ruler_image, use_gray=False,description='x',use_hull=True):
    for thresh_val in thresholds:
        for blur in blurs:
            key = "{}th_{}bl".format(thresh_val, blur)
            if use_gray:
                key = key+"_gray_quarter"
            else:
                key = key+"_quarter"

            key = "{}_{}".format(key, description)
            tgt_contour = get_bw_ruler_contour(ruler_image.copy(), ruler_template.copy(), enclosing_contour,
                thresh_val,blur,False,0,True, use_gray,use_hull)
            bw_ruler_shapes = add_shape_by_match(ruler_shapes, ruler_image.copy(),
                tgt_contour, ruler_template.copy(), thresh_val, blur, key, False, 0, True, True)
                
    return ruler_shapes

def noResults(key, val):
    if key is None or len(key) == 0 or val >= 1000000:
        return True
    else:
        return False

def print_time(msg):
    now = time.time()
    elapsed = now - _start_time
    print "{} time elapsed: {}".format(msg, elapsed)
    #logger.debug(msg)


def do_dynamo_put(name, email, uuid, locCode, picDate, len_in_inches, rating, notes):
    dynamodb = boto3.resource('dynamodb')
    table = dynamodb.Table('ab_length')
    try:
        lenfloat = round(float(len_in_inches),2)
    except StandardError, e:
        lenfloat = -1.0

    try:
        table.put_item(
            Item={
                'username': name,
                'email': email,
                'uuid': uuid,
                'locCode': locCode,
                'picDate': decimal.Decimal(picDate),
                'length_in_inches':decimal.Decimal('{}'.format(lenfloat)),
                'rating':decimal.Decimal('{}'.format(rating)),
                'usernotes': notes
            }
        )
    except ClientError as e:
        print(e.response['Error']['Message'])
    else:
        print("{} length updated to {}".format(uuid, lenfloat))

def do_s3_upload(final_thumb, uuid):
    s3 = boto3.resource('s3')

    #s3.Bucket('abalone').put_object(Key="full_size/"+uuid+".png", Body=image_data)
    #print_time("done putting full size")

    #s3.Bucket('abalone').put_object(Key="thumbs/"+uuid+".png", Body=thumb)
    #print_time("done with thumb")
    s3.Bucket('abalone').put_object(Key="thumbs/"+uuid+".png", Body=final_thumb)
    print_time("don with final")

def get_thumbnail(image_full):
    target_cols = 200.0

    orig_cols = len(image_full[0]) 
    orig_rows = len(image_full)
 
    target_rows = (float(orig_rows)/(float(orig_cols))*200.0)
    fx = float(target_cols/orig_cols)
    fy = float(target_rows/orig_rows)

    thumb = cv2.resize( image_full, (0,0), fx = fx, fy = fy)
    return thumb


def find_abalone_length(is_deployed, req):
    _start_time = time.time()
    print_time( "start")
    #width of US quarter in inches
    quarter_width = 0.955

    #all the work
    thresholds = [10, 30, 70]
    blurs = [1,3,5,7]


    bestRulerContour = None
    bestAbaloneContour = None

    if is_deployed:
        #user info
        name = req[u'username']
        email = req[u'email']
        uuid = req[u'uuid']
        locCode = req[u'locCode']
        picDate = req[u'picDate']
        rating = '-1'
        notes = 'none'
        #img info
        img_str = req[u'base64Image']
        img_data = base64.b64decode(img_str)
        tmp_filename = '/tmp/ab_length_{}.png'.format(time.time()) 

        with open(tmp_filename, 'wb') as f:
            f.write(img_data)
        

        imageName = tmp_filename
        image_full = cv2.imread(imageName)
        thumb = get_thumbnail(image_full)

        showResults = False
        rulerWidth = quarter_width
        out_file = None

        
    else:
        (imageName, showResults, rulerWidth, out_file) = read_args()
        image_full = cv2.imread(imageName)
        thumb = get_thumbnail(image_full)
        uuid = "delete_me"
        img_data = cv2.imencode('.png', image_full)[1].tostring()
        thumb_str = cv2.imencode('.png', thumb)[1].tostring()
        rating = '-1'
        notes = 'none'
        name = 'DUploadTest'
        email = 'foo@bar.c'
        uuid = 'a412c020-3254-430a-a108-243113f9fde5'
        locCode = "S88 Bodega Head"
        picDate = int(time.time()*1000);


    #read the image
    orig_cols = len(image_full[0]) 
    orig_rows = len(image_full)

    #if its vertical, flip it 90
    if orig_cols < orig_rows:
        img = cv2.transpose(image_full)  
        img = cv2.flip(img, 0)
        image_full = img.copy()
        orig_cols = len(image_full[0])
        orig_rows = len(image_full)

    rescaled_image, scaled_rows, scaled_cols = get_scaled_image(image_full)
    abalone_template_contour, small_abalone_template_contour, quarter_template_contour = get_template_contours(rescaled_image)
    print_time("template loads...")
    minEdged = None
            
    abalone_shapes = []
    large_color_abalone_shapes = []
    small_color_abalone_shapes = []
    bw_abalone_shapes = []
    ruler_shapes = []
    bw_ruler_shapes = []
    gray_ruler_shapes = []
    pixelsPerMetric = None
    
    if orig_cols < 1250:
        is_small = True
    else:
        is_small = False

    
    #ruler_image = rescaled_image[int(rows/2):rows, 0:cols].copy()
    ruler_image = rescaled_image.copy()

    is_color_bkground = utils.is_color(rescaled_image)
    background_val_diff = utils.is_background_similar_color(rescaled_image)
    

    #show_img("half ruler", ruler_image.copy())
    #ruler_mask, ruler_top_offset_x, ruler_top_offset_y = match_template(ruler_image.copy(), "../images/ruler_image_2x.png",False)
    #alt_ruler_mask, alt_ruler_top_offset_x,alt_ruler_top_offset_y = match_template(ruler_image.copy(), "../images/alt_ruler_image2_2x.png",False)

    #qmask, qoffset_x, qoffset_y = match_template(ruler_image.copy(), "../images/quarter_image_template_2x.png", False)
    newBestAbaloneContour = None
    bestAbaloneKey = None
    bestAbaloneValue = 0
    
    print "abalone, color first"
    diff_h, diff_s, diff_v = utils.get_mean_abalone_color(rescaled_image)
    low_contrast = diff_v < 60 

    large_color_abalone_shapes = get_color_abalone(thresholds, blurs, 
        large_color_abalone_shapes, abalone_template_contour, rescaled_image, 
        first_pass=True, is_small=is_small, use_gray_threshold=False, description="strict_large")
    print_time("done with color")
    if False:
        print "not a color background"
        bw_abalone_shapes = get_bw_abalone(thresholds, blurs, abalone_shapes, abalone_template_contour, 
            rescaled_image, True, description="strict")
    print_time("done with bw")
    newBestAbaloneContour, bestAbaloneKey, bestAbaloneValue = utils.get_best_contour(large_color_abalone_shapes+bw_abalone_shapes, 
        0.45, 1.5, ABALONE, None, False, scaled_rows, scaled_cols)
    
    if noResults(bestAbaloneKey, bestAbaloneValue):
        if True:
            print "doing bw for color abalone...."
            bw_abalone_shapes = get_bw_abalone(thresholds, blurs, abalone_shapes, abalone_template_contour, 
                rescaled_image, True, description="strict")
            #if there is still nothing, loosen the area restrictions and try again
            newBestAbaloneContour, bestAbaloneKey, bestAbaloneValue = utils.get_best_contour(bw_abalone_shapes, 0.45, 1.75, ABALONE, None, False, scaled_rows, scaled_cols)
        
        if noResults(bestAbaloneKey, bestAbaloneValue) and is_small:
            #try gray small
            small_color_abalone_shapes = get_color_abalone(thresholds, blurs, 
                small_color_abalone_shapes, small_abalone_template_contour, rescaled_image,first_pass=False,
                is_small=True, use_gray_threshold=False, description="color_small_loose")
            print_time("done with small color")
            newBestAbaloneContour, bestAbaloneKey, bestAbaloneValue = utils.get_best_contour(small_color_abalone_shapes, 
                0.4, 1.55, ABALONE, None, False, scaled_rows, scaled_cols)
            small_color_abalone_shapes = []
            if noResults(bestAbaloneKey, bestAbaloneValue) and is_small:
                #try gray small
                small_color_abalone_shapes = get_color_abalone(thresholds, blurs, 
                    small_color_abalone_shapes, small_abalone_template_contour, rescaled_image,first_pass=False,
                    is_small=is_small, use_gray_threshold=True, description="gray_small_loose")
                print_time("done with small gray color")
                newBestAbaloneContour, bestAbaloneKey, bestAbaloneValue = utils.get_best_contour(small_color_abalone_shapes, 
                    0.10, 1.25, ABALONE, None, False, scaled_rows, scaled_cols)
        
    if low_contrast:
        bw_ruler_shapes = get_bw_ruler(thresholds, blurs, 
                    bw_ruler_shapes, quarter_template_contour, newBestAbaloneContour, ruler_image, False, description="strict",use_hull=True)
        print_time("did bw quarter...")
    ruler_shapes = get_color_ruler(thresholds, blurs, ruler_shapes, quarter_template_contour, ruler_image, 
        newBestAbaloneContour, False, first_pass=True, use_adaptive=True, description="strict",is_small=False)
    
    print_time("getting color strict")
    newBestRulerContour, bestRulerKey, bestRulerValue = utils.get_best_contour(ruler_shapes+bw_ruler_shapes, 0.60, 1.9, RULER, newBestAbaloneContour, False, scaled_rows, scaled_cols, rescaled_image.copy())
    
    if noResults(bestRulerKey, bestRulerValue):
        print "working on color quarter without adaptive"
        ruler_shapes = get_color_ruler(thresholds, blurs, ruler_shapes, quarter_template_contour, 
            ruler_image, newBestAbaloneContour, False, first_pass=True, use_adaptive=False,
            description="adaptive",is_small=False)
        newBestRulerContour, bestRulerKey, bestRulerValue = utils.get_best_contour(ruler_shapes, 0.4, 1.9, RULER, newBestAbaloneContour, False, scaled_rows, scaled_cols, rescaled_image.copy())

        if noResults(bestRulerKey, bestRulerValue):
            if not low_contrast:
                print "working on b&w quarter..."
                bw_ruler_shapes = get_bw_ruler(thresholds, blurs, 
                    bw_ruler_shapes, quarter_template_contour, newBestAbaloneContour, ruler_image, True, description="strict", use_hull=True)
                newBestRulerContour, bestRulerKey, bestRulerValue =  utils.get_best_contour(ruler_shapes+bw_ruler_shapes, 0.4, 1.9, RULER, newBestAbaloneContour, False, scaled_rows, scaled_cols, rescaled_image.copy())
            
            if noResults(bestRulerKey, bestRulerValue):
                print "no good, trying gray?"
                gray_ruler_shapes = get_color_ruler(thresholds, blurs, gray_ruler_shapes, 
                    quarter_template_contour, ruler_image, newBestAbaloneContour, True,
                    first_pass=True,use_adaptive=False, description="gray",is_small=False)
                newBestRulerContour, bestRulerKey, bestRulerValue =  utils.get_best_contour(gray_ruler_shapes, 0.5, 1.5, RULER, newBestAbaloneContour, False, scaled_rows, scaled_cols, rescaled_image.copy())

                if noResults(bestRulerKey, bestRulerValue):
                    #trying getting results with a lower boundary
                    print "trying color with looser rules"
                    ruler_shapes = []
                    ruler_shapes = get_color_ruler(thresholds, blurs, ruler_shapes, quarter_template_contour, 
                        ruler_image, newBestAbaloneContour, False, first_pass=False,use_adaptive=False,
                        description="loose",is_small=True)
                    newBestRulerContour, bestRulerKey, bestRulerValue =  utils.get_best_contour(ruler_shapes, 0.40, 3.0, RULER, newBestAbaloneContour, False, scaled_rows, scaled_cols, rescaled_image.copy())
                
                    if noResults(bestRulerKey, bestRulerValue):
                        print "failed - trying b&w with looser rules"
                        bw_ruler_shapes = get_bw_ruler(thresholds, blurs, bw_ruler_shapes, 
                            quarter_template_contour, newBestAbaloneContour, ruler_image, description="loose", use_hull=False)
                        newBestRulerContour, bestRulerKey, bestRulerValue =  utils.get_best_contour(bw_ruler_shapes, 0.25, 1.9, RULER, newBestAbaloneContour, False, scaled_rows, scaled_cols, rescaled_image.copy())

                        if noResults(bestRulerKey, bestRulerValue):
                            print "failed - trying b&w with no enclosing contour"
                            bw_ruler_shapes = get_bw_ruler(thresholds, blurs, bw_ruler_shapes, 
                                quarter_template_contour, None, ruler_image, description="loose", use_hull=False)
                            newBestRulerContour, bestRulerKey, bestRulerValue =  utils.get_best_contour(bw_ruler_shapes, 0.25, 1.9, RULER, newBestAbaloneContour, False, scaled_rows, scaled_cols, rescaled_image.copy())

    #add this for abalone, too, to prevent empty results
    if noResults(bestRulerKey, bestRulerValue):
        print "trying quarter with loose guidelines..."
        all_rulers = bw_ruler_shapes+ruler_shapes+gray_ruler_shapes
        newBestRulerContour,bestRulerKey, bestRulerValue = utils.get_best_contour((bw_ruler_shapes+ruler_shapes+gray_ruler_shapes), 0.2, 3.0, RULER, newBestAbaloneContour, False, scaled_rows, scaled_cols, None, True)

    if noResults(bestAbaloneKey, bestAbaloneValue):
        print "falling back on abalone with very loose guidelines..."

        newBestAbaloneContour, bestAbaloneKey, bestAbaloneValue = utils.get_best_contour(abalone_shapes+large_color_abalone_shapes+small_color_abalone_shapes+bw_abalone_shapes, 
            0.10, 1.25, ABALONE, None, False, scaled_rows, scaled_cols)
    
        if noResults(bestAbaloneKey, bestAbaloneValue):
            print "ok, trying one more time with width and height limits turned off"
            newBestAbaloneContour, bestAbaloneKey, bestAbaloneValue = utils.get_best_contour(abalone_shapes+large_color_abalone_shapes+small_color_abalone_shapes, 
                0.10, 1.25, ABALONE, None, False, scaled_rows, scaled_cols, None, True)

    is_quarter = True
    print_time( "done with all lengths")
    #fit a centroid to the quarter and trim outlying scratches/lines
    centroidX, centroidY, qell, quarter_ellipse = get_quarter_contour_and_center(newBestRulerContour)
    origRulerContour = newBestRulerContour.copy()

    if qell is not None and len(qell) > 0:
        newBestRulerContour = qell.copy()

    print_time("clipped quarter")
    #add a loose ellipse to the abalone and trim the width. trying to get rid of large horizontal lines/outliers
    #trimmed_ab_contour, ab_ell = trim_abalone_contour(newBestAbaloneContour.copy())
    #if trimmed_ab_contour is None or len(trimmed_ab_contour) == 0:
    #    trimmed_ab_contour = newBestAbaloneContour.copy()

    showText = showResults and not is_deployed
    pixelsPerMetric, rulerLength,left_ruler_point, right_ruler_point = draw_contour(rescaled_image, 
        newBestRulerContour, None, "Ruler", 0, rulerWidth,is_quarter, showText)
    pixelsPerMetric, abaloneLength, left_point, right_point = draw_contour(rescaled_image, 
        newBestAbaloneContour, pixelsPerMetric, "Abalone", 0, rulerWidth, False,showText)
    print_time("done drawing")

    all_rows = {}
    if is_mac():
        file_utils.read_write_csv(out_file, imageName, bestAbaloneKey, bestRulerKey, abaloneLength, rulerLength, bestRulerValue, background_val_diff)
    
    print "final best abalone key is -->>>{}<<<-----, value of {}".format(bestAbaloneKey, bestAbaloneValue)
    print "final best ruler key is -->>>{}<<<-----, value of {}".format(bestRulerKey, bestRulerValue)

    if bestRulerKey.endswith("_masked_quarter"):
        offx = qoffset_x
        offy = qoffset_y
    else:
        offx = 0
        offy = 0

    #drawing these for now...just not showing in web app
    if True:
        cv2.drawContours(rescaled_image, [origRulerContour], 0, (50,50,50),3,lineType=cv2.LINE_AA)
        if newBestAbaloneContour is not None:
            cv2.drawContours(rescaled_image, [newBestAbaloneContour], 0, (50,255,150), 3,lineType=cv2.LINE_AA)
        cv2.drawContours(rescaled_image, [newBestRulerContour], 0, (0,255,0), 3,lineType=cv2.LINE_AA)
        
    bounded_image = cv2.copyMakeBorder(rescaled_image,10,10,10,10,cv2.BORDER_CONSTANT,value=(0,0,0))
    if showResults and not is_deployed:
        cv2.namedWindow(imageName, cv2.WINDOW_NORMAL)
        cv2.imshow(imageName, bounded_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if is_deployed:
        upload_worker(rescaled_image, thumb, img_data, name, email, uuid, locCode, picDate, abaloneLength, rating, notes)

    else:
        final_tmp_filename = 'ab_final_tmp.png'
        cv2.imwrite(final_tmp_filename,rescaled_image) 

    rows = len(rescaled_image)
    cols = len(rescaled_image[0])
    rval =  {
                "left_x":str(left_point[0]), "right_x":str(right_point[0]), "y":str(left_point[1]),
                "length":str(abaloneLength),
                "width":str(cols),"height":str(rows),
                "quarter_left_x":str(left_ruler_point[0]),
                "quarter_right_x":str(right_ruler_point[0]),
                "quarter_y":str(left_ruler_point[1]),
                "uuid":str(uuid)
            }
    jsonVal = json.dumps(rval)
    print "the json val::::"
    print jsonVal
    print "-----"
    return jsonVal

def upload_worker(rescaled_image, thumb, img_data, 
    name, email, uuid, locCode, picDate, abaloneLength, rating, notes):
    #print_time("uploading data now....")
    #final_image = cv2.imencode('.png', rescaled_image)[1].tostring()
    #print_time("done encoding image")
    do_dynamo_put(name, email, uuid, locCode, picDate, abaloneLength, rating, notes)
    #print_time("done putting things into dynamo db")

    original_thumb_str = cv2.imencode('.png', thumb)[1].tostring()
    #print_time("done encoding thumb")
    final_thumb = get_thumbnail(rescaled_image)
    thumb_str = cv2.imencode('.png', final_thumb)[1].tostring()
    do_s3_upload(thumb_str, uuid)
    #do_s3_upload(None, thumb_str, None, uuid)
    print_time("done uploading data...")

def lambda_handler(event, context):
    try:
        ab_length = find_abalone_length(True, event)
    except StandardError, e:
        ab_length = "Unknown"
        
    return ab_length

def is_mac():
    os_name = sys.platform
    return os_name == "darwin"

def run_program():
    os_name = sys.platform
    if os_name == "darwin":
        res = find_abalone_length(False, None)
    else:
        res = find_abalone_length(False, None)


if __name__ == "__main__":
    run_program()


    