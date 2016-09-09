# import the necessary packages
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2
import csv
import os

def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)


def sort_by_matching_shape(target_contour, template_shape):
    templateHull = cv2.convexHull(template_shape)
    templateArea = cv2.contourArea(templateHull)
    targetHull = cv2.convexHull(target_contour)
    targetArea = cv2.contourArea(targetHull)

    val = cv2.matchShapes(target_contour,template_shape,1,0.0)
    
    return target_contour, val, (targetArea/templateArea)

        

def get_template_contours():
    print "getting template contours...."
    template = cv2.imread("../template.jpg")

    template_edged = cv2.Canny(template, 15, 100)
    #edged_img = cv2.Canny(threshhold, 50, 100)
    edged_img = cv2.dilate(template_edged, None, iterations=1)

    #edged_img = cv2.erode(edged_img, None, iterations=1)
    #cv2.imshow("edged", edged_img)
    #cv2.waitKey(0)
    im2, template_shapes, hierarchy = cv2.findContours(edged_img, 2,1)
 
    return template_shapes[0], template_shapes[2]

def get_width_from_ruler(dB, rulerWidth):
    return dB/float(rulerWidth)

def get_ruler_image(imageName, thresh_val, blur_window):
    # load the image, convert it to grayscale, and blur it slightly
    image_full = cv2.imread(imageName)
    if(len(image_full) > 500):
        rescaled_image = cv2.resize( image_full, (0,0), fx = 0.25, fy = 0.25)
    else:
        rescaled_image = image_full

 
    rows = len(rescaled_image)
    cols = len(rescaled_image[0])      
    image = rescaled_image[int(rows/2):rows, 0:cols].copy()
        
    rows = len(image)
    cols = len(image[0])  

    hist = cv2.calcHist([image],[0],None,[10],[0,256])
    #maybe use this to see if we should threshold?
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    gray = cv2.GaussianBlur(gray, (blur_window, blur_window), 0)    

    #assumes the abalone is centered
    mid_row_start = int(rows/3)*2 - thresh_val
    mid_col_start = int(cols/2) - thresh_val

    mid_row_end = mid_row_start+thresh_val
    mid_col_end = mid_col_start+thresh_val

    mid_patch = gray[mid_row_start:mid_row_end, mid_col_start:mid_col_end]
    mn = np.mean(mid_patch) 

    retval, threshold = cv2.threshold(gray,mn,255,cv2.THRESH_BINARY)
    return image, gray, threshold

def get_color_image(image, lower_color, upper_color):
# Threshold the HSV image to get only blue colors
    mask = cv2.inRange(image, lower_color, upper_color)
    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(image,image, mask= mask)
    return res

def get_image_with_color_mask(imageName, thresh_val, blur_window):
    image_full = cv2.imread(imageName)
    if(len(image_full) > 500):
        rescaled_image = cv2.resize( image_full, (0,0), fx = 0.25, fy = 0.25)
    else:
        rescaled_image = image_full

    rows = len(rescaled_image)
    cols = len(rescaled_image[0])    

    image = rescaled_image


    non_blue_lower = np.array([60,90,90])
    non_blue_upper = np.array([240,255,255])
    res = get_color_image(image, non_blue_lower, non_blue_upper)


    #maybe use this to see if we should threshold?
    gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)

    gray = cv2.GaussianBlur(gray, (blur_window, blur_window), 0)
 
    #assumes the abalone is centered
    mid_row_start = int(rows/2) - thresh_val
    mid_col_start = int(cols/2) - thresh_val

    mid_row_end = mid_row_start+thresh_val
    mid_col_end = mid_col_start+thresh_val


    mid_patch = gray[mid_row_start:mid_row_end, mid_col_start:mid_col_end]
    mn = np.mean(mid_patch) 

    retval, threshold = cv2.threshold(gray,mn,255,cv2.THRESH_BINARY)

    return image, gray, threshold, int(rows/2)

def get_image(imageName, thresh_val, blur_window):
    # load the image, convert it to grayscale, and blur it slightly
    image_full = cv2.imread(imageName)
    if(len(image_full) > 500):
        rescaled_image = cv2.resize( image_full, (0,0), fx = 0.25, fy = 0.25)
    else:
        rescaled_image = image_full

    rows = len(rescaled_image)
    cols = len(rescaled_image[0])    

    image = rescaled_image

    #maybe use this to see if we should threshold?
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    gray = cv2.GaussianBlur(gray, (blur_window, blur_window), 0)

    #assumes the abalone is centered
    mid_row_start = int(rows/2) - thresh_val
    mid_col_start = int(cols/2) - thresh_val

    mid_row_end = mid_row_start+thresh_val
    mid_col_end = mid_col_start+thresh_val


    mid_patch = gray[mid_row_start:mid_row_end, mid_col_start:mid_col_end]
    mn = np.mean(mid_patch) 

    retval, threshold = cv2.threshold(gray,mn,255,cv2.THRESH_BINARY)

    return image, gray, threshold, int(rows/2)

def find_edges(gray, threshhold):
    # perform edge detection, then perform a dilation + erosion to
    # close gaps in between object edges
    edged_img = cv2.Canny(threshhold, 50, 100)
    edged_img = cv2.dilate(edged_img, None, iterations=1)
    edged_img = cv2.erode(edged_img, None, iterations=1)
    return edged_img

def read_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('allimages', metavar='fp', nargs='+', help='file names')
    ap.add_argument("-i", "--image", required=False,
        help="path to the input image")

    ap.add_argument("-w", "--width", required=False,
        help="width of ruler. defaults to 7.5 inches")

    ap.add_argument("-s", "--show", required=False,
        help="show the results. if not set, the results will write to a csv file")


    args = vars(ap.parse_args())

    rulerWidth = args["width"]
    if not rulerWidth:
        rulerWidth = 7.5

    showResults = args["show"]
    if not showResults:
        showResults = False
    else:
        showResults = bool(showResults)

    imageName = args['allimages'][0]
    if not imageName:
        imageName = args["image"]
    else:
        imageParts = imageName.split()
        if(len(imageParts) > 1):
            imageName = "{} {}".format(imageParts[0], imageParts[1])
        print("imageName: ", imageName)

    
    return imageName, showResults, rulerWidth

def get_largest_edge(cnts):
    max_size = 0
    targetDex = 0
    for i, contour in enumerate(cnts):
        carea = cv2.contourArea(contour)
        if carea > max_size:
            max_size = carea
            targetDex = i
    return cnts[targetDex], max_size


def draw_contour(base_img, con, pixelsPerMetric, pre, top_offset=None):

    brect = cv2.boundingRect(con)

    brect_arr = np.array(brect, dtype="int")

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

    #cv2.drawContours(orig, [corners], -1, (0, 255, 0), 2)

    # loop over the original points and draw them
    #for (x, y) in corners:
    #   cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)

    (tltrX, tltrY) = midpoint(tl, tr)
    (blbrX, blbrY) = midpoint(bl, br)

    # compute the midpoint between the top-left and top-right points,
    # followed by the midpoint between the top-righ and bottom-right
    (tlblX, tlblY) = midpoint(tl, bl)
    (trbrX, trbrY) = midpoint(tr, br)

    # draw the midpoints on the image
    #cv2.circle(base_img, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
    #cv2.circle(base_img, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
    cv2.circle(base_img, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
    cv2.circle(base_img, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)

    # draw lines between the midpoints
    #cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
    #   (255, 0, 255), 2)
    cv2.line(base_img, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
        (255, 0, 255), 1)

    if top_offset == 0:
        left_ruler_top = (int(tlblX), int(tlblY)-100)
        left_ruler_bottom = (int(tlblX), int(tlblY)+150)
        cv2.line(base_img, left_ruler_top, left_ruler_bottom,
            (255, 0, 255), 1)

        right_ruler_top = (int(trbrX), int(trbrY)-100)
        right_ruler_bottom = (int(trbrX), int(trbrY)+100)
        cv2.line(base_img, right_ruler_top, right_ruler_bottom,
            (255, 0, 255), 1)


    # compute the Euclidean distance between the midpoints
    dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
    dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

    # if the pixels per metric has not been initialized, then
    # compute it as the ratio of pixels to supplied metric
    # (in this case, inches)
    if pixelsPerMetric is None:
        pixelsPerMetric = get_width_from_ruler(dB, rulerWidth)

    # compute the size of the object
    dimA = dA / pixelsPerMetric
    dimB = dB / pixelsPerMetric

    # draw the object sizes on the image
    #cv2.putText(orig, "{:.1f}in".format(dimA),
    #       (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
    #   0.65, (255, 255, 255), 2)

    cv2.putText(base_img, "{}: {:.1f}in".format(pre, dimB),
        (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
        0.65, (255, 255, 255), 2)

    return pixelsPerMetric


def do_color_image_match(imageName):
    #try the color image
    color_image, gray, thresh1, mid_row = get_image_with_color_mask(imageName, 30, 5)
    orig = color_image.copy()
    edged = find_edges(gray, thresh1)

    # find contours in the edge map
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]

    abalone_contour, abalone_size = get_largest_edge(cnts)
    abalone, abalone_val, area_perc= sort_by_matching_shape(abalone_contour, abalone_template_contour)
    return abalone, abalone_val, area_perc


#all the work
thresholds = [30, 50]
blurs = [1,3,5,7]

#thresholds = [30]
#blurs = [5]
bestRulerContour = None
bestAbaloneContour = None

minRulerValue = 1000000
minAbaloneValue = 1000000
(imageName, showResults, rulerWidth) = read_args()
template_contours = get_template_contours()
ruler_template_contour, abalone_template_contour = get_template_contours()

minEdged = None
for thresh_val in thresholds:
    for blur in blurs:
        #read full image
        #orig_image, gray, thresh1, mid_row = get_image_with_color_mask(imageName, thresh_val, blur)
        orig_image, gray, thresh1, mid_row = get_image(imageName, thresh_val, blur)
        orig = orig_image.copy()
        edged = find_edges(gray, thresh1)

        # find contours in the edge map
        cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if imutils.is_cv2() else cnts[1]

        abalone_contour, abalone_size = get_largest_edge(cnts)
        
        #(template_contours, _) = contours.sort_contours(template_contours)
        pixelsPerMetric = None
        abalone, abalone_val, area_perc= sort_by_matching_shape(abalone_contour, abalone_template_contour)

        if abalone_val < minAbaloneValue:
            if area_perc > 0.5 and area_perc < 1.5:
                minAbaloneValue = abalone_val
                bestAbaloneContour = abalone.copy()
           
        #read the image for the ruler - assumes its in the lower half
        ruler_image, ruler_gray, ruler_thresh = get_ruler_image(imageName, thresh_val, blur)
        ruler_edged = find_edges(ruler_gray, ruler_thresh)
        #find the contours in the ruler half image
        ruler_contours = cv2.findContours(ruler_edged.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
        ruler_contours = ruler_contours[0] if imutils.is_cv2() else ruler_contours[1]
        ruler_contour, ruler_size = get_largest_edge(ruler_contours)

        ruler, ruler_val, ruler_perc = sort_by_matching_shape(ruler_contour, ruler_template_contour)
        print " ruler value is {}; area fraction is {}; thresh is {}; blur is {}".format(ruler_val, ruler_perc, thresh_val, blur)

        #ruler_process_val = ruler_val
        if ruler_val < minRulerValue:
            if ruler_perc > 0.75 and ruler_perc < 1.25:
                minRulerValue = ruler_val
                bestRulerContour = ruler.copy()
            

color_abalone, color_abalone_val, color_perc = do_color_image_match(imageName)
if color_abalone_val < minAbaloneValue:
    if color_perc > 0.5 and color_perc < 1.5:
        print " new best abalone  value is color:: val:{}; area fraction is {};".format(color_abalone_val, color_perc)
        minAbaloneValue = color_abalone_val
        bestAbaloneContour = color_abalone.copy()
    else:
        print "color was NOT best: {};{}; didn't beat {} ".format(color_abalone_val, color_perc, minAbaloneValue)

print "min ruler val is {}".format(minRulerValue)
pixelsPerMetric = draw_contour(orig, bestRulerContour, None, "Ruler", mid_row)
pixelsPerMetric = draw_contour(orig, bestAbaloneContour, pixelsPerMetric, "Abalone", 0)

cv2.drawContours(orig, [bestRulerContour], 0, (255,255,0), 3)
cv2.drawContours(orig, [bestAbaloneContour], 0, (0,255,255), 3)
cv2.imshow("Size of Abalone", orig)
cv2.waitKey(0)

    
