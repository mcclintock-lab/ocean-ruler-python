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

def show_img(title, img):
    cv2.imshow(title, img)
    cv2.waitKey(0)

def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)


def sort_by_matching_shape(target_contour, template_shape):
    templateHull = cv2.convexHull(template_shape)
    templateArea = cv2.contourArea(templateHull)
    targetHull = cv2.convexHull(target_contour)
    targetArea = cv2.contourArea(targetHull)

    val = cv2.matchShapes(target_contour,template_shape,1,0.0)
    haus_dist = hausdorffDistanceExtractor.computeDistance(target_contour, template_shape)

    return target_contour, val, (targetArea/templateArea), haus_dist


def get_template_contours():
    template = cv2.imread("../template.jpg")
    template2 = cv2.imread("../ruler2.jpg")

    template_edged = cv2.Canny(template, 15, 100)
    template_edged2 = cv2.Canny(template2, 15, 100)

    edged_img = cv2.dilate(template_edged, None, iterations=1)
    edged_img2 = cv2.dilate(template_edged2, None, iterations=1)    


    im2, template_shapes, hierarchy = cv2.findContours(edged_img, 2,1)
    im2_2, template_shapes2, hierarchy2 = cv2.findContours(edged_img2, 2,1)
 
    

    return template_shapes[0], template_shapes[2], template_shapes2[1]

def get_width_from_ruler(dB, rulerWidth):
    return dB/float(rulerWidth)

def get_scaled_image(image_full):
    orig_cols = len(image_full[0]) 
    orig_rows = len(image_full)
    fx = float(640.0/orig_cols)
    fy = float(480.0/orig_rows)

    rescaled_image = cv2.resize( image_full, (0,0), fx = fx, fy = fx)
    return rescaled_image

def get_ruler_image(imageName, thresh_val, blur_window):
    # load the image, convert it to grayscale, and blur it slightly
    image_full = cv2.imread(imageName)
    rescaled_image = get_scaled_image(image_full)
 
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
    
    
    return image, gray, threshold, int(rows/2)

def get_color_image(image, lower_color, upper_color):
    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(image, lower_color, upper_color)
    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(image,image, mask= mask)
    return res

def get_image_with_color_mask(imageName, thresh_val, blur_window):
    image_full = cv2.imread(imageName)
    rescaled_image = get_scaled_image(image_full)

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

    return image, gray, threshold, rows

def get_image(imageName, thresh_val, blur_window):
    # load the image, convert it to grayscale, and blur it slightly
    image_full = cv2.imread(imageName)
    rescaled_image = get_scaled_image(image_full)

    rows = len(rescaled_image)
    cols = len(rescaled_image[0])    
    image = rescaled_image

    #maybe use this to see if we should threshold?
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    gray = cv2.GaussianBlur(gray, (blur_window, blur_window), 0)
 
    mid_row_start = (rows/2)-thresh_val
    mid_col_start = (cols/2)-thresh_val
    
    mid_row_end = mid_row_start+thresh_val
    mid_col_end = mid_col_start+thresh_val
    mid_patch = gray[mid_row_start:mid_row_end, mid_col_start:mid_col_end]
    mn = np.mean(mid_patch) 

    retval, threshold = cv2.threshold(gray,mn,255,cv2.THRESH_BINARY)
    

    return image, gray, threshold, int(rows/2)

def find_edges(gray, threshhold, is_ruler=False):
    # perform edge detection, then perform a dilation + erosion to
    # close gaps in between object edges
    if not is_ruler:
        edged_img = cv2.Canny(threshhold, 50, 100)
        
    else:
        edged_img = cv2.Canny(gray, 60, 255)

    edged_img = cv2.dilate(edged_img, None, iterations=1)
    edged_img = cv2.erode(edged_img, None, iterations=1)
    


    return edged_img

def read_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('allimages', metavar='fp', nargs='+', help='file names')
    ap.add_argument("-i", "--image", required=False,
        help="path to the input image")


    ap.add_argument("-s", "--show", required=False,
        help="show the results. if not set, the results will write to a csv file")


    args = vars(ap.parse_args())


    rulerWidth = 8.5

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
        


    return imageName, showResults, rulerWidth

def get_largest_edge(cnts):
    if len(cnts) == 0:
        return None, None
    
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

    return pixelsPerMetric, dimB


def do_color_image_match(imageName, thresh_val, blur_window):
    #try the color image
    color_image, gray, thresh1, mid_row = get_image_with_color_mask(imageName, thresh_val, blur_window)
    orig = color_image.copy()
    edged = find_edges(gray, thresh1)

    # find contours in the edge map
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]

    abalone_contour, abalone_size = get_largest_edge(cnts)
    if abalone_contour is None:
        return None, None, None

    abalone, abalone_val, area_perc, area_dist= sort_by_matching_shape(abalone_contour, abalone_template_contour)
    return abalone, abalone_val, area_perc, area_dist

#experimental, didn't work so far
def do_feature_match(input_img):
    img1 = cv2.imread(input_img)          # queryImage
    img2 = cv2.imread('../template.jpg') # trainImage
     
    # Initiate ORB detector
    orb = cv2.ORB_create()
    
    # find the keypoints and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(img1,None)
    kp2, des2 = orb.detectAndCompute(img2,None)
    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
     
    # Match descriptors.
    matches = bf.match(des1,des2)
    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)
     
    # Draw first 10 matches.
    img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:10], img1, flags=0)
    cv2.imshow("feature match", img3)
    cv2.waitKey(0)

#experimental, didn't work so far
def do_stereo_work():
    img1 = cv2.imread("2016-08-01 18_21_56.1A.jpg")
    img1_re = cv2.resize( img1, (0,0), fx = 0.25, fy = 0.25)
    gray1 = cv2.cvtColor(img1_re, cv2.COLOR_BGR2GRAY)

    img2 = cv2.imread("2016-08-01 18_21_56.2A.jpg")
    img2_re = cv2.resize( img2, (0,0), fx = 0.25, fy = 0.25)
    gray2 = cv2.cvtColor(img2_re, cv2.COLOR_BGR2GRAY)

    stereo = cv2.StereoBM_create(numDisparities=32, blockSize=15)
    disparity = stereo.compute(gray1,gray2)
    cv2.imshow("stereo", disparity)
    cv2.waitKey(0)

def get_real_size(imageName):
    real_sizes = "../real_sizes.csv"
    size = "Unknown"
    with open(real_sizes, 'rU') as csvfile:

        csvreader = csv.reader(csvfile, delimiter=delimeter, quotechar='|')
        try:
            for row in csvreader:
                name = row[0]
                currSize = row[1]
                name = "{}.jpg".format(name)
                name = name.replace(":", "_")
                
                if name == imageName:
                    return float(currSize)
        except StandardError, e:
            print "problem trying to find size for {}: {}".format(imageName, e)
                
    return size

def get_best_contour(by_value, lower_area, upper_area, which_one):
    
    i=0
    smallest_values = []

    target_contour = None

    for values in by_value:
        #find the smallest 5 that fall within the area range
        if i > 4:
            break
        area_perc = values[1]
        haus_dist = values[2]
        contour = values[3]
        contour_key = values[4]
        if area_perc > lower_area and area_perc < upper_area:
            smallest_values.append(values)
            i+=1


    print "smallest shape value for {} is {}".format(which_one, smallest_values[0][0])
    min_dist = 100000000
    target_contour = None
    taret_val = None
    target_area = 0.0
    #now, from the 5 with the smallest value (the best shape match)
    #find the one that has the smallest dist value
    #should be a good compromise - may need to weigh these 2?
    for values in smallest_values:
        val = values[0]
        area_perc = values[1]
        haus_dist = values[2]
        contour = values[3]
        contour_key = values[4]
        if haus_dist < min_dist:
            target_area = area_perc
            target_contour = contour
            target_val = val
            min_dist = haus_dist

    print "best {} has a min shape of {} with dist of {}; area_perc is {}".format(which_one, target_val, min_dist, target_area)
    return target_contour



def get_best_contours(abalone_shapes, ruler_shapes):
    
    by_value = sorted(abalone_shapes, key=lambda abalone: abalone[0]) 
    abalone_contour = get_best_contour(by_value, 0.6, 1.5, "abalone")

    by_value = sorted(ruler_shapes, key=lambda ruler: ruler[0]) 
    ruler_contour = get_best_contour(by_value, 0.75, 2.0, "ruler")


    return abalone_contour, ruler_contour

#width of US quarter in inches
quarter_width = 0.955

#do_stereo_work()
#all the work
thresholds = [5,10, 30, 50]
#thresholds = [10]
blurs = [1,3,5,7]

#thresholds = [30]
#blurs = [5]
bestRulerContour = None
bestAbaloneContour = None

minRulerValue = 1000000
minAbaloneValue = 1000000
bestRulerPerc = -1.0
bestAbalonePerc = -1.0
(imageName, showResults, rulerWidth) = read_args()

ruler_template_contour, abalone_template_contour, alt_ruler_contour= get_template_contours()

abalone_temp_moment = cv2.moments(abalone_template_contour)
tX = int(abalone_temp_moment["m10"] / abalone_temp_moment["m00"])
tY = int(abalone_temp_moment["m01"] / abalone_temp_moment["m00"])
print "tX:{},tY:{}".format(tX, tY)
#img = get_scaled_image(cv2.imread(imageName))
#templateImg = get_scaled_image(cv2.imread("../abalone1.jpg"))
#res = cv2.matchTemplate(img,templateImg,cv2.TM_CCORR_NORMED)

minEdged = None
circles = []
abalone_shapes = []
ruler_shapes = []
hausdorffDistanceExtractor = cv2.createHausdorffDistanceExtractor()
for thresh_val in thresholds:
    for blur in blurs:
        key = "{}t:{}b".format(thresh_val, blur)
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
        if abalone_contour is None:
            continue

        #(template_contours, _) = contours.sort_contours(template_contours)
        pixelsPerMetric = None
        abalone, abalone_val, area_perc, abalone_dist = sort_by_matching_shape(abalone_contour, abalone_template_contour)
        abalone_shapes.append((abalone_val, area_perc, abalone_dist, abalone, key+"ab"))
        if abalone_val < minAbaloneValue:
            if area_perc > 0.5 and area_perc < 1.40:
                minAbaloneValue = abalone_val
                bestAbaloneContour = abalone.copy()
                bestAbalonePerc = area_perc
           
        #read the image for the ruler - assumes its in the lower half
        ruler_image, ruler_gray, ruler_thresh, yoffset = get_ruler_image(imageName, thresh_val, blur)

        ruler_edged = find_edges(ruler_gray, ruler_thresh, True)
        #find the contours in the ruler half image
        ruler_contours = cv2.findContours(ruler_edged.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)

        ruler_contours = ruler_contours[0] if imutils.is_cv2() else ruler_contours[1]
        ruler_contour, ruler_size = get_largest_edge(ruler_contours)
        if ruler_contour is None:
            continue

        ruler, ruler_val, ruler_perc, ruler_dist = sort_by_matching_shape(ruler_contour, ruler_template_contour)
        ruler_shapes.append((ruler_val, ruler_perc, ruler_dist, ruler, key+"ru"))
        #print " ruler value is {}; area fraction is {}; thresh is {}; blur is {}".format(ruler_val, ruler_perc, thresh_val, blur)

        #ruler_process_val = ruler_val
        
        if ruler_val < minRulerValue:
            if ruler_perc > 0.75 and ruler_perc < 1.75:
                minRulerValue = ruler_val
                bestRulerContour = ruler.copy()
                bestRulerPerc = ruler_perc
        

        alt_ruler, alt_ruler_val, alt_ruler_perc, alt_ruler_dist = sort_by_matching_shape(ruler_contour, alt_ruler_contour)
        ruler_shapes.append((alt_ruler_val, alt_ruler_perc, alt_ruler_dist, alt_ruler, key+"ar"))
        #print "-- alt ruler value is {}; area fraction is {}; thresh is {}; blur is {}".format(alt_ruler_val, alt_ruler_perc, thresh_val, blur)
        
        #cv2.drawContours(orig_image, [ruler_contour], 0, (0,0,255), 3)

        #print "ruler val: {},{}p alt ruler_val: {},{}p".format(ruler_val, ruler_perc, alt_ruler_val, alt_ruler_perc)
        #ruler_process_val = ruler_val
        if alt_ruler_val < minRulerValue:
            if alt_ruler_perc > 0.75 and alt_ruler_perc < 2.0:

                minRulerValue = alt_ruler_val
                bestRulerContour = alt_ruler.copy()
                bestRulerPerc = alt_ruler_perc
            

        color_abalone, color_abalone_val, color_perc, color_dist = do_color_image_match(imageName, thresh_val, blur)
        abalone_shapes.append((color_abalone_val, color_perc, color_dist, color_abalone, key+"ca"))
        M = cv2.moments(color_abalone)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        circles.append((cX, cY))
        

        dX = abs(tX-cX)
        dY = abs(tY-cY)

        if color_abalone is not None:
            if color_abalone_val < minAbaloneValue:
                if color_perc > 0.5 and color_perc < 1.5:
                    minAbaloneValue = color_abalone_val
                    bestAbaloneContour = color_abalone.copy()
                    bestAbalonePerc = color_perc
newBestAbaloneContour, newBestRulerContour = get_best_contours(abalone_shapes, ruler_shapes)

pixelsPerMetric, rulerLength = draw_contour(orig, newBestRulerContour, None, "Ruler", mid_row)
pixelsPerMetric, abaloneLength = draw_contour(orig, newBestAbaloneContour, pixelsPerMetric, "Abalone", 0)
all_rows = {}

out_file = "../blue_data.csv"

delimeter = ","
quotechar = '|'
all_rows = {}
all_diffs = {}
last_total_diff = 0.0
if os.path.exists(out_file):
    with open(out_file, 'rb') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=delimeter, quotechar='|')
        
        try:
            for row in csvreader:
                name = row[0]
                size = row[1]
                real_size = row[2]
                diff = row[3]
                
                if name != "Total":
                    all_rows[name] = [size, real_size]
                    all_diffs[name] = float(diff)
                else:
                    last_total_diff = float(diff)

        except StandardError, e:
            print("problem here: {}".format(e))

try:
    real_size = get_real_size(imageName)
    diff = abs(((abaloneLength - real_size)/real_size)*100.0)
    all_rows[imageName] = [abaloneLength, real_size]
    all_diffs[imageName] = diff
    total_diffs = np.sum(all_diffs.values())
    with open(out_file, 'wb') as csvfile:
        writer = csv.writer(csvfile, delimiter=delimeter, quotechar=quotechar, quoting=csv.QUOTE_MINIMAL)
        
        for name, sizes in all_rows.items():
            diff = all_diffs.get(name)
            writer.writerow([name, sizes[0], sizes[1], diff])

        writer.writerow(["Total", 0,0,total_diffs])

    print "------------->>>>>>>> totals diffs are {}; previous total diffs were {}".format(total_diffs, last_total_diff)
except StandardError, e:
    print "error trying to write the real size and diff: {}".format(e)

if showResults:
    targetHull = cv2.convexHull(newBestAbaloneContour)
    cv2.drawContours(orig, [newBestRulerContour], 0, (0,255,0), 2,offset=(0,yoffset*2))
    cv2.drawContours(orig, [newBestAbaloneContour], 0, (0,255,255), 3)
    cv2.drawContours(orig, [targetHull], 0, (255,0,0), 4)
    for circle in circles:
        cv2.circle(orig, circle, 3, (255, 255, 255), -1)
    cv2.circle(orig, (tX, tY), 5, (10, 10, 10), -1)
    cv2.imshow("Size of Abalone", orig)
    cv2.waitKey(0)



    
