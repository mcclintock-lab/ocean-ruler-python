# import the necessary packages
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2


#my files
import matching 
import utils
import color_images as ci
import file_utils

def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

def get_template_contours(input_abalone_template):
    row_offset = 30
    col_offset = 30
    #by default, using the big abalone template
    abalone_template = cv2.imread(input_abalone_template)
    abalone_template = abalone_template[30:len(abalone_template),30:len(abalone_template[0])-30]

    ruler_only = cv2.imread("../ruler_only_2x.png")
    ruler_only = ruler_only[30:len(ruler_only),30:len(ruler_only[0])-30]

    alt_ruler_only = cv2.imread("../alt_ruler_only_2x.png")
    alt_ruler_only = alt_ruler_only[30:len(alt_ruler_only),30:len(alt_ruler_only[0])-30]

    quarter_only = cv2.imread("../quarter_only_2x.png")
    quarter_only = quarter_only[30:len(quarter_only),30:len(quarter_only[0])-30]

    
    template_edged = cv2.Canny(abalone_template, 15, 100)
    ruler_only_edged = cv2.Canny(ruler_only, 15, 100)
    alt_ruler_only_edged = cv2.Canny(alt_ruler_only, 15, 100)
    quarter_only_edged = cv2.Canny(quarter_only, 15,100)

    edged_img = cv2.dilate(template_edged, None, iterations=1)
    ruler_edged_img = cv2.dilate(ruler_only_edged, None, iterations=1)  
    alt_ruler_edged_img =   cv2.dilate(alt_ruler_only_edged, None, iterations=1)  
    quarter_edged_img = cv2.dilate(quarter_only_edged, None, iterations=1)

    im2, abalone_shapes, hierarchy = cv2.findContours(edged_img, cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
    abalone_shape = abalone_shapes[1]

    im2_e, ruler_shapes, hierarchy2 = cv2.findContours(ruler_edged_img,  cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    ruler_shape = ruler_shapes[0]

    im2_alt_e, alt_ruler_shapes, hierarchy2 = cv2.findContours(alt_ruler_edged_img,  cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    alt_ruler_shape = alt_ruler_shapes[1] 


    quarter_e, quarter_shapes, hierarchy2 = cv2.findContours(quarter_edged_img,  cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    quarter_shape = quarter_shapes[0] 

    return abalone_shape, ruler_shape, alt_ruler_shape,quarter_shape

def get_width_from_ruler(dB, rulerWidth):
    return dB/float(rulerWidth)

def get_scaled_image(image_full):
    target_cols = 1280.0
    target_rows = 960.0
    orig_cols = len(image_full[0]) 
    orig_rows = len(image_full)
    fx = float(target_cols/orig_cols)
    fy = float(target_rows/orig_rows)

    scaled_image = cv2.resize( image_full, (0,0), fx = fx, fy = fx)
    rows = int(target_rows)
    cols = int(target_cols)

    scaled_image = scaled_image[30:rows,30:cols-30]
    return scaled_image, rows-30, cols-60


def get_ruler_image(input_image, thresh_val, blur_window):

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
    
    
    return input_image, gray, threshold, int(rows/2)


def get_image(input_image, thresh_val, blur_window):

    rows = len(input_image)
    cols = len(input_image[0])    
    image = input_image

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



def read_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('allimages', metavar='fp', nargs='+', help='file names')
    ap.add_argument("-i", "--image", required=False,
        help="path to the input image")
    ap.add_argument("-s", "--show", required=False,
        help="show the results. if not set, the results will write to a csv file")
    ap.add_argument("-o", "--output_file", required=False,
        help="file to read/write results from")
    ap.add_argument("-c", "--contour", required=False,
        help="use the big or small contour. will default to big, assuming most of the image is filled by abalone")

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
        
    out_file = args['output_file']
    if not out_file:
        out_file ="./data.csv"

    ab_size = args['contour']

    abalone_template = None
    if not ab_size:
        abalone_template = "../big_abalone_only_2x.png"
    else:
        if ab_size == "small":
            abalone_template = "../abalone_only_2x.png"
        else:
            abalone_template = "../big_abalone_only_2x.png"

    return imageName, showResults, rulerWidth, out_file, abalone_template


def draw_contour(base_img, con, pixelsPerMetric, pre, top_offset, rulerWidth,is_quarter):
    brect = cv2.boundingRect(con)
    brect_arr = np.array(brect, dtype="int")

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

    (tltrX, tltrY) = midpoint(tl, tr)
    (blbrX, blbrY) = midpoint(bl, br)

    # compute the midpoint between the top-left and top-right points,
    # followed by the midpoint between the top-righ and bottom-right
    (tlblX, tlblY) = midpoint(tl, bl)
    (trbrX, trbrY) = midpoint(tr, br)

    # draw the midpoints on the image
    cv2.circle(base_img, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
    cv2.circle(base_img, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)

    # draw lines between the midpoints

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
    if pre == "Ruler":
            # draw the object sizes on the image
        cv2.putText(base_img, "{}: {}in".format(pre,dimB),
            (int(trbrX)+10, int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
            0.65, (255, 255, 255), 2)
    else:
        # draw the object sizes on the image
        cv2.putText(base_img, "{}".format(pre),
            (int(trbrX)+10, int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
            0.65, (255, 255, 255), 2)

        cv2.putText(base_img, "{:.1f}in".format(dimB),
            (int(trbrX)+10, int(trbrY)+50), cv2.FONT_HERSHEY_SIMPLEX,
            0.65, (255, 255, 255), 2)

    return pixelsPerMetric, dimB


#experimental, didn't work so far
def do_feature_match(input_img):
    img1 = cv2.imread(input_img) # queryImage
    img2 = cv2.imread('../abalone_only_2x.png') # trainImage
     
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




def get_best_contour(shapes, lower_area, upper_area, which_one):
    ab_by_combined = sorted(shapes, key=lambda shape: shape[5])
    ab_by_value =  sorted(shapes, key=lambda shape: shape[0])
    ab_by_dist = sorted(shapes, key=lambda shape: shape[2])
    
    i=0
    lowest_val = ab_by_value[0][0]
    lowest_combined = ab_by_combined[0][5]
    lowest_dist = ab_by_dist[0][2]
    lowest_cdiff = ab_by_dist[0][6]


    minValue = 1000000
    targetContour = None
    targetKey = ""
    
    for values in ab_by_combined:
        val = values[0]
        if lowest_val == 0.0:
            norm_val = 0.0
        else:
            norm_val = val/lowest_val
        haus_dist = values[2]
        if lowest_dist == 0.0:
            norm_dist = 0.0
        else:
            norm_dist = haus_dist/lowest_dist
        

        area_perc = values[1]
        contour = values[3]
        contour_key = values[4]
        combined = values[5]
        
        centroid_diff = values[6]
        if lowest_cdiff == 0.0:
            norm_cdiff = 0.0
        else:
            norm_cdiff = centroid_diff/lowest_cdiff

        combined = val*haus_dist*centroid_diff

        #if contour_key.endswith("_big_color_ab"):
        #    print "big: {} {}.combined: {}; val:{};dist:{};key:{};area:{};cdiff:{}".format(which_one, i,combined,val,haus_dist,contour_key,area_perc,centroid_diff)
            
        if contour_key.endswith("_quarter"):
            lower_area = 0.3
        if i < 6:
            print "{} {}.combined: {}; val:{};dist:{};key:{};area:{};cdiff:{}".format(which_one, i,combined,val,haus_dist,contour_key,area_perc,centroid_diff)
            i+=1
        if area_perc > lower_area and area_perc < upper_area:
            x,y,w,h = cv2.boundingRect(contour)
            #get rid of the ones with big outlying streaks or edges
            if combined < minValue and ((w/1280.0) < 0.9) and ((h/960.0) < 0.9):
                minValue = combined
                targetContour = contour
                targetKey = contour_key

    return targetContour, targetKey


def get_best_contours(abalone_shapes, ruler_shapes):

    ab_contour,ab_key = get_best_contour(abalone_shapes, 0.46, 1.45, "abalone")
    r_contour, ruler_key = get_best_contour(ruler_shapes, 0.46, 1.9, "ruler")

    return ab_contour, ab_key, r_contour,ruler_key

#get the set of largest contours, then find the one that has the best shape match
#note: have to cycle through the whole set because there can be contours of the same size that are diff
def get_abalone_contour(input_image, template_contour, thresh_val, blur):
    #segmentation and edge finding
    orig_image, gray, thresh1, mid_row = get_image(input_image, thresh_val, blur)
    orig = orig_image.copy()
    edged = utils.find_edges(gray, thresh1)
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]

    #get all the largest edges
    contours, size = utils.get_largest_edge(cnts)
    smallest_combined = 10000000.0
    target_contour = None
    rval = 1000000
    aperc = 1000000
    adist = 1000000
    cdiff = 1000000
    for contour in contours:
        the_contour, result_val, area_perc, area_dist, centroid_diff= matching.sort_by_matching_shape(contour, template_contour, False,input_image)
        comb = result_val*area_dist
        if comb < smallest_combined:
            smallest_combined = comb
            rval = result_val
            aperc = area_perc
            adist = area_dist
            target_contour = the_contour
            cdiff = centroid_diff

    epsilon = 0.003*cv2.arcLength(target_contour,True)
    approx = cv2.approxPolyDP(target_contour,epsilon,True)
    return approx


def get_ruler_contour(input_image, template_contour, thresh_val, blur, showImg, top_offset=0):
    ruler_image, ruler_gray, ruler_thresh, yoffset = get_ruler_image(input_image, thresh_val, blur)
    ruler_edged = utils.find_edges(ruler_gray, ruler_thresh, True)
    #find the contours in the ruler half image
    #these were APPROX_SIMPLE and RETR_EXTR
    ruler_contours = cv2.findContours(ruler_edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ruler_contours = ruler_contours[0] if imutils.is_cv2() else ruler_contours[1]
    
    smallest_combined = 10000000.0
    target_contour = None

    ok_contours = []
    for contour in ruler_contours:
        the_contour, result_val, area_perc, area_dist, centroid_diff = matching.sort_by_matching_shape(contour, template_contour, False,input_image)
        
            
        comb = result_val*area_dist
        if comb < smallest_combined and area_perc > 0.25 and area_perc < 2.0:
            smallest_combined = comb
            target_contour = the_contour
            ok_contours.append(the_contour)


    if showImg:
        cv2.drawContours(input_image, [target_contour], 0, (0,0,255), 4)
        cv2.drawContours(input_image, [template_contour], 0, (255,0,0), 3)
        utils.show_img("ruler contour {}x{}".format(thresh_val, blur), input_image)
    return target_contour


def add_shape_with_color(shapes, input_image, template_contour, thresh_val, blur, key,printResults, showImg,contour_color):
    #find the matching shape using the color images with blue background
    #(input_image, template_contour, thresh_val, blur_window, showImg):
    col_contour, val, area_perc, dist, centroid_diff = ci.do_color_image_match(input_image, template_contour, thresh_val, blur,printResults, showImg, contour_color)
    if printResults:
        cv2.fillPoly(input_image, [col_contour], contour_color)
        cv2.fillPoly(input_image, [template_contour], (255,255,255))
        utils.show_img("{}: {}".format(thresh_val, blur), input_image)
    shapes.append((val, area_perc, dist, col_contour, key,val*dist*centroid_diff, centroid_diff))
    return shapes

def add_shape_by_match(shapes, input_image, target_contour, template_contour, thresh_val, blur, key,printResults, top_offset, use_hull):
    ii = input_image.copy()
    if target_contour is None:
        return shapes

    #find the matching ruler shape with the alt ruler 
    s_contour, val, area_perc, dist, centroid_diff = matching.sort_by_matching_shape(target_contour, template_contour, use_hull, input_image)
    if printResults:
        cv2.fillPoly(ii, [s_contour], (0,255,0), offset=(0,top_offset))
        cv2.fillPoly(ii, [template_contour], (0,0,255), offset=(0,top_offset))
        utils.show_img("color shape {}x{}".format(thresh_val, blur), ii)
    shapes.append((val, area_perc, dist, s_contour, key, val*dist*centroid_diff, centroid_diff))
    return shapes

def match_template(input_image, template, showImg):
    template_image = cv2.imread(template)

    h,w = template_image.shape[:2]
    res = cv2.matchTemplate(input_image,template_image,cv2.TM_CCOEFF)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    top_left = max_loc
    x = top_left[0]
    y = top_left[1]
    #bottom_right = (top_left[0] + w, top_left[1] + h)
   
    #cv2.rectangle(input_image,top_left, bottom_right, 255, 2)
    mask = np.zeros(input_image.shape,np.uint8)
    mask[y:y+h,x:x+w] = input_image[y:y+h,x:x+w]
    if showImg:
        utils.show_img("template ", mask)
    return mask, y

def find_image():
    #width of US quarter in inches
    quarter_width = 0.955

    #all the work
    thresholds = [10, 30, 50,70]
    blurs = [1,3,5,7]

    #thresholds = [30]
    #blurs = [5]
    bestRulerContour = None
    bestAbaloneContour = None

    (imageName, showResults, rulerWidth, out_file, input_abalone_template) = read_args()
    abalone_template_contour, ruler_template_contour, alt_ruler_template_contour, quarter_template_contour = get_template_contours(input_abalone_template)

    minEdged = None
    abalone_shapes = []
    ruler_shapes = []
    pixelsPerMetric = None
    
    image_full = cv2.imread(imageName)
    #image_full = cv2.cvtColor(image_full, cv2.COLOR_BGR2HSV)

    rescaled_image, rows, cols = get_scaled_image(image_full)

    ruler_image = rescaled_image[int(rows/2):rows, 0:cols].copy()


    #show_img("half ruler", ruler_image.copy())
    ruler_mask, ruler_top_offset = match_template(ruler_image.copy(), "../ruler_image_2x.png",False)
    alt_ruler_mask, alt_ruler_top_offset = match_template(ruler_image.copy(), "../alt_ruler_image2_2x.png",False)

    qmask, qoffset= match_template(ruler_image.copy(), "../quarter_image_template_2x.png", False)
    
    for thresh_val in thresholds:
        for blur in blurs:
            key = "{}thresh_{}blur".format(thresh_val, blur)

            print "working on {}x{}".format(thresh_val, blur)
            #read the image for the abalone contour with these settings
            abalone_contour = get_abalone_contour(rescaled_image.copy(), abalone_template_contour, thresh_val, blur)
            abalone_shapes = add_shape_by_match(abalone_shapes, rescaled_image.copy(), abalone_contour, 
                abalone_template_contour.copy(),thresh_val, blur, (key+"_bw_ab"),False, 0,False)

            #read the image for the ruler - assumes its in the lower half
            ruler_contour = get_ruler_contour(ruler_image.copy(),ruler_template_contour, thresh_val, blur, False)

            ruler_shapes = add_shape_by_match(ruler_shapes, ruler_image.copy(), ruler_contour, 
                ruler_template_contour.copy(),thresh_val, blur, (key+"_bw_ru"),False, 0,False)

            alt_ruler_contour = get_ruler_contour(ruler_image.copy(),alt_ruler_template_contour, thresh_val, blur, False)
            ruler_shapes = add_shape_by_match(ruler_shapes, ruler_image.copy(), ruler_contour, 
                alt_ruler_template_contour.copy(), thresh_val, blur, (key+"_bw_alt_ru"),False, 0,False)
            
            #image template matches
            ruler_trimmed = ruler_mask[5:len(ruler_mask), 10:len(ruler_mask[0])]
            masked_ruler_contour = get_ruler_contour(ruler_image.copy(), ruler_template_contour, thresh_val, blur, False)
            ruler_shapes = add_shape_by_match(ruler_shapes, ruler_trimmed.copy(), masked_ruler_contour, 
                ruler_template_contour, thresh_val, blur, (key+"_bw_mask_ru"), False, ruler_top_offset, False)

            alt_ruler_trimmed = alt_ruler_mask[10:len(alt_ruler_mask)-5, 5:len(alt_ruler_mask[0])-5]
            alt_masked_ruler_contour = get_ruler_contour(ruler_image.copy(), alt_ruler_template_contour, thresh_val, blur,False)
            ruler_shapes = add_shape_by_match(ruler_shapes, alt_ruler_trimmed.copy(), 
                alt_masked_ruler_contour, alt_ruler_template_contour, thresh_val, blur, (key+"_bw_alt_mask_ru"), False, 
                alt_ruler_top_offset, False)

            '''
            quarter_masked_contour = get_ruler_contour(qmask, quarter_template_contour, thresh_val, blur, False)
            ruler_shapes = add_shape_by_match(ruler_shapes, ruler_image.copy(), 
                quarter_masked_contour, quarter_template_contour, thresh_val, blur, (key+"_masked_quarter"), False, 
                qoffset, False)
            '''

            #quarter image match
            quarter_contour = get_ruler_contour(ruler_image.copy(), quarter_template_contour.copy(), thresh_val,blur,False,0)
            ruler_shapes = add_shape_by_match(ruler_shapes, ruler_image.copy(), quarter_contour, 
                quarter_template_contour.copy(), thresh_val, blur, (key+"_quarter"), False, 20, False)
            

            #color abalone
            abalone_shapes = add_shape_with_color(abalone_shapes, rescaled_image.copy(), 
                abalone_template_contour.copy(), thresh_val, blur, (key+"_color_ab"),False, False,(0,0,255))
          
            #big color abalone
            #abalone_shapes = add_shape_with_color(abalone_shapes, rescaled_image.copy(), 
            #    big_abalone_template_contour.copy(), thresh_val, blur, (key+"_big_color_ab"),False, False,(0,0,255))
            

            #color quarter
            ruler_shapes = add_shape_with_color(ruler_shapes,ruler_image.copy(), 
                quarter_template_contour.copy(), thresh_val, blur, (key+"_color_quarter"),False, False, (255,0,0))

            #color ruler
            ruler_shapes = add_shape_with_color(ruler_shapes,ruler_image.copy(), 
                ruler_template_contour.copy(), thresh_val, blur, (key+"_color_ru"),False, False, (255,0,0))

            #color ruler
            ruler_shapes = add_shape_with_color(ruler_shapes,ruler_image.copy(), 
                alt_ruler_template_contour.copy(), thresh_val, blur, (key+"_alt_color_ru"),False, False, (0,255,0))
            #ruler_shapes = add_shape_with_color(ruler_shapes,ruler_image.copy(), alt_ruler_template_contour, thresh_val, blur, key+"acru",True)

            

    newBestAbaloneContour, bestAbaloneKey, newBestRulerContour,bestRulerKey = get_best_contours(abalone_shapes, ruler_shapes)

    is_quarter = bestRulerKey.endswith("_quarter")
    off = 465 if is_quarter else 0

    pixelsPerMetric, rulerLength = draw_contour(rescaled_image, newBestRulerContour, None, "Ruler", off, rulerWidth,is_quarter)
    pixelsPerMetric, abaloneLength = draw_contour(rescaled_image, newBestAbaloneContour, pixelsPerMetric, "Abalone", 0, rulerWidth, False)
    all_rows = {}


    file_utils.read_write_csv(out_file, imageName, bestAbaloneKey, bestRulerKey, abaloneLength, rulerLength)
    
    if showResults:
        cv2.namedWindow(imageName, cv2.WINDOW_NORMAL)
        cv2.drawContours(rescaled_image, [newBestRulerContour], 0, (0,255,0), 3,offset=(0,465))
        cv2.drawContours(rescaled_image, [newBestAbaloneContour], 0, (255,0,0), 3)
        bounded = cv2.copyMakeBorder(rescaled_image,10,10,10,10,cv2.BORDER_CONSTANT,value=(0,0,0))

        cv2.imshow(imageName, bounded)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

#lazily run the it here...
find_image()




    
