import cv2
import time
import utils
import numpy as np
import color_images as ci
import math
import constants
import matplotlib.pyplot as plt

#find the abalone or scallop edges...
def get_target_oval_contour(input_image, abalone_template_contour, lower_percent_bounds, white_or_gray, 
                            use_opposite, is_square_ref_object, fishery_type):
    
    target_contour = None
    gray = utils.get_gray_image(input_image, white_or_gray, use_opposite)
    
    if use_opposite:
        white_or_gray = not white_or_gray

    blur = cv2.GaussianBlur(gray, (5,5),0)
    #this was white or gray
    if white_or_gray or utils.is_dark_gray(input_image):
        lower_bound = 20
        upper_bound = 100

        edged_img = cv2.Canny(blur, lower_bound, upper_bound,9)
        #edged_img = get_canny(blur, 0.50)
        '''
        cv2.imshow("original, auto canny", np.hstack([edged_img, new_edged_img]))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        '''
    else:
        lower_bound = 0
        upper_bound = 99
        
        hsv = cv2.cvtColor(input_image, cv2.COLOR_BGR2HSV)
       
        hue_layer = hsv[:,:,0]
        #hue_layer[hue_layer > 80] = 80

        sat_layer = hsv[:,:,1]
        value_layer = hsv[:,:,2]
        

        '''
        cv2.imshow("sat", sat_layer)
        cv2.imshow("val", value_layer)
        '''
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        if utils.is_dark_gray(input_image):
            print("its dark gray")
            edged_img = cv2.Canny(blur, lower_bound, upper_bound,7) 
        else:
            #blur = cv2.GaussianBlur(hue_layer, (5,5),0)
            #edged_img = get_canny(blur, 0.6)
            hsv = cv2.cvtColor(input_image, cv2.COLOR_BGR2HSV)

            hue_layer = hsv[:,:,0]
            ret, thresh = cv2.threshold(hue_layer, 30, 100, 0)
            #utils.show_img("thresh hue", thresh)
            edged_img = cv2.Canny(thresh, lower_bound, upper_bound,7) 



    if False:
        print("is white: {}".format(white_or_gray))
        utils.show_img("edges", edged_img)
        

    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(13,17))
    erode_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,7))
    kernel = np.ones((3,3), np.uint8)

 
    iters=3
    edged_img = cv2.dilate(edged_img, kernel, iterations=iters)
    edged_img = cv2.morphologyEx(edged_img, cv2.MORPH_CLOSE, dilate_kernel)
    edged_img = cv2.erode(edged_img, erode_kernel, iterations=1)


    ret, thresh = cv2.threshold(edged_img.copy(), 127,255,0)
    cnts = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
 
    if (cnts[1] is None or len(cnts[1]) == 0) and (cnts[0] is not None and len(cnts[0]) > 0):
        largest, largest_contours_only = utils.get_largest_edges(cnts[0])
    else :
        largest, largest_contours_only = utils.get_largest_edges(cnts[1])

    if False:
        try:
            cv2.drawContours(input_image, cnts[0], -1, (255,0,0),4)
        except Exception as e:
            print("nothing in 0")
        try:
            cv2.drawContours(input_image, cnts[1], -1, (0,0,255),4)
        except Exception:
            print("couldn't draw 1")
        try:
            cv2.drawContours(input_image, cnts[2], -1, (0,255,0),4)
        except Exception:
            print("coudln't draw 2")

        #cv2.drawContours(input_image, [largest], -1, (255,255,255),3)
        utils.show_img("cnts 0 is blue, 1 is red,2 is green ", input_image)
    

    if False:
        n=50
        for l in largest:

            if n < 300:
                cv2.drawContours(input_image, l[1], -1, (n,n-20,n*2),12)
            n=n+50
        utils.show_img("biggest contours", input_image)

    
    ncols = len(input_image[0]) 
    nrows = len(input_image)
    img_area = nrows*ncols
    minVal = 100000000
    dex = 0
    if largest is not None and len(largest) > 0 and largest[0] and largest[1] is not None:
        
        for i, contour in enumerate(largest):
            perc = contour[0]/img_area
            actual_perc = contour[2]/img_area
            current_contour = contour[1]
            if is_square_ref_object and is_square_contour(current_contour):
                continue

            if perc <= 0.95 and perc > lower_percent_bounds:
                if(current_contour is None or len(current_contour) == 0):
                    continue

                x,y,w,h = cv2.boundingRect(current_contour)
                compactness = get_compactness(current_contour)
                if w >= ncols*0.98 and h >= nrows*0.98:
                    #sanity check for the crazy squiggly contours, like in
                    #Glass_Beach_Memorial_Day_\ -\ 1252_181.jpg
                    matchVal = cv2.matchShapes(current_contour, abalone_template_contour, 2, 0.0)
                else:
                    #do I need to compare to the abalone shape, or use compactness and size only?
                    matchVal = 1
                val=matchVal*compactness*(1/actual_perc)

                if matchVal < minVal:
                    dex = i
                    minVal = matchVal
        
                    target_contour = current_contour


 
    #orig contours are returned for display/testing
    
    return target_contour, cnts[1]

#calculation to see if its a smooth, compact contour - like a quarter
def get_compactness(match_contour):
    matchPerimeter = cv2.arcLength(match_contour,True)
    matchArea = cv2.contourArea(match_contour)
    compactness = (matchPerimeter*matchPerimeter)/(4*matchArea*math.pi)
    return compactness

def get_width_and_height(cnt):
    leftmost = tuple(cnt[cnt[:,:,0].argmin()][0])
    rightmost = tuple(cnt[cnt[:,:,0].argmax()][0])
    topmost = tuple(cnt[cnt[:,:,1].argmin()][0])
    bottommost = tuple(cnt[cnt[:,:,1].argmax()][0])
    w = abs(leftmost[0] - rightmost[0])
    h = abs(topmost[1] - bottommost[1])
    return w,h

def is_square_contour(contour):
    w,h = get_width_and_height(contour)
    ratio = float(w)/float(h)
    #turn it into a polygon and see if its got 4 sides
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.01*perimeter, True)
    area = cv2.contourArea(contour)
    
    if ratio >= 0.85 and ratio <= 1.15:
        if len(approx) == 4:
            return True
        else:
            return False
    else:
        return False

#this is the big test square for assessing distance to camera v. pixel size
def get_big_square_target_contour(input_image, size_place):
    target_contour = None
    white_or_gray = True
    lower_percent_bounds = 0.15

    denoised = cv2.fastNlMeansDenoisingColored(input_image,None,10,10,5,9)
    gray = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)
    

    if white_or_gray:
        lower_bound = 0
        upper_bound = 100
        thresh_lower = 70
        thresh_upper = 255


    scale_img = cv2.Canny(gray, lower_bound, upper_bound,7) 
  
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(13,17))
    erode_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
    kernel = np.ones((3,3), np.uint8)

    iters=3
        
    edged_img = cv2.dilate(scale_img, kernel, iterations=iters)
   
    edged_img = cv2.morphologyEx(edged_img, cv2.MORPH_CLOSE, dilate_kernel)
    edged_img = cv2.erode(edged_img, erode_kernel, iterations=1)


    #gray_denoised = cv2.cvtColor(edged_img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(edged_img.copy(), 127,255,0)
    cnts = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    contours = cnts[1]

    dex = 0
    tcontours = []

    biggest = 0
    target_dex = 0
    for i, contour in enumerate(contours):
        try:
            tcontours.append(contour)
            carea = cv2.contourArea(contour)  
            if carea > biggest:
                biggest = carea
                target_dex = i
        except Exception as e:

            continue

    nextBiggestDex = 0
    nextBiggest = 0
    
    for j, contour in enumerate(contours):
        carea = cv2.contourArea(contour)
        
        if j != target_dex and carea > nextBiggest and carea < biggest*0.75:
            nextBiggest = carea
            nextBiggestDex = j

    if size_place == 1:
        target_contour = tcontours[nextBiggestDex]
    else:
        target_contour = tcontours[target_dex]

    #orig contours are returned for display/testing
    if False:
        cv2.drawContours(input_image, [target_contour], 0, (0,255,255),4)
        utils.show_img("square contours {}".format(size_place), input_image)
    
    return target_contour, tcontours



def get_target_contour(clipped_image,original_image, template_contour, is_square_ref_object, is_abalone, isWhiteOrGray, fishery_type):

    lower_perc_bounds = 0.1
    if(not constants.isAbalone(fishery_type)):
        lower_perc_bounds = 0.05
    target_contour, orig_contours = get_target_oval_contour(clipped_image.copy(), template_contour, lower_perc_bounds, isWhiteOrGray, False, is_square_ref_object, fishery_type)
    if target_contour is None:
        if(not constants.isAbalone(fishery_type)):
            lower_perc_bounds = 0.01
        target_contour, orig_contours = get_target_oval_contour(clipped_image.copy(), template_contour, lower_perc_bounds, isWhiteOrGray, True, is_square_ref_object, fishery_type)
        if target_contour is None:
            if(not constants.isAbalone(fishery_type)):
                lower_perc_bounds = 0.01
            #in case the main abalone/scallop is cut off
            target_contour, orig_contours = get_target_oval_contour(original_image.copy(), template_contour, lower_perc_bounds, isWhiteOrGray, True, is_square_ref_object, fishery_type)
            
            
    ncols = len(clipped_image[0]) 
    nrows = len(clipped_image)
    img_area = nrows*ncols

    cx, cy = utils.get_centroid(target_contour)   
    x,y,w,h = cv2.boundingRect(target_contour)

    rectCX = x+int(w/2)
    rectCY = y+int(h/2)

    xDiff = abs(cx-rectCX)
    yDiff = abs(cy-rectCY)

    xOffset = cx - rectCX
    percOffset = abs(float(xOffset)/float(ncols))

    if abs(percOffset) > 0.008 and is_abalone:
        trimmed_contour = trim_abalone_contour(target_contour)
        if trimmed_contour is not None:
            target_contour = trimmed_contour

    contours = np.array(target_contour)

    return contours, orig_contours

def showResults(img1, img2):
    fig=plt.figure(figsize=(8, 8))
    fig.add_subplot(1,1, 1)
    plt.imshow(img1)
    fig.add_subplot(1,2, 2)
    plt.imshow(img2)
    plt.show()

def get_quarter_image(input_image, use_opposite, isWhiteOrGray, use_thresh, low_bounds=100):
    print("getting quarter image, isWhite or Gray::: {}".format(isWhiteOrGray))
    if not isWhiteOrGray:
        '''
        thresh_val = 30
        blur_window = 5
        first_pass = True
        is_ruler = True
        use_adaptive = False
        color_image, threshold_bw, color_img, mid_row = ci.get_image_with_color_mask(input_image, thresh_val, 
            blur_window, False, first_pass, is_ruler, use_adaptive)
        

        if utils.is_dark_gray(input_image):
            gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
        '''
        hsv = cv2.cvtColor(input_image.copy(), cv2.COLOR_BGR2HSV)
        hue_layer = hsv[:,:,0]
        ret, thresh = cv2.threshold(hue_layer, 30, 100, 0)
        scale_img = cv2.Canny(thresh, 0, 99,7) 
        #utils.show_img("color quarter", scale_img)
    else:
        print("going to denoise now...")
        if use_thresh:
            #hsv = cv2.cvtColor(input_image, cv2.COLOR_BGR2HSV)
            #value_layer = hsv[:,:,2]
            

            #note: calculating image stats doesnt help. problem is if quarter is in shadow, a high low-bounds (above 60) won't see it
            
            
            grayIn = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(grayIn, (5,5),0)
            '''
            hsv = cv2.cvtColor(input_image, cv2.COLOR_BGR2HSV)
            hue_layer = hsv[:,:,0]
            sat_layer = hsv[:,:,1]
            value_layer = hsv[:,:,2]
            
            cv2.imshow("hue, sat, value", np.hstack([hue_layer, sat_layer, value_layer]))
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            '''

            
            ret, gray = cv2.threshold(blur, low_bounds,255,0)
           
            #gray = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
            
            gray[gray < 254 ] = 0 
            #utils.show_img("gray", gray)
            #print("------>>> image type: {}".format(gray.dtype))
            scale_img = gray
        else:
            print("not doing thresh...")
            #denoised = cv2.fastNlMeansDenoisingColored(input_image,None,7,21,5,9)
            #gray = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)
            #scale_img = get_canny(input_image,0.35)
        
            #changed this on 3/23/20 - seems like bilateral gray works best on white background...
            blur = cv2.medianBlur(input_image,3)
            gray2 = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
            bilateralGray = cv2.bilateralFilter(gray2, 9, 5, 5)
            scale_img = get_canny(bilateralGray)

            if False:
                utils.show_img("quarter default ", scale_img)
                utils.show_img("bilateral gray ", scale_img2)

    return scale_img

def get_target_quarter_contours2(input_image, quarter_template_contour, use_opposite,  too_close_to_abalone=False, 
                                 isWhiteOrGray=True, use_thresh=False, low_bounds=100, target_contour=None,
                                 original_size=None,lastPass=False):
    ncols = len(input_image[0]) 
    nrows = len(input_image)
    img_area = nrows*ncols
    
    scale_img = get_quarter_image(input_image, use_opposite, isWhiteOrGray,use_thresh,low_bounds=low_bounds)
    
    kernel = np.ones((3,3), np.uint8)
    if not use_thresh:
        if low_bounds != 75:
            scale_img = cv2.dilate(scale_img, kernel, iterations=1)


    #ret, thresh = cv2.threshold(scale_img.copy(), 50,140,0)
    
    contour_image = scale_img.copy()
    if use_thresh:
        scale_cnts = cv2.findContours(scale_img, cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
        scale_contours = np.array(scale_cnts[1])
        #fill the contours to make sure circles are found
        #this helps for the contours that are circular on the outside but a mess inside,
        #which increases compactness substantially -- close up images with bright quarters 
        #where you can see details can have very complex contours
        for c in scale_contours:
            if len(c) >= 5:
                (x,y), radius = cv2.minEnclosingCircle(c)
                if radius >= 15 and radius <= 60:
                    cv2.fillPoly(contour_image, c, 0)
        
        scale_cnts = cv2.findContours(contour_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        scale_contours = np.array(scale_cnts[1])
        
        circle_image = contour_image.copy()
    else:
        ret, thresh = cv2.threshold(contour_image, 50,140,0)
        scale_cnts = cv2.findContours(contour_image, cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
        scale_contours = np.array(scale_cnts[1])
        circle_image = thresh.copy()
    print("iswg:{}, use thresh: {}, low_bounds:{}".format(isWhiteOrGray, use_thresh, low_bounds))
    if False:
        d = input_image.copy()
        cv2.drawContours(d, scale_contours, -1,(255,0,0),3)
        utils.show_img("contours for thresh: {}, low_bounds:{}".format(use_thresh, low_bounds),d)
        showOutput = True
    else:
        showOutput = False
        
    if False:
        utils.show_img("scale img", scale_img)

    matches, matching_contours, okComp = get_filtered_quarter_contours_by_radius(scale_contours, img_area, 
                                                                                 quarter_template_contour, showOutput,
                                                                                 lastPass=lastPass)
    
    if showOutput:
        d = input_image.copy()
        for ok in okComp:
            n = 0
            cv2.drawContours(d, [ok], 0,(n,n+75,n),3)
        utils.show_img("lighter are later, {}:{}".format(use_thresh, low_bounds), d)

    return matches, matching_contours, okComp, circle_image


def is_point_in_circle(point, circle):
    circleCenter = (circle[0], circle[1])
    radius = circle[2]
    inX = (point[0] <= circleCenter[0]+radius) and (point[0] >= circleCenter[0]-radius)
    inY = (point[1] <= circleCenter[1]+radius) and (point[1] >= circleCenter[1]-radius)
    return inX and inY

def get_matches(circles, matches):
    circle_matches = []
    for x, match in enumerate(matches):
        contourHull = cv2.convexHull(match,returnPoints=True)
        contourCenter = utils.get_centroid(match)
        if circles is not None and len(circles) > 0:
            for circ in circles[0,:]:
                center = (circ[0], circ[1])

                circleCenterInHull = cv2.pointPolygonTest(contourHull,center,False) >= 0
                hullCenterInCircle =  is_point_in_circle(contourCenter, circ)

                if(circleCenterInHull and hullCenterInCircle):
                    row = [circ, match, contourHull]
                    circle_matches.append(row)
    return circle_matches

def get_circle_info(circle):
    cx = circle[0]
    cy = circle[1]
    radius = circle[2]
    return cx, cy, radius



def get_filtered_quarter_contours(scale_contours, target_contour, img_area, check_roundness,original_size=None,lastPass=False):
    matches = []
    
    for i, scontour in enumerate(scale_contours):
        try:
            carea = cv2.contourArea(scontour)
            hull = cv2.convexHull(scontour,returnPoints=True)
            hullArea = cv2.contourArea(hull)
            perc = hullArea/original_size

            if perc <= 0.05:
                #if not utils.is_contour_enclosed(scontour, target_contour, False, not check_roundness):
                if check_roundness:
                    if utils.is_really_round(scontour):
                        matches.append(scontour)

                else:
                    matches.append(scontour)

        except Exception as e:
            continue

    return matches
                        #
def get_quarter_results(clippedImages, target_contour, quarter_template_contour, look_for_shapes, 
                        origCellCount, isWhiteOrGray,original_size=None, use_thresh=False,low_bounds=100,lastPass=False):
    results = []
    print("get qr iswg: {}".format(isWhiteOrGray))
    for i, ciData in enumerate(clippedImages):
        ci = ciData[0]
        xOffset = ciData[1]
        yOffset = ciData[2]

        refObjectCenterX, refObjectCenterY, refRadius, matches, score = get_quarter_dimensions(ci, target_contour, 
                                                                 quarter_template_contour, look_for_shapes, origCellCount, 0,0,
                                                                 isWhiteOrGray,original_size=original_size, 
                                                                 use_thresh=use_thresh,low_bounds=low_bounds,
                                                                 lastPass=lastPass)
        if refRadius > 0:
            results.append([refObjectCenterX+xOffset, refObjectCenterY+yOffset, refRadius, matches, score])
    
    return results

def get_best_quarter_dimensions(clippedImages, originalImage, target_contour, quarter_template_contour, look_for_shapes, origCellCount, isWhiteOrGray, original_size=None):
    results = []
    #first passed, quarter images clipped with target contour masked
    whichOne = ""
    imagesToUse = [clippedImages, originalImage, originalImage]
    for i, imageToUse in enumerate(imagesToUse):
        lastPass = i==2
        results = get_quarter_results(imageToUse, target_contour, quarter_template_contour, look_for_shapes, origCellCount, isWhiteOrGray,original_size=original_size,use_thresh=False,low_bounds=105, lastPass=lastPass)
        whichOne = "Thresh-105-{}".format(lastPass)
        if results is None or len(results) == 0:
            #second, without the target contour masked out
            results = get_quarter_results(imageToUse, target_contour, quarter_template_contour, look_for_shapes, origCellCount, isWhiteOrGray,original_size=original_size, use_thresh=True,low_bounds=105,lastPass=lastPass)
            whichOne = "NoThresh-105-{}".format(lastPass)
            if results is None or len(results) == 0:
                #for very bright images, do a really high number if last pass
                if lastPass:
                    low_bounds = 170
                else:
                    low_bounds = 165
                results = get_quarter_results(imageToUse, target_contour, quarter_template_contour, look_for_shapes, origCellCount, isWhiteOrGray,original_size=original_size, use_thresh=True,low_bounds=low_bounds,lastPass=lastPass)
                whichOne = "Thresh-{}-{}".format(low_bounds, lastPass)
                print("done with third with low thresh (shadowy pictures)")
                if results is None or len(results) == 0:
                    # and finally, color without target contour masked
                    #for pale/blurry images
                    print("done with fourth with high threshold for overexposed...")
                    results = get_quarter_results(imageToUse, target_contour, quarter_template_contour, look_for_shapes, origCellCount, isWhiteOrGray,original_size=original_size, use_thresh=True,low_bounds=125,lastPass=lastPass)
                    whichOne = "Thresh-160-{}".format(lastPass)
                    #do we need to keep going?
                    if results is None or len(results) == 0:
                        print("try low low")
                        low_bounds=75
                        results = get_quarter_results(imageToUse, target_contour, quarter_template_contour, look_for_shapes, origCellCount, isWhiteOrGray,original_size=original_size, use_thresh=True,low_bounds=low_bounds,lastPass=lastPass)
                        whichOne = "Thresh-{}-{}".format(low_bounds, lastPass)
                        if results is None or len(results) == 0:
                            print("trying no thresh 75")
                            results = get_quarter_results(imageToUse, target_contour, quarter_template_contour, look_for_shapes, origCellCount, isWhiteOrGray,original_size=original_size, use_thresh=False,low_bounds=75,lastPass=lastPass)
                            whichOne = "NoThresh-75-{}".format(lastPass)
                            if results is None or len(results) == 0:
                                print("trying color pics")
                                results = get_quarter_results(imageToUse, target_contour, quarter_template_contour, look_for_shapes, origCellCount, (not isWhiteOrGray),original_size=original_size, use_thresh=False,low_bounds=75,lastPass=lastPass)
                                whichOne = "Color-{}".format(lastPass)

        if results is not None and len(results) > 0:
            break
    print("sorting results...")

    results = sorted(results, key=lambda result: result[4])

    
    return results[0][0], results[0][1], results[0][2], results[0][3], whichOne

def offset_contour(contour, x, y):

    newX = x
    newY = y
    for points in contour:
        ndims = points.ndim
        if ndims > 1:
            for point in points:
                point[0] = point[0]+newX
                point[1] = point[1]+newY
        else:
            points[0] = points[0]+newX
            points[1] = points[1]+newY
    return contour

def get_filtered_quarter_contours_by_radius(scale_contours, img_area, quarter_template_contour, showOutput,lastPass=False):
    matches = []
    matching_contours = []
    okComp = []

    radiusMax = 82
    radiusMin = 17
    
    for i,match_contour in enumerate(scale_contours):
        try:
            (x,y),circleRadius = cv2.minEnclosingCircle(match_contour)
           
            circleRadius = int(circleRadius)
            matchPerimeter = cv2.arcLength(match_contour,False)
            matchArea = cv2.contourArea(match_contour)
            '''
            hull = cv2.convexHull(contour, returnPoints=Tre)
            hullArea = cv2.contourArea(hull) 
            '''
            if matchArea > 0:
                compactness = (matchPerimeter*matchPerimeter)/(4*matchArea*math.pi)
                circularity = 4*math.pi*(matchArea/(matchPerimeter*matchPerimeter))
                
                compactnessLimit = 1.5

                
                if False:
                    print("compactness: {}, circularity: {}".format(compactness, circularity))
                
                if lastPass:
                    #images where quarter is slightly cut off for heavy shadows or edges
                    compactnessLimit = 2.5
                    #for the zoomed out images
                    radiusMin=12
                if (0.5 <= circularity <= 1.5) or lastPass:
                    if circleRadius >= radiusMin and circleRadius <= radiusMax:
                        val = cv2.matchShapes(match_contour, quarter_template_contour, cv2.CONTOURS_MATCH_I3, 0.0)
                        score = compactness*val
                        matches.append([match_contour, circleRadius, (x, y), compactness,  val, score])
                        #for drawing while debugging
                        matching_contours.append(match_contour)
                        print("matching perimeter: {}, matching area: {}".format(matchPerimeter, matchArea))
                    if False:
                        print("ok circularity, radius is outside range")

                else:
                    if showOutput and (circleRadius >= radiusMin and circleRadius <= radiusMax):
                        if circleRadius < 50 and matchArea > 20:
                             
                            print("{}. radius: {}, circularity {}, compactness:{},  area:{}, perimeter: {}".format(i, circleRadius,circularity, compactness,matchArea, matchPerimeter))
                            okComp.append(match_contour)


        except Exception as e:
            print("blew up while filtereing: {}".format(e))
            continue


    return matches, matching_contours, okComp

def get_bounds(c, offset, nrows, ncols):
    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBottom = tuple(c[c[:, :, 1].argmax()][0])

    extLeft = extLeft[0]
    extRight = extRight[0]
    extTop = extTop[1]
    extBottom = extBottom[1]

    left = extLeft-offset if extLeft-offset >= 0 else extLeft
    right = extRight+offset if extRight+offset < ncols else extRight
    top = extTop-offset if extTop-offset >= 0 else extTop
    bottom = extBottom+offset if extBottom+offset < nrows else extBottom
    if left != extLeft:
        left_offset = extLeft-offset
    else:
        left_offset = extLeft
    
    if top != extTop:
        top_offset = extTop-offset
    else:
        top_offset = extTop

    return left, right, top, bottom, left_offset, top_offset

def get_quarter_dimensions(input_image, target_contour, quarter_template_contour, 
                            look_for_shapes, origCellCount, xOffset=0, yOffset=0, isWhiteOrGray=True, 
                            maskedInputImage=None,original_size=None, use_thresh=False, low_bounds=100,lastPass=False):

    ncols = len(input_image[0]) 
    nrows = len(input_image)
    img_area = nrows*ncols
    
    cx = 0
    cy = 0
    radius = 0
    circle_matches = []

    #1. use white or color

    filtered_contours, just_conts, okComp, circle_image = get_target_quarter_contours2(input_image.copy(), 
        quarter_template_contour, False, False, isWhiteOrGray, use_thresh, low_bounds=low_bounds, 
        target_contour=target_contour,original_size=None,lastPass=lastPass)

    if filtered_contours is not None and len(filtered_contours) > 0:
        if False:
            d = input_image.copy()
            cv2.drawContours(d, just_conts, -1,(150,150,150),3)
            utils.show_img("filtered contours", d)

        #sort by score
        filtered_contours = sorted(filtered_contours, key=lambda shape: shape[4], reverse=False)
        for x, fc in enumerate(filtered_contours):
            print("{}. fc val: score: {}, compact: {}, radius: {}".format(x, fc[4], fc[3], fc[1]))
        #if there are multiple contours, ditch ones that don't get circles...
        for target in filtered_contours:

            target_contour = target[0]
        
            if lastPass:
                quarterBuffer = 3
            else:
                quarterBuffer = 1
            left, right, top, bottom, left_offset, top_offset = get_bounds(target_contour, quarterBuffer, nrows, ncols)
            circle_image = input_image.copy()
            circle_image = circle_image[top:bottom,left:right]
            
            circle_image = cv2.cvtColor(circle_image, cv2.COLOR_BGR2GRAY)
            
            circles = cv2.HoughCircles(circle_image,cv2.HOUGH_GRADIENT,1,20,param1=60,param2=30,minRadius=20,maxRadius=70)
            
            if circles is not None and circles.any():
                break
            

        if circles is not None and circles.any():
            if False:
                
                for i in circles[0,:]:
                    # draw the outer circle
                    cv2.circle(circle_image,(i[0],i[1]),i[2],(0,200,0),1)
                
                utils.show_img("--->>>>>>circles", circle_image)
            
            contourWidth = int(abs(left-right)/2)
            print("all circles: {}".format(circles))
            circles = circles[0,:,:]
            circles = sorted(circles, key=lambda circle: circle[2], reverse=True)
            
            target_circle = circles[0]
            print('target circle: {}'.format(target_circle))
            #add a check for big diff in hough circles?
            radius = target_circle[2]
            (cx,cy) = (target_circle[0]+left_offset,target_circle[1]+top_offset)
            score = target[3]
            print("-----_>>>>>>>>>>>>radius: {}, contour width:{}".format(radius, contourWidth))

            return cx,cy,radius, [], score
        else:
            if lastPass:
                print("no hough circles, but desperate times...")
                radius = target[1]
                (cx, cy) = target[2]
                score = target[3]
                return cx, cy, radius, [], score
            else:
                print("no hough circles found...")
                return 0,0,0,[],10000000
            

    else:
        return 0,0,0,[],10000000

def atEdges(x,y, rows,cols):
    if int(x) < 90 or int(x) > cols-90:
        return True
    elif int(y) < 90 or int(y) > rows-90:
        return True
    else:
        return False

def trim_lobster_contour(target_contour, e_center, eAxes, eAngle):
    try:
    
        center = np.array(e_center)

        con = target_contour.copy()

        for pt in con:
            if pt[0] < xmin:
                pt[0] = xmin
            elif pt[0] > xmax:
                pt[0] = xmax


        return target_contour
    except Exception as e:
        return target_contour

def trim_abalone_contour(target_contour):
    #trimmed_contour, trimmed_ellipse = contour_utils.trim_abalone_contour(target_contour)
    try:
        cX, cY = utils.get_centroid(target_contour)
        ab_ellipse = cv2.fitEllipse(target_contour)
        size = ab_ellipse[1]
        width = int(size[1]/2)
        height = int(size[0])
        w = int(size[0])
        h = int(size[1])
        center = np.array([cX, cY])
        radius = width
        acon = np.squeeze(target_contour)
        rx,ry,rw,rh = cv2.boundingRect(target_contour)

        rectCX = rx+int(rw/2)
        rectCY = ry+int(rh/2)
        offsetVal = abs(cX-rectCX)

        if cX-rectCX > 0:

            #the bigger it is, the smaller the offset
            left_offset = min(abs(-offsetVal+70),20)
            right_offset = offsetVal*3
        else:
            
            left_offset = offsetVal*3
            right_offset = min(abs(-offsetVal+70),20)


        xmin = (center[0]-((int(w/2)+50)))-left_offset
        xmax = (center[0]+((int(w/2))+50))+right_offset
        for pt in acon:
            if pt[0] < xmin:
                pt[0] = xmin
            elif pt[0] > xmax:
                pt[0] = xmax

        if acon is not None:
            ab_contour = acon.copy()
        else:
            ab_contour = target_contour.copy()

        return ab_contour
    except Exception as e:
        return target_contour

def get_quarter_contour_and_center(quarter_contour):
    cX, cY = utils.get_centroid(quarter_contour)
   
    quarter_ellipse = cv2.fitEllipse(quarter_contour)

    size = quarter_ellipse[1]
    width = int(size[1]/2)
    height = int(size[0])
    w = int(size[0])
    h = int(size[1])
    center = np.array([cX, cY])
    radius = width-2
    qcon = np.squeeze(quarter_contour)

    xmin = center[0]-((int(w/2)))
    xmax = center[0]+((int(w/2)))
    for pt in qcon:
        if pt[0] < xmin:
            pt[0] = xmin
        elif pt[0] > xmax:
            pt[0] = xmax

    #mask = (qcon[:,0] - center[0])**2 + (qcon[:,1] - center[1])**2 < (radius**2)+2
    
    #contourWithinQuarterCircle = qcon[mask,:]
    return cX, cY, qcon, quarter_ellipse

def get_target_full_lobster_contour(input_image):
    
    white_or_gray = True
    target_contour = None
    use_opposite = False

    gray = ci.get_lobster_image(input_image.copy())

    #gray = utils.get_gray_image(input_image, white_or_gray, False)

    blur = cv2.GaussianBlur(gray, (5,5),0)
    if white_or_gray:
        lower_bound = 50
        upper_bound = 100
    else:
        lower_bound = 50
        upper_bound = 200

    blur2 = cv2.bilateralFilter(input_image.copy(),17,100,100)
    gray2 = cv2.cvtColor(blur2, cv2.COLOR_BGR2GRAY)
    ret2, thresh2 = cv2.threshold(gray, 127,255,0)
    thresh2[thresh2 > 0] = 255
    

    #utils.show_img("thresh2", thresh2)
    
    edged_img = cv2.Canny(blur, lower_bound, upper_bound,3) 
   
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,13))
    erode_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
    kernel = np.ones((3,3), np.uint8)
    if not white_or_gray:
        iters = 2
    else:
        iters=3
        
    edged_img = cv2.dilate(edged_img, kernel, iterations=iters)
    edged_img = cv2.morphologyEx(edged_img, cv2.MORPH_CLOSE, dilate_kernel)

    #check this - seems like dark on white needs a cleanup, color needs an thickening
    if not white_or_gray:
        edged_img = cv2.dilate(edged_img, kernel, iterations=1)
    else:
        edged_img = cv2.erode(edged_img, erode_kernel, iterations=1)


    #gray_denoised = cv2.cvtColor(edged_img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(edged_img.copy(), 127,255,0)
    if False:
        utils.show_img("threshold ", thresh)
    cnts = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    all_contours = cnts[1]
    largest = 0
    largestDex = 0

    all_contours = sorted(all_contours, key=lambda shape: cv2.contourArea(shape), reverse=True)
    '''
    for i, contour in enumerate(all_contours):
        try:
            hull = cv2.convexHull(contour)
            carea = cv2.contourArea(hull) 
            if carea > largest:
                largest = carea
                largestDex = i
        except Exception as e:
            continue
    '''
    target_contour = all_contours[0]

    #orig contours are returned for display/testing
    return target_contour, all_contours

def get_target_lobster_contour(input_image, lobster_template_contour, lower_percent_bounds, white_or_gray, use_opposite, center_offset):
    
    white_or_gray = True
    target_contour = None

    top_offset = 140
    left_offset = 230
    right_offset = 160
    bottom_offset = 140
    trimmed_image = input_image[top_offset:len(input_image) - bottom_offset,left_offset:len(input_image[0])-right_offset]

    gray = utils.get_gray_image(trimmed_image, white_or_gray, use_opposite)
    if use_opposite:
        white_or_gray = not white_or_gray
    blur = cv2.GaussianBlur(gray, (5,5),0)
    if white_or_gray:
        lower_bound = 50
        upper_bound = 100
    else:
        lower_bound = 50
        upper_bound = 200

    edged_img = cv2.Canny(blur, lower_bound, upper_bound,3) 

    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,13))
    erode_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
    kernel = np.ones((3,3), np.uint8)
    if not white_or_gray:
        iters = 2
    else:
        iters=3
        
    edged_img = cv2.dilate(edged_img, kernel, iterations=iters)
    edged_img = cv2.morphologyEx(edged_img, cv2.MORPH_CLOSE, dilate_kernel)

    #check this - seems like dark on white needs a cleanup, color needs an thickening
    if not white_or_gray:
        edged_img = cv2.dilate(edged_img, kernel, iterations=1)
    else:
        edged_img = cv2.erode(edged_img, erode_kernel, iterations=1)


    #gray_denoised = cv2.cvtColor(edged_img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(edged_img.copy(), 127,255,0)
    #utils.show_img("threshold ", thresh)
    cnts = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    all_contours = cnts[1]

    ncols = len(trimmed_image[0]) 
    nrows = len(trimmed_image)
    img_area = nrows*ncols
    minVal = 100000000
    dex = 0
    img_center = (int(ncols/2),int(nrows/2))

    largest = utils.get_largest_contours(all_contours,10)

    center_contours = []
    for i, contour in enumerate(all_contours):
        hull = cv2.convexHull(contour)
        centerIn = cv2.pointPolygonTest(hull,img_center,False) >= 0
        if centerIn:
            center_contours.append(contour)

    largest = utils.get_largest_contours(center_contours,1)

    target_contour = largest[0]
    #orig contours are returned for display/testing
    return target_contour, cnts, top_offset, left_offset

def get_lobster_contour(input_image, lobster_template_contour):
    white_or_gray = True
    target_contour, orig_contours, top_offset, left_offset = get_target_lobster_contour(input_image.copy(), lobster_template_contour, 0.02, white_or_gray, False, 150)
    if target_contour is None:
        target_contour, orig_contours, top_offset, left_offset = get_target_lobster_contour(input_image.copy(), lobster_template_contour, 0.005, white_or_gray, True, 300)

    ncols = len(input_image[0]) 
    nrows = len(input_image)
    img_area = nrows*ncols


    contours = np.array(target_contour)

    return contours, orig_contours, left_offset, top_offset

def get_histo_max(img):
    hist,bins = np.histogram(img.flatten(),256,[0,256])

    
    binmax = np.max(bins)

    return float(binmax)

def get_target_square_contours(input_image, square_template_contour, white_or_gray, lower_percent_bounds, check_for_square, use_actual_size, start_time,
                                try_bright=False):
    target_contour = None
    white_or_gray = True

    '''
    if not white_or_gray:
        print("its white or gray?")
        thresh_val = 30
        blur_window = 5
        first_pass = True
        is_ruler = True
        use_adaptive = False
        color_image, threshold_bw, color_img, mid_row = ci.get_image_with_color_mask(input_image, thresh_val, 
            blur_window, False, first_pass, is_ruler, use_adaptive)
        current_best = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
    else:
    '''

    print ('its white or gray')
    '''
    this approach didn't seem to work asget well
    denoised = cv2.fastNlMeansDenoisingColored(input_image,None,10,10,5,9)
    gray = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)
    '''
    if try_bright:
        gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
        max_val = get_histo_max(gray)
        print("max histo val is {}".format(max_val))
        alpha = 255.0/max_val
        print("alpha is {}".format(alpha))
        brightened = cv2.multiply(gray, alpha)
        #clahe = apply_clahe(input_image, max_contrast=12, grid_size=10)
        
        #blur = cv2.medianBlur(input_image,5)
        
        #equalized_image = equalize_hist(gray)
        bilateralGray = cv2.bilateralFilter(brightened, 11, 7, 5)
        bilateralThresh = 80
        
        _,current_best = cv2.threshold(bilateralGray,bilateralThresh,255,cv2.THRESH_BINARY)
        if False:
            utils.show_img("current best ", current_best)

        utils.print_time("denoising image finished", start_time)
    else:
        denoised = cv2.fastNlMeansDenoisingColored(input_image,None,10,10,5,9)
        current_best = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)
        
    if white_or_gray:
        lower_bound = 0
        upper_bound = 100
        thresh_lower = 70
        thresh_upper = 255
    else:
        lower_bound = 20
        upper_bound = 250
        thresh_lower = 127
        thresh_upper = 250

    scale_img = cv2.Canny(current_best, lower_bound, upper_bound,7) 
    
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(13,17))
    erode_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
    kernel = np.ones((3,3), np.uint8)

    if not white_or_gray:
        iters = 2
    else:
        iters=3
        
    edged_img = cv2.dilate(scale_img, kernel, iterations=iters)
    utils.print_time("dilate finished", start_time)
    edged_img = cv2.morphologyEx(edged_img, cv2.MORPH_CLOSE, dilate_kernel)
    utils.print_time("erode finished", start_time)
    #check this - seems like dark on white needs a cleanup, color needs an thickening
    if not white_or_gray:
        edged_img = cv2.dilate(edged_img, kernel, iterations=1)
    else:
        edged_img = cv2.erode(edged_img, erode_kernel, iterations=1)


    #gray_denoised = cv2.cvtColor(edged_img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(edged_img.copy(), 127,255,0)
    utils.print_time("threshold done", start_time)
    cnts = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    utils.print_time("contours finished", start_time)
    contours = cnts[1]

    ncols = len(input_image[0]) 
    nrows = len(input_image)
    img_area = nrows*ncols
    minVal = 100000000
    dex = 0
    tcontours = []


    for i, contour in enumerate(contours):
        try:
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)   
            contour_area = cv2.contourArea(contour)

            hull_perc = hull_area/img_area
            actual_perc = contour_area/img_area

            rotRect = cv2.minAreaRect(contour)
            width = rotRect[1][0]
            height = rotRect[1][1]
            if height == 0 or width is None or height is None:
                continue

            ratio = float(width)/float(height)
            if check_for_square and (ratio <= 0.75 or ratio >= 1.25):
                continue

            if use_actual_size:
                perc_target = actual_perc
            else:
                #on second pass through, use the hull percent
                perc_target = hull_perc

            if perc_target <= 0.4 and perc_target > lower_percent_bounds:
                print("hull_area: {}; percent: {}".format(hull_area, hull_perc))
                if(len(contour) == 0):
                    continue

                w,h = get_width_and_height(contour)
                #ditch cutting board borders around the outside
                if w < 0.5*ncols and h < 0.5*nrows:
                    
                    val = cv2.matchShapes(contour, square_template_contour, 2, 0.0)
                   
                    if val < minVal:
                        dex = i
                        minVal = val
                        tcontours.append(contour)
                        target_contour = contour
            
        except Exception as e:
            continue

    if False:
        draw = input_image.copy()
        cv2.drawContours(draw, [target_contour], -1, (0,255,255),4)
        utils.show_img("square contours", draw)

    if target_contour is None and not check_for_square:
        print("short circuiting...")
        return contours[0], tcontours

    utils.print_time("finished finding target square", start_time)

    return target_contour, tcontours

def get_square_contour(input_image, lobster_contour, square_template_contour, start_time):

    ncols = len(input_image[0]) 
    nrows = len(input_image)
    img_area = nrows*ncols
    img_area

    target_perc = 30*30/img_area

    cx = 0
    cy = 0
    radius = 0
    circle_matches = []
    #1. use white or color
    utils.print_time("j------------>>>>>>> first pass on square with contour time ", start_time)
    white_or_gray = utils.is_white_or_gray(input_image, False)
    square_contour, scale_contours = get_target_square_contours(input_image, square_template_contour, white_or_gray, 0.005, True, True, start_time)
    

    if square_contour is None or len(square_contour) == 0:
        utils.print_time("------------------------->>>> nothing on first pass, doing second", start_time)
        square_contour, scale_contours = get_target_square_contours(input_image, square_template_contour, white_or_gray,0.0005,True, False, start_time)

        if square_contour is None or len(square_contour) == 0:
            utils.print_time("=============>>>>>>>>>>>>  second failed, doing last one...", start_time)
            square_contour, scale_contours = get_target_square_contours(input_image, square_template_contour, white_or_gray,0.0005,True, False, start_time, try_bright=True)            

            if square_contour is None or len(square_contour) == 0:
                utils.print_time("=============>>>>>>>>>>>>  second failed, doing last one...", start_time)
                square_contour, scale_contours = get_target_square_contours(input_image, square_template_contour, white_or_gray,0.0001,False, False, start_time,try_bright=False)            
                        
    return square_contour, scale_contours


def get_target_finfish_contour(full_image, clipped_image, template_contour, is_square_ref_object=False,isWhiteOrGray=True, edge_of_mask=None):

    print("first pass...")
    target_contour, orig_contours = get_finfish_contour(full_image, clipped_image, template_contour, 10.0, isWhiteOrGray, use_opposite=False, is_square_ref_object=is_square_ref_object, edge_of_mask=edge_of_mask,canny_range=0.35,kernel_size=(7,5),erase_size=14,final_try=False)
    
    if target_contour is None:
        print("working on second pass...")
        target_contour, orig_contours = get_finfish_contour(full_image, clipped_image, template_contour, 0.5, isWhiteOrGray, use_opposite=False, is_square_ref_object=is_square_ref_object, edge_of_mask=edge_of_mask, canny_range=0.75,kernel_size=(11,7),erase_size=15,final_try=False)
        if target_contour is None:
            print("working on third pass...")
            target_contour, orig_contours = get_finfish_contour(full_image, clipped_image, template_contour, 0.05, isWhiteOrGray, use_opposite=False, is_square_ref_object=is_square_ref_object, edge_of_mask=edge_of_mask, canny_range=0.90,kernel_size=(11,9),erase_size=17,final_try=True)
 
    contours = np.array(target_contour)

    return contours, orig_contours

def mask_non_finfish(full_image, clipped_image):


    rows = len(clipped_image)
    cols = len(clipped_image[0])

    full_rows = len(full_image)
    full_cols = len(full_image[0])
    masks = []
    final_mask = np.zeros((full_rows,full_cols), np.uint8)
    clipped_mask = np.zeros((rows, cols), np.uint8)
    for i in range(int((rows/2))-30, int((rows/2))+30):
        for j in range(int((cols/2))-30, int((cols/2))+30):
            hlow = clipped_image[i][j][0]
            slow = clipped_image[i][j][1]
            vlow = clipped_image[i][j][2]
           
            lower_range = np.array([hlow-8, slow-4, vlow-4])
            upper_range = np.array([hlow+9, slow+8, vlow+8])
            
            mask = cv2.inRange(full_image, lower_range, upper_range)
            cmask = cv2.inRange(clipped_image, lower_range, upper_range)
            final_mask = cv2.bitwise_or(final_mask, mask)
            clipped_mask = cv2.bitwise_or(clipped_mask, cmask)
    

    #utils.show_img("masked", clipped_mask)
    return clipped_mask

#draw the biggest contour then erase the edge of the machine learning clipping mask...
def erase_edge_of_clipped_mask(current_contour, edge_of_mask, draw,erase_size):

    origMask = np.zeros(draw.shape[:2], dtype="uint8")

    cv2.fillPoly(origMask, [current_contour],255)
    cv2.drawContours(origMask, [edge_of_mask],0,0,erase_size)
 
    edge_area = cv2.contourArea(edge_of_mask)
    this_area = cv2.contourArea(current_contour)
    if this_area >= edge_area*0.95:
        print('same as edge')
        return None
    #rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    #threshed = cv2.morphologyEx(origMask, cv2.MORPH_CLOSE, rect_kernel)
    im2, target_shapes, hierarchy = cv2.findContours(origMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    target_shapes = sorted(target_shapes, key=lambda shape: cv2.contourArea(cv2.convexHull(shape,returnPoints=True)), reverse=True)
    
    #target_contour = np.concatenate(target_shapes[:1],0)
    if len(target_shapes) == 0:
        return None

    target_contour = target_shapes[0]
    #target_contour = np.concatenate(target_shapes,0)
    if False:
        cv2.drawContours(origMask, [target_contour], -1,(255,10,50),3)
        utils.show_img("contours: ", origMask)
    #utils.show_img("orig mask", origMask)
    return target_contour

def get_canny(input_image, sigma=0.4):
	img_median = np.median(input_image)
 
	lower = int(max(0, (1.0 - sigma) * img_median))
	upper = int(min(255, (1.0 + sigma) * img_median))
	edged = cv2.Canny(input_image, lower, upper)
 
	# return the edged image
	return edged

def get_image_stats(input_image, sigma=0.33):
    img_median = np.median(input_image)
    lower = int(max(0, (1.0 - sigma) * img_median))
    upper = int(min(255, (1.0 + sigma) * img_median))
    return img_median, lower, upper

def equalize_hist(clipped_image):
    equ = cv2.equalizeHist(clipped_image)
    return equ

def apply_clahe(clipped_image,max_contrast=8,grid_size=8):


    lab = cv2.cvtColor(clipped_image, cv2.COLOR_BGR2LAB)

    lab_planes = cv2.split(lab)


    clahe = cv2.createCLAHE(clipLimit=max_contrast,tileGridSize=(grid_size,grid_size))

    lab_planes[0] = clahe.apply(lab_planes[0])

    lab = cv2.merge(lab_planes)

    bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    #utils.show_all_imgs(["orig","clahe"], [clipped_image,bgr], rescale_width=800)
    return bgr


def get_finfish_contour(full_image, clipped_image, template_contour, lower_percent_bounds, white_or_gray, 
                            use_opposite=False, is_square_ref_object=False, edge_of_mask=None, canny_range=0.33,kernel_size=(7,5),erase_size=11,
                            final_try=False):
    
    target_contour = None
    clahe_image = None
    print("is white or gray: {}".format(white_or_gray))
    if use_opposite:
        white_or_gray = not white_or_gray

    
    #this was white or gray
    if white_or_gray or utils.is_dark_gray(clipped_image):
        lower_bound = 30
        upper_bound = 100
        is_dark = utils.is_dark_gray(clipped_image)
        if is_dark:
            #clahe_image = apply_clahe(clipped_image)
            gray = cv2.cvtColor(clipped_image, cv2.COLOR_BGR2GRAY)
            #simple brightening seems to work best...
            max_val = get_histo_max(gray)
            alpha = 255/max_val
            clahe_image = cv2.multiply(clipped_image, alpha)

    else:
        lower_bound = 10
        upper_bound = 220
        
    if not final_try:
    
        #some of the really light/blurry images fail complete with the canny call, using this instead...
        if clahe_image is not None:
            ret, thresh = cv2.threshold(clahe_image.copy(), 75,255,0)
        else:
            ret, thresh = cv2.threshold(clipped_image.copy(), 100,255,0)
        #edged_img = get_canny(value_layer,canny_range)
        edged_img = get_canny(thresh,canny_range)
    else:
        hsv_image = cv2.cvtColor(clipped_image, cv2.COLOR_BGR2HSV)
        value_layer = hsv_image[:,:,2]
        #denoised = cv2.fastNlMeansDenoisingColored(value_layer,None,7,21,13,10)
        #gray = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)
        edged_img = get_canny(value_layer,canny_range)
    
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,kernel_size)
    erode_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,7))
    kernel = np.ones((3,3), np.uint8)
    
    
    iters=1
    edged_img = cv2.dilate(edged_img, dilate_kernel, iterations=iters)
    if False:
        hsv_image = cv2.cvtColor(clipped_image, cv2.COLOR_BGR2HSV)
        value_layer = hsv_image[:,:,2]
        gray = cv2.cvtColor(clipped_image.copy(), cv2.COLOR_BGR2GRAY)
        cv2.imshow("hue", hsv_image[:,:,0])
        cv2.imshow("sat", hsv_image[:,:,1])
        cv2.imshow("val", hsv_image[:,:,2])
        cv2.imshow("gray", hsv_image[:,:,1])
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    '''
    edged_img = cv2.erode(edged_img, erode_kernel, iterations=1)
    if False:
        utils.show_img("eroded", edged_img)
    
    '''
    edged_img = cv2.morphologyEx(edged_img, cv2.MORPH_CLOSE, dilate_kernel)
    if False:
        utils.show_img("edged", edged_img)
    ret, thresh = cv2.threshold(edged_img.copy(), 127,255,0)

        
    cnts = cv2.findContours(edged_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
 
    if (cnts[1] is None or len(cnts[1]) == 0) and (cnts[0] is not None and len(cnts[0]) > 0):
        largest = sorted(cnts[0], key=lambda shape: cv2.contourArea(cv2.convexHull(shape,returnPoints=True)), reverse=True)
    else :
        largest = sorted(cnts[1], key=lambda shape: cv2.contourArea(cv2.convexHull(shape,returnPoints=True)), reverse=True)
    
    if False:
        try:
            cv2.drawContours(clipped_image, cnts[0], -1, (255,0,0),4)
        except Exception as e:
            print("nothing in 0")
        try:
            cv2.drawContours(clipped_image, cnts[1], -1, (0,0,255),4)
        except Exception:
            print("couldn't draw 1")
        try:
            cv2.drawContours(clipped_image, cnts[2], -1, (0,255,0),4)
        except Exception:
            print("coudln't draw 2")

        #cv2.drawContours(input_image, [largest], -1, (255,255,255),3)
        utils.show_img("cnts 0 is blue, 1 is red,2 is green ", clipped_image)
    

    if False:
        tst = clipped_image.copy()
        if largest is not None and len(largest) > 0:
            n=50
            for l in largest:

                if n < 300:
                    cv2.drawContours(tst, l, -1, (n,n,n),3)
                n=n+60
            utils.show_img("biggest contours", tst)


    ncols = len(clipped_image[0]) 
    nrows = len(clipped_image)
    img_area = nrows*ncols
    minVal = 100000000
    dex = 0
    
    if largest is not None and len(largest) > 0:
        
        for i, contour in enumerate(largest):
            current_contour = erase_edge_of_clipped_mask(contour, edge_of_mask, clipped_image.copy(),erase_size)
            if current_contour is None or len(current_contour) < 3:
                continue
            area = cv2.contourArea(current_contour)
            area_perc = (float(area)/float(img_area))*100.0
            #in case there is a big contour at the edge with dangly bits, so the erased contour isn't empty, but it's tiny
            if area_perc > lower_percent_bounds:
                print("{} has area perc {}".format(i, area_perc))
                target_contour = current_contour
                break

    else:
        target_contour = None

    if False:
        draw = clipped_image.copy()
        cv2.drawContours(draw, [target_contour], -1, (255,255,0),12)
        utils.show_img("target-->>>>", draw)

    #complete fail, use the ml outline
    if final_try and target_contour is None:
        print("failed, using machine learning edge...")
        target_contour = largest[0]
    #orig contours are returned for display/testing

    print("done...")
    if cnts is None or len(cnts) < 1:
        orig_contours = []
    else:
        orig_contours = cnts[1]
    return target_contour, orig_contours