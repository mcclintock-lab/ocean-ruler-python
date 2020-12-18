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
        #most images will probably be white or light gray
        edged_img = cv2.Canny(blur, lower_bound, upper_bound,9)
    else:
        lower_bound = 0
        upper_bound = 99
        
        hsv = cv2.cvtColor(input_image, cv2.COLOR_BGR2HSV)
       
        hue_layer = hsv[:,:,0]

        sat_layer = hsv[:,:,1]
        value_layer = hsv[:,:,2]

        
        if utils.is_dark_gray(input_image):
            #for really dark images, don't apply a threshold
            edged_img = cv2.Canny(blur, lower_bound, upper_bound,7) 
        else:

            hsv = cv2.cvtColor(input_image, cv2.COLOR_BGR2HSV)

            hue_layer = hsv[:,:,0]
            ret, thresh = cv2.threshold(hue_layer, 30, 100, 0)
            edged_img = cv2.Canny(thresh, lower_bound, upper_bound,7) 


    #dilate and erode thicken/thin the contour to get rid of disconnected contours
    #and merge together broken up contours that are close together
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

    
    ncols = len(input_image[0]) 
    nrows = len(input_image)
    img_area = nrows*ncols
    minVal = 100000000
    dex = 0
    #find the largest contour with a matching shape size
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
    """ Check the 'squareness' of the contour. Simplify the polygon, check to see if
        width and height are close together and it has 4 corners

    """
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


def get_target_contour(clipped_image,original_image, template_contour, is_square_ref_object, is_abalone, isWhiteOrGray, fishery_type):
    """ Getting the target contour if tthere is no masking image


    """
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


def get_quarter_image(input_image, use_opposite, isWhiteOrGray, use_thresh, low_bounds=100):
    """ Do image preprocessing for quarters. Flags allow for multiple passes through 
        with different criteria (if early passees don't detect a good quarter, keep looking)


    """
    if not isWhiteOrGray:
        hsv = cv2.cvtColor(input_image.copy(), cv2.COLOR_BGR2HSV)
        hue_layer = hsv[:,:,0]
        ret, thresh = cv2.threshold(hue_layer, 30, 100, 0)
        scale_img = cv2.Canny(thresh, 0, 99,7) 
       
    else:
        if use_thresh:
            #note: calculating image stats doesnt help. problem is if quarter is in shadow, a high low-bounds (above 60) won't see it            
            grayIn = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(grayIn, (5,5),0)

            ret, gray = cv2.threshold(blur, low_bounds,255,0)
            gray[gray < 254 ] = 0 

            scale_img = gray
        else:
            #changed this on 3/23/20 - seems like bilateral gray works best on white background...
            blur = cv2.medianBlur(input_image,3)
            gray2 = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
            bilateralGray = cv2.bilateralFilter(gray2, 9, 5, 5)
            scale_img = get_canny(bilateralGray)

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

    check_roundness = not lastPass
    matches, matching_contours, okComp = get_filtered_quarter_contours_by_radius(scale_contours, img_area, 
                                                                                 quarter_template_contour,
                                                                                 lastPass=lastPass)
    
    return matches, matching_contours, okComp, circle_image


def is_point_in_circle(point, circle):
    """ Check contours to see if the center is inside the contour hull. Used to reject bogus shapes


    """
    circleCenter = (circle[0], circle[1])
    radius = circle[2]
    inX = (point[0] <= circleCenter[0]+radius) and (point[0] >= circleCenter[0]-radius)
    inY = (point[1] <= circleCenter[1]+radius) and (point[1] >= circleCenter[1]-radius)
    return inX and inY


    return matches
                        
def get_quarter_results(clippedImages, target_contour, quarter_template_contour, look_for_shapes, 
                        origCellCount, isWhiteOrGray,original_size=None, use_thresh=False,low_bounds=100,lastPass=False):
    results = []
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
    """ Try multiple passes through to get the best quarter image

    """
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
                
                if results is None or len(results) == 0:
                    # and finally, color without target contour masked
                    #for pale/blurry images
                    
                    results = get_quarter_results(imageToUse, target_contour, quarter_template_contour, look_for_shapes, origCellCount, isWhiteOrGray,original_size=original_size, use_thresh=True,low_bounds=125,lastPass=lastPass)
                    whichOne = "Thresh-160-{}".format(lastPass)
                    #do we need to keep going?
                    if results is None or len(results) == 0:
                        
                        low_bounds=75
                        results = get_quarter_results(imageToUse, target_contour, quarter_template_contour, look_for_shapes, origCellCount, isWhiteOrGray,original_size=original_size, use_thresh=True,low_bounds=low_bounds,lastPass=lastPass)
                        whichOne = "Thresh-{}-{}".format(low_bounds, lastPass)
                        if results is None or len(results) == 0:
                           
                            results = get_quarter_results(imageToUse, target_contour, quarter_template_contour, look_for_shapes, origCellCount, isWhiteOrGray,original_size=original_size, use_thresh=False,low_bounds=75,lastPass=lastPass)
                            whichOne = "NoThresh-75-{}".format(lastPass)
                            if results is None or len(results) == 0:
                               
                                results = get_quarter_results(imageToUse, target_contour, quarter_template_contour, look_for_shapes, origCellCount, (not isWhiteOrGray),original_size=original_size, use_thresh=False,low_bounds=75,lastPass=lastPass)
                                whichOne = "Color-{}".format(lastPass)

        if results is not None and len(results) > 0:
            break

    results = sorted(results, key=lambda result: result[4])
    return results[0][0], results[0][1], results[0][2], results[0][3], whichOne

def offset_contour(contour, x, y):
    """ Shift the contour based on a given offset (for clipped images)

    """
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

def get_filtered_quarter_contours_by_radius(scale_contours, img_area, quarter_template_contour,lastPass=False):
    """ Filter quarter contours based on a bunch of heuristics, getting progressively
        looser as it goes...

    """
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

            if matchArea > 0:
                compactness = (matchPerimeter*matchPerimeter)/(4*matchArea*math.pi)
                circularity = 4*math.pi*(matchArea/(matchPerimeter*matchPerimeter))
                
                compactnessLimit = 1.5

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
                    if (circleRadius >= radiusMin and circleRadius <= radiusMax):
                        if circleRadius < 50 and matchArea > 20:
                            okComp.append(match_contour)


        except Exception as e:
            print("blew up while filtereing: {}".format(e))
            continue


    return matches, matching_contours, okComp

def get_bounds(c, offset, nrows, ncols):
    """ Calculate the contour bound points


    """
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

    """ Fit circles to the quarter contours that were found, so that the final contour is a more 
        quarter-y shape. Helps take care of shading/lighting edge problems

    """
    ncols = len(input_image[0]) 
    nrows = len(input_image)
    img_area = nrows*ncols
    
    cx = 0
    cy = 0
    radius = 0
    circle_matches = []

    filtered_contours, just_conts, okComp, circle_image = get_target_quarter_contours2(input_image.copy(), 
        quarter_template_contour, False, False, isWhiteOrGray, use_thresh, low_bounds=low_bounds, 
        target_contour=target_contour,original_size=None,lastPass=lastPass)

    if filtered_contours is not None and len(filtered_contours) > 0:
        #sort by score
        filtered_contours = sorted(filtered_contours, key=lambda shape: shape[4], reverse=False)

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

            contourWidth = int(abs(left-right)/2)
            circles = circles[0,:,:]
            circles = sorted(circles, key=lambda circle: circle[2], reverse=True)
            
            target_circle = circles[0]
            
            #add a check for big diff in hough circles?
            radius = target_circle[2]
            (cx,cy) = (target_circle[0]+left_offset,target_circle[1]+top_offset)
            score = target[3]
            return cx,cy,radius, [], score
        else:
            if lastPass:
                radius = target[1]
                (cx, cy) = target[2]
                score = target[3]
                return cx, cy, radius, [], score
            else:
                print("no hough circles found...")
                return 0,0,0,[],10000000
            

    else:
        return 0,0,0,[],10000000

def trim_abalone_contour(target_contour):
    """ To clean up abalone edges, fit an ellipse (abalone shape) and shop off the contour edges
    """
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

def get_target_full_lobster_contour(input_image):
    """ Get the target contour for the entire lobster. Just gets largest, doesn't try to match shape
        because they're crazy shapes. Used to alignment lines


    """
    white_or_gray = True
    target_contour = None
    use_opposite = False

    gray = ci.get_lobster_image(input_image.copy())
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

    cnts = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    all_contours = cnts[1]
    largest = 0
    largestDex = 0

    all_contours = sorted(all_contours, key=lambda shape: cv2.contourArea(shape), reverse=True)
    target_contour = all_contours[0]

    #orig contours are returned for display/testing
    return target_contour, all_contours

def get_target_lobster_contour(input_image, lobster_template_contour, lower_percent_bounds, white_or_gray, use_opposite, center_offset):
    """ Get the lobster contour for just the carapace

    """
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


    ret, thresh = cv2.threshold(edged_img.copy(), 127,255,0)
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
    """ Helper for getting max histo value of image

    """
    hist,bins = np.histogram(img.flatten(),256,[0,256])
    binmax = np.max(bins)

    return float(binmax)

def get_target_square_contours(input_image, square_template_contour, white_or_gray, lower_percent_bounds, check_for_square, use_actual_size, start_time,
                                try_bright=False):
    target_contour = None
    white_or_gray = True

    if try_bright:
        gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
        max_val = get_histo_max(gray)
        alpha = 255.0/max_val
        brightened = cv2.multiply(gray, alpha)
        #tried clahe processing - it worked well for some images, not others...
        bilateralGray = cv2.bilateralFilter(brightened, 11, 7, 5)
        bilateralThresh = 80
        _,current_best = cv2.threshold(bilateralGray,bilateralThresh,255,cv2.THRESH_BINARY)
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
   
    edged_img = cv2.morphologyEx(edged_img, cv2.MORPH_CLOSE, dilate_kernel)
    
    if not white_or_gray:
        edged_img = cv2.dilate(edged_img, kernel, iterations=1)
    else:
        edged_img = cv2.erode(edged_img, erode_kernel, iterations=1)


    ret, thresh = cv2.threshold(edged_img.copy(), 127,255,0)
    cnts = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
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

    if target_contour is None and not check_for_square:
        return contours[0], tcontours

    return target_contour, tcontours

def get_square_contour(input_image, lobster_contour, square_template_contour, start_time):
    """ Multi-pass attempt to find good square contours

    """
    ncols = len(input_image[0]) 
    nrows = len(input_image)
    img_area = nrows*ncols
    img_area

    target_perc = 30*30/img_area

    cx = 0
    cy = 0
    radius = 0
    circle_matches = []

    white_or_gray = utils.is_white_or_gray(input_image, False)
    square_contour, scale_contours = get_target_square_contours(input_image, square_template_contour, white_or_gray, 0.005, True, True, start_time)
    
    if square_contour is None or len(square_contour) == 0:
        square_contour, scale_contours = get_target_square_contours(input_image, square_template_contour, white_or_gray,0.0005,True, False, start_time)
        if square_contour is None or len(square_contour) == 0:
            square_contour, scale_contours = get_target_square_contours(input_image, square_template_contour, white_or_gray,0.0005,True, False, start_time, try_bright=True)            
            if square_contour is None or len(square_contour) == 0:
                square_contour, scale_contours = get_target_square_contours(input_image, square_template_contour, white_or_gray,0.0001,False, False, start_time,try_bright=False)            
                        
    return square_contour, scale_contours


def get_target_finfish_contour(full_image, clipped_image, template_contour, is_square_ref_object=False,isWhiteOrGray=True, edge_of_mask=None):

    target_contour, orig_contours = get_finfish_contour(full_image, clipped_image, template_contour, 10.0, isWhiteOrGray, use_opposite=False, is_square_ref_object=is_square_ref_object, edge_of_mask=edge_of_mask,canny_range=0.35,kernel_size=(7,5),erase_size=14,final_try=False)
    if target_contour is None:

        target_contour, orig_contours = get_finfish_contour(full_image, clipped_image, template_contour, 0.5, isWhiteOrGray, use_opposite=False, is_square_ref_object=is_square_ref_object, edge_of_mask=edge_of_mask, canny_range=0.75,kernel_size=(11,7),erase_size=15,final_try=False)
        if target_contour is None:
            target_contour, orig_contours = get_finfish_contour(full_image, clipped_image, template_contour, 0.05, isWhiteOrGray, use_opposite=False, is_square_ref_object=is_square_ref_object, edge_of_mask=edge_of_mask, canny_range=0.90,kernel_size=(11,9),erase_size=17,final_try=True)
 
    contours = np.array(target_contour)

    return contours, orig_contours



def erase_edge_of_clipped_mask(current_contour, edge_of_mask, draw,erase_size):
    """ Draw the biggest contour then erase the edge of the machine learning clipping mask...

    """
    origMask = np.zeros(draw.shape[:2], dtype="uint8")

    cv2.fillPoly(origMask, [current_contour],255)
    cv2.drawContours(origMask, [edge_of_mask],0,0,erase_size)
 
    edge_area = cv2.contourArea(edge_of_mask)
    this_area = cv2.contourArea(current_contour)
    if this_area >= edge_area*0.95:
        return None

    im2, target_shapes, hierarchy = cv2.findContours(origMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    target_shapes = sorted(target_shapes, key=lambda shape: cv2.contourArea(cv2.convexHull(shape,returnPoints=True)), reverse=True)

    if len(target_shapes) == 0:
        return None

    target_contour = target_shapes[0]

    return target_contour


def get_canny(input_image, sigma=0.4):
    img_median = np.median(input_image)
    lower = int(max(0, (1.0 - sigma) * img_median))
    upper = int(min(255, (1.0 + sigma) * img_median))
    edged = cv2.Canny(input_image, lower, upper)
    return edged

def get_finfish_contour(full_image, clipped_image, template_contour, lower_percent_bounds, white_or_gray, 
                            use_opposite=False, is_square_ref_object=False, edge_of_mask=None, canny_range=0.33,kernel_size=(7,5),erase_size=11,
                            final_try=False):
    """ Generate the finfish contour.

    """
    target_contour = None
    clahe_image = None
   
    if use_opposite:
        white_or_gray = not white_or_gray

    
    #this was white or gray
    if white_or_gray or utils.is_dark_gray(clipped_image):
        lower_bound = 30
        upper_bound = 100
        is_dark = utils.is_dark_gray(clipped_image)
        if is_dark:
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
            ret, thresh = cv2.threshold(clahe_image.copy(), 85,255,0)
        else:
            
            ret, thresh = cv2.threshold(clipped_image.copy(), 100,255,0)
        edged_img = get_canny(thresh,canny_range)
    else:
        hsv_image = cv2.cvtColor(clipped_image, cv2.COLOR_BGR2HSV)
        value_layer = hsv_image[:,:,2]
        edged_img = get_canny(value_layer,canny_range)
    
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,kernel_size)
    erode_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,7))
    kernel = np.ones((3,3), np.uint8)
        
    iters=1
    edged_img = cv2.dilate(edged_img, dilate_kernel, iterations=iters)


    edged_img = cv2.morphologyEx(edged_img, cv2.MORPH_CLOSE, dilate_kernel)
    ret, thresh = cv2.threshold(edged_img.copy(), 127,255,0)

    cnts = cv2.findContours(edged_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
 
    if (cnts[1] is None or len(cnts[1]) == 0) and (cnts[0] is not None and len(cnts[0]) > 0):
        largest = sorted(cnts[0], key=lambda shape: cv2.contourArea(cv2.convexHull(shape,returnPoints=True)), reverse=True)
    else :
        largest = sorted(cnts[1], key=lambda shape: cv2.contourArea(cv2.convexHull(shape,returnPoints=True)), reverse=True)


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
                target_contour = current_contour
                break
    else:
        target_contour = None

    #complete fail, use the ml outline
    if final_try and target_contour is None:
        target_contour = largest[0]

    #orig contours are returned for display/testing
    if cnts is None or len(cnts) < 1:
        orig_contours = []
    else:
        orig_contours = cnts[1]
    return target_contour, orig_contours