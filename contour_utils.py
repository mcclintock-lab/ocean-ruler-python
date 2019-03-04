import cv2
import time
import utils
import numpy as np
import color_images as ci
import math
import constants
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
    else:
        lower_bound = 10
        upper_bound = 220
        
    '''
    if constants.isScallop(fishery_type):
        g = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(g, 100, 255, 0)
        edged_img = cv2.Canny(thresh, lower_bound, upper_bound,7) 
    else:
    '''
    edged_img = cv2.Canny(blur, lower_bound, upper_bound,7) 

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
        largest = utils.get_largest_edges(cnts[0])
    else :
        largest = utils.get_largest_edges(cnts[1])

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

def get_target_contour(input_image, template_contour, is_square_ref_object, is_abalone, isWhiteOrGray, fishery_type):

    lower_perc_bounds = 0.1
    if(constants.isScallop(fishery_type)):
        #note: this is here because i'm temporarily ignoring the ML for scallops, since its chopping tails off...
        lower_perc_bounds = 0.05
    target_contour, orig_contours = get_target_oval_contour(input_image.copy(), template_contour, lower_perc_bounds, isWhiteOrGray, False, is_square_ref_object, fishery_type)
    if target_contour is None:
        if(constants.isScallop(fishery_type)):
            lower_perc_bounds = 0.01
        target_contour, orig_contours = get_target_oval_contour(input_image.copy(), template_contour, lower_perc_bounds, isWhiteOrGray, True, is_square_ref_object, fishery_type)

    ncols = len(input_image[0]) 
    nrows = len(input_image)
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

def get_quarter_image(input_image, use_opposite, isWhiteOrGray):

    if not isWhiteOrGray and not use_opposite:
        
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

    else:

        denoised = cv2.fastNlMeansDenoisingColored(input_image,None,10,10,5,9)
        gray = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)

    if isWhiteOrGray or utils.is_dark_gray(input_image):
        lower_bound = 0
        upper_bound = 100
        thresh_lower = 70
        thresh_upper = 255
    else:
        lower_bound = 20
        upper_bound = 250
        thresh_lower = 127
        thresh_upper = 250
        
    scale_img = cv2.Canny(gray, lower_bound, upper_bound,7) 

    return scale_img, gray

def get_target_quarter_contours(input_image, use_opposite, too_close_to_abalone=False, isWhiteOrGray=True):
    scale_img, gray = get_quarter_image(input_image, use_opposite, isWhiteOrGray)
    kernel = np.ones((5,3), np.uint8)
    if not too_close_to_abalone:
        scale_img = cv2.dilate(scale_img, kernel, iterations=1)
    else:
        scale_img = cv2.erode(scale_img, kernel, iterations=2)

    ret, thresh = cv2.threshold(scale_img.copy(), 127,255,0)
    scale_cnts = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    scale_contours = np.array(scale_cnts[1])

    if False:
        cv2.drawContours(input_image, scale_contours, -1, (0,255,255),4)
        utils.show_img("quarter threshold", input_image)

    matches = []
    cnt = 0
    circle_img = gray.copy()
    circles = cv2.HoughCircles(circle_img,cv2.HOUGH_GRADIENT,1,20,param1=50,param2=30,minRadius=15,maxRadius=80)
    if False:
        circles = np.uint16(np.around(circles))
        for i in circles[0,:]:
            # draw the outer circle
            cv2.circle(circle_img,(i[0],i[1]),i[2],(0,255,0),2)
            # draw the center of the circle
            cv2.circle(circle_img,(i[0],i[1]),2,(0,0,255),3)
        utils.show_img("circles", circle_img)
    return circles, scale_contours

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

def get_quarter_contour_info(target_quarter_contour):
    cx, cy, trimmed_quarter_contour, quarter_ellipse = get_quarter_contour_and_center(target_quarter_contour)
    size = quarter_ellipse[1]
    width = int(size[1]/2)-3
    height = int(size[0]/2)-3
    radius = min(width, height)
    return cx, cy, radius

def get_circle_info(circle):
    cx = circle[0]
    cy = circle[1]
    radius = circle[2]
    return cx, cy, radius

def get_filtered_quarter_contours(scale_contours, target_contour, img_area, check_roundness):
    matches = []
    for i, scontour in enumerate(scale_contours):
        try:
            carea = cv2.contourArea(scontour)
            hull = cv2.convexHull(scontour,returnPoints=True)
            hullArea = cv2.contourArea(hull)
            perc = hullArea/img_area

            if perc <= 0.06:
                #if not utils.is_contour_enclosed(scontour, target_contour, False, not check_roundness):
                if check_roundness:
                    if utils.is_really_round(scontour):
                        matches.append(scontour)

                else:
                    matches.append(scontour)

        except Exception as e:
            continue

    return matches

def get_quarter_results(clippedImages, target_contour, quarter_template_contour, look_for_shapes, origCellCount, isWhiteOrGray):
    results = []
    for i, ciData in enumerate(clippedImages):
        ci = ciData[0]
        xOffset = ciData[1]
        yOffset = ciData[2]

        refObjectCenterX, refObjectCenterY, refRadius, matches, score = get_quarter_dimensions(ci, target_contour, 
                                                                 quarter_template_contour, look_for_shapes, origCellCount, isWhiteOrGray)

        if refRadius > 0:
            results.append([refObjectCenterX+xOffset, refObjectCenterY+yOffset, refRadius, matches, score])
    return results
def get_best_quarter_dimensions(clippedImages, maskedInputImages, target_contour, quarter_template_contour, look_for_shapes, origCellCount, isWhiteOrGray):
    results = []
    #first passed, quarter images clipped with target contour masked
    results = get_quarter_results(maskedInputImages, target_contour, quarter_template_contour, look_for_shapes, origCellCount, isWhiteOrGray)
    if len(results) == 0:
        #second, without the target contour masked out
        results = get_quarter_results(clippedImages, target_contour, quarter_template_contour, look_for_shapes, origCellCount, isWhiteOrGray)
        if len(results) == 0:
            #no good, try color with target contour masked
            results = get_quarter_results(maskedInputImages, target_contour, quarter_template_contour, look_for_shapes, origCellCount, not isWhiteOrGray)
            if len(results) == 0:
                # and finally, color without target contour masked
                results = get_quarter_results(clippedImages, target_contour, quarter_template_contour, look_for_shapes, origCellCount, not isWhiteOrGray)
                #do we need to keep going?
                '''
                if len(circle_matches) == 0:
                        #2. use the opposite of white or color
                        circles, scale_contours = get_target_quarter_contours(input_image, True, False, isWhiteOrGray)
                        matches = get_filtered_quarter_contours(scale_contours, abalone_contour, img_area, True)
                        circle_matches = get_matches(circles, matches)
                        if len(circle_matches) == 0:
                            #3. try the original but with check roundness turned off, so no rebuild target contours
                            circles, scale_contours = get_target_quarter_contours(input_image, False, True, isWhiteOrGray)
                            matches = get_filtered_quarter_contours(scale_contours, abalone_contour, img_area, False)
                            circle_matches = get_matches(circles, matches)
                            
                            if len(circle_matches) == 0:
                                #4. finally, try it with original but with a small area
                                circles, scale_contours = get_target_quarter_contours(input_image, True, True, isWhiteOrGray)
                                matches = get_filtered_quarter_contours(scale_contours, abalone_contour, img_area, True)
                                circle_matches = get_matches(circles, matches)
                    '''

    results = sorted(results, key=lambda result: result[4])

    
    return results[0][0], results[0][1], results[0][2], results[0][3]

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


def get_quarter_dimensions(input_image, abalone_contour, quarter_template_contour, look_for_shapes, origCellCount, xOffset=0, yOffset=0, isWhiteOrGray=True, maskedInputImage=None):

    ncols = len(input_image[0]) 
    nrows = len(input_image)
    img_area = nrows*ncols

    cx = 0
    cy = 0
    radius = 0
    circle_matches = []
    #1. use white or color
    circles, scale_contours = get_target_quarter_contours(input_image, False, False, isWhiteOrGray)
    matches = get_filtered_quarter_contours(scale_contours, abalone_contour, img_area, True)

    circle_matches = get_matches(circles, matches)
    if False:
        #cv2.drawContours(input_image, matches, -1, (0,0,255),4)
        cv2.drawContours(input_image, scale_contours, -1, (255,255,0),6)
        #cv2.drawContours(input_image, [abalone_contour], 0, (0,255,0),4)
        
        utils.show_img("quarter contours", input_image)


    '''
    if len(circle_matches) == 0:
        #2. use the opposite of white or color
        circles, scale_contours = get_target_quarter_contours(input_image, True, False, isWhiteOrGray)
        matches = get_filtered_quarter_contours(scale_contours, abalone_contour, img_area, True)
        circle_matches = get_matches(circles, matches)
        if len(circle_matches) == 0:
            #3. try the original but with check roundness turned off, so no rebuild target contours
            circles, scale_contours = get_target_quarter_contours(input_image, False, True, isWhiteOrGray)
            matches = get_filtered_quarter_contours(scale_contours, abalone_contour, img_area, False)
            circle_matches = get_matches(circles, matches)
            
            if len(circle_matches) == 0:
                #4. finally, try it with original but with a small area
                circles, scale_contours = get_target_quarter_contours(input_image, True, True, isWhiteOrGray)
                matches = get_filtered_quarter_contours(scale_contours, abalone_contour, img_area, True)
                circle_matches = get_matches(circles, matches)
    '''

    dex = 0
    minVal = score = 1000000000
    rows = len(input_image)
    cols = len(input_image[0])
    matched = False
    if(len(circle_matches)) >= 1:
        compactness_values = []
        for i, match_data in enumerate(circle_matches):
            circle = match_data[0]
            match_contour = match_data[1]
            match_hull = match_data[2]

            (x,y),radius = cv2.minEnclosingCircle(match_contour)


            atEdge = atEdges(x,y,rows,cols)
            circleRadius = circle[2]

            matchPerimeter = cv2.arcLength(match_contour,True)
            matchArea = cv2.contourArea(match_contour)
            compactness = (matchPerimeter*matchPerimeter)/(4*matchArea*math.pi)
            cX, cY = utils.get_centroid(match_contour)
            compactness_values.append([radius, compactness, (cX, cY),10000000])

            if False:
                cv2.drawContours(input_image, [match_contour], 0, (0,0,255),5)
                utils.show_img("hull and match "+str(i), input_image)

            #for small images, set the radius max lower. for example:
            #feb_2017/IMG_3.36_36.JPG, which is 480x640
            if origCellCount < 1228800:
                radiusMax = 75
                radiusMin = 30
            else:
                radiusMax = 60
                radiusMin = 22

            
            #its outside the key range, so set value to big
            if radius < radiusMin or radius > radiusMax:
                compactness_values[i] = [radius, 10000000, (cX, cY),10000000]

            #make sure the radius is within a certain size and not at the edges - gets rid of little areas/squiggles
            if radius < 24 or radius > radiusMax or compactness > 1.5:
                continue
        
            val = cv2.matchShapes(match_contour, quarter_template_contour, cv2.CONTOURS_MATCH_I3, 0.0)

            hull = cv2.convexHull(match_contour,returnPoints = False)
            #update the compactness values to have new matching value
           
            compactness_values[i] = [radius, compactness, (cX, cY),val]
            
            modVal = val*compactness
            
            if modVal < minVal:
                dex = i
                minVal = modVal
                matched=True

            #tcx, tcy, tradius = get_circle_info(circle_matches[i][0])
        #cx, cy, radius = get_quarter_contour_info(circle_matches[dex][1])
        if not matched:
            #if none of them meet criteria (bad compactness, e.g.),
            #loosen and try again
            for k,vals in enumerate(compactness_values):
                r = vals[0]
                c = vals[1]
                if r > 20 and r < 70 and c < 2.2:
                    dex = k
            
        cx, cy, radius = get_circle_info(circle_matches[dex][0])
        
        orig_radius = compactness_values[dex][0]
        
        #the circles don't match, and the hough circle seems wrong...
        if(orig_radius > 22 and radius > 25):
            #xperiment - try the circle not the contour
            #radius = orig_radius
            #cx = compactness_values[dex][2][0]
            #cy = compactness_values[dex][2][1]
            target_circle = circle_matches[dex][0]
            target_quarter_contour = circle_matches[dex][1]
           
            cval = compactness_values[dex][1]
            vval = compactness_values[dex][3]
            score = cval*vval
        else:
            return 0,0,0,[],10000000

    else:
        return 0,0,0,[],10000000

    if target_quarter_contour is not None:
        #cv2.drawContours(input_image, [target_quarter_contour], -1, (255,0,0),4)
        
        #draw the circle 
        cv2.circle(input_image,(target_circle[0],target_circle[1]),target_circle[2],(0,255,0),2)
        ellipse = cv2.fitEllipse(target_quarter_contour)
        eX = ellipse[1][0]
        eY = ellipse[1][1]
        eDiff = abs(eX*2-eY*2)

    return cx, cy, radius, matches, score*eDiff

def atEdges(x,y, rows,cols):
    if int(x) < 90 or int(x) > cols-90:
        return True
    elif int(y) < 90 or int(y) > rows-90:
        return True
    else:
        return False

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
    for i, contour in enumerate(all_contours):
        try:
            hull = cv2.convexHull(contour)
            carea = cv2.contourArea(hull) 
            if carea > largest:
                largest = carea
                largestDex = i
        except Exception as e:
            continue

    target_contour = all_contours[largestDex]

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

def get_square_image(input_image, use_opposite):
    white_or_gray = utils.is_white_or_gray(input_image, False)
    if not white_or_gray and not use_opposite:
        thresh_val = 30
        blur_window = 5
        first_pass = True
        is_ruler = True
        use_adaptive = False


        color_image, threshold_bw, color_img, mid_row = ci.get_image_with_color_mask(input_image, thresh_val, 
            blur_window, False, first_pass, is_ruler, use_adaptive)
        gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
    else:
        denoised = cv2.fastNlMeansDenoisingColored(input_image,None,10,10,5,9)
        gray = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)

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
        

    scale_img = cv2.Canny(gray, lower_bound, upper_bound,7) 
    return scale_img, gray

def do_corner_detection(input_image, gray):
    
    
    #gray = np.float32(input_image)
    dst = cv2.cornerHarris(gray,2,3,0.04)

    #result is dilated for marking the corners, not important
    dst = cv2.dilate(dst,None)

    # Threshold for an optimal value, it may vary depending on the image.
    input_image[dst>0.01*dst.max()]=[0,0,255]


    cv2.imshow('dst',input_image)
    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()

def do_square_detection(input_image,contours):
    square_contours = []
    rect_contours = []
    for i, contour in enumerate(contours):

        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)
        if len(approx) == 4:
            # compute the bounding box of the contour and use the
            # bounding box to compute the aspect ratio
            (x, y, w, h) = cv2.boundingRect(approx)
            ar = w / float(h)
 
            # a square will have an aspect ratio that is approximately
            # equal to one, otherwise, the shape is a rectangle
            if ar >= 0.95 and ar <= 1.05:
                square_contours.append(contour)
            else:
                rect_contours.append(contour)

    #cv2.drawContours(input_image, )



def get_target_square_contours(input_image, square_template_contour, white_or_gray, lower_percent_bounds, check_for_square, use_actual_size, start_time):
    target_contour = None
    white_or_gray = True

    if not white_or_gray:
        thresh_val = 30
        blur_window = 5
        first_pass = True
        is_ruler = True
        use_adaptive = False
        color_image, threshold_bw, color_img, mid_row = ci.get_image_with_color_mask(input_image, thresh_val, 
            blur_window, False, first_pass, is_ruler, use_adaptive)
        gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
    else:
        denoised = cv2.fastNlMeansDenoisingColored(input_image,None,10,10,5,9)
        gray = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)
    
    if False:
        do_corner_detection(input_image.copy(), gray)

    if False:
        do_square_detection(input_image.copy())

    utils.print_time("denoising image finished", start_time)
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

    scale_img = cv2.Canny(gray, lower_bound, upper_bound,7) 
    utils.print_time("canny done", start_time)
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

    if False:
        do_square_detection(contours)
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

    utils.print_time("finished finding target square", start_time)
    #orig contours are returned for display/testing
    if False:
        cv2.drawContours(input_image, [target_contour], 0, (0,255,255),4)
        utils.show_img("square contours", input_image)
    
    return target_contour, tcontours

def get_filtered_square_contours(scale_contours, target_contour, target_perc, img_area, check_squareness):
    matches = []
    for scontour in scale_contours:
        try:
            carea = cv2.contourArea(scontour)
            hull = cv2.convexHull(scontour,returnPoints=True)
            hullArea = cv2.contourArea(hull)

            perc = hullArea/img_area

            if perc <= 0.3 and perc >= target_perc:
                if check_squareness:
                    if utils.is_really_square(scontour):
                        matches.append(scontour)
                else:
                    matches.append(scontour)

        except Exception as e:
            continue

    return matches



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
    utils.print_time("first pass on square with contour time ", start_time)
    white_or_gray = utils.is_white_or_gray(input_image, False)
    square_contour, scale_contours = get_target_square_contours(input_image, square_template_contour, white_or_gray, 0.005, True, True, start_time)
    

    if square_contour is None or len(square_contour) == 0:
        utils.print_time("nothing on first pass, doing second", start_time)
        square_contour, scale_contours = get_target_square_contours(input_image, square_template_contour, white_or_gray,0.0001,True, False, start_time)

        if square_contour is None or len(square_contour) == 0:
            utils.print_time("second failed, doing last one...", start_time)
            square_contour, scale_contours = get_target_square_contours(input_image, square_template_contour, white_or_gray,0.0001,False, False, start_time)            
              
    return square_contour, scale_contours
