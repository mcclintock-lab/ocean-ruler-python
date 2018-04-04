import cv2
import utils
import numpy as np
import color_images as ci

def get_filtered_quarter_contours(scale_contours, target_contour, target_perc, img_area, check_roundness):
    matches = []
    for scontour in scale_contours:
        try:
            carea = cv2.contourArea(scontour)
            hull = cv2.convexHull(scontour,returnPoints=True)
            hullArea = cv2.contourArea(hull)

            perc = hullArea/img_area

            if perc <= 0.05 and perc >= target_perc:
                if not utils.is_contour_enclosed(scontour, target_contour, True, not check_roundness):
                    if check_roundness:
                        if utils.is_really_round(scontour):
                            matches.append(scontour)
                    else:
                        matches.append(scontour)

        except Exception as e:
            continue

    return matches



def get_target_abalone_contour(input_image, abalone_template_contour, lower_percent_bounds, white_or_gray, use_opposite):
    
    target_contour = None
    gray = utils.get_gray_image(input_image, white_or_gray, use_opposite)
    if use_opposite:
        white_or_gray = not white_or_gray
    blur = cv2.GaussianBlur(gray, (5,5),0)
    if white_or_gray:
        lower_bound = 20
        upper_bound = 100
    else:
        lower_bound = 50
        upper_bound = 200
        
    edged_img = cv2.Canny(blur, lower_bound, upper_bound,7) 

    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(13,17))
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

    #do this if edges are continuos and huge
    #erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9))
    #edged_img = cv2.morphologyEx(edged_img, cv2.MORPH_OPEN, erode_kernel)
    #edged_img = cv2.erode(edged_img, erode_kernel, iterations=3)
    #

    #gray_denoised = cv2.cvtColor(edged_img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(edged_img.copy(), 127,255,0)
    cnts = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)


    largest = utils.get_largest_edges(cnts[1])
    ncols = len(input_image[0]) 
    nrows = len(input_image)
    img_area = nrows*ncols
    minVal = 100000000
    dex = 0
    for i, contour in enumerate(largest):
        perc = contour[0]/img_area
        actual_perc = contour[2]/img_area
        current_contour = contour[1]
        if perc <= 0.75 and actual_perc > lower_percent_bounds:
            if(current_contour is None or len(current_contour) == 0):
                continue
            x,y,w,h = cv2.boundingRect(current_contour)
            #ditch cutting board borders around the outside
            if w < 0.9*ncols and h < 0.9*nrows:

                val = cv2.matchShapes(current_contour, abalone_template_contour, 2, 0.0)
                val=val*(1/contour[2])
                if val < minVal:
                    dex = i
                    minVal = val
        
                target_contour = largest[dex][1]
            
    #orig contours are returned for display/testing
    return target_contour, cnts



def get_abalone_contour(input_image, abalone_template_contour):
    white_or_gray = utils.is_white_or_gray(input_image)
    target_contour, orig_contours = get_target_abalone_contour(input_image.copy(), abalone_template_contour, 0.02, white_or_gray, False)
    if target_contour is None:
        target_contour, orig_contours = get_target_abalone_contour(input_image.copy(), abalone_template_contour, 0.005, white_or_gray, True)

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
    percOffset = xOffset/ncols
    if abs(percOffset) > 0.008:
        trimmed_contour = trim_abalone_contour(target_contour)
        if trimmed_contour is not None:
            target_contour = trimmed_contour

    contours = np.array(target_contour)

    return contours, orig_contours

def get_quarter_image(input_image, use_opposite):
    white_or_gray = utils.is_white_or_gray(input_image)
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
        denoised = cv2.fastNlMeansDenoisingColored(input_image,None,10,10,7,21)
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

def get_target_quarter_contours(input_image, use_opposite, too_close_to_abalone=False):
    scale_img, gray = get_quarter_image(input_image, use_opposite)
    kernel = np.ones((5,3), np.uint8)
    if not too_close_to_abalone:
        scale_img = cv2.dilate(scale_img, kernel, iterations=2)
    else:
        scale_img = cv2.erode(scale_img, kernel, iterations=2)

    ret, thresh = cv2.threshold(scale_img.copy(), 127,255,0)
    scale_cnts = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    scale_contours = np.array(scale_cnts[1])


    matches = []
    cnt = 0
    circle_img = gray.copy()
    circles = cv2.HoughCircles(circle_img,cv2.HOUGH_GRADIENT,1,20,param1=50,param2=30,minRadius=15,maxRadius=80)
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



def get_quarter_dimensions(input_image, abalone_contour, quarter_template_contour, look_for_shapes):

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
    print("first passs.....")
    circles, scale_contours = get_target_quarter_contours(input_image, False, False)
    matches = get_filtered_quarter_contours(scale_contours, abalone_contour, target_perc, img_area, True)
    circle_matches = get_matches(circles, matches)


    if len(circle_matches) == 0:
        #2. use the opposite of white or color
        circles, scale_contours = get_target_quarter_contours(input_image, True, False)
        matches = get_filtered_quarter_contours(scale_contours, abalone_contour, target_perc, img_area, True)
        circle_matches = get_matches(circles, matches)
        if len(circle_matches) == 0:
            #3. try the original but with check roundness turned off, so no rebuild target contours
            circles, scale_contours = get_target_quarter_contours(input_image, False, True)
            matches = get_filtered_quarter_contours(scale_contours, abalone_contour, target_perc/2, img_area, False)
            circle_matches = get_matches(circles, matches)
            if len(circle_matches) == 0:
                print("finally, doing original with half size")

                #4. finally, try it with original but with a small area
                circles, scale_contours = get_target_quarter_contours(input_image, True, True)
                matches = get_filtered_quarter_contours(scale_contours, abalone_contour, target_perc/2, img_area, True)
                circle_matches = get_matches(circles, matches)
              
    minVal=1000000
    dex = 0

    if(len(circle_matches)) > 1:
        for i, match_data in enumerate(circle_matches):
            circle = match_data[0]
            match_contour = match_data[1]
            match_hull = match_data[2]

            val = cv2.matchShapes(match_contour, quarter_template_contour, 2, 0.0)

            #defects = cv2.convexityDefects(match_contour, match_hull)
            #print("defects: {}".format(defects))

            if val < minVal:
                dex = i
                minVal = val

        #cx, cy, radius = get_quarter_contour_info(circle_matches[dex][1])
        cx, cy, radius = get_circle_info(circle_matches[dex][0])

    elif(len(circle_matches) == 1):
        '''
        target_quarter_contour = circle_matches[0][0]
        cx = target_quarter_contour[0]
        cy = target_quarter_contour[1]
        radius = target_quarter_contour[2]
        '''
        cx, cy, radius = get_circle_info(circle_matches[0][0])
    elif(len(circle_matches) == 0):
        print("picking rando...")
        return int(ncols/2), int(nrows*0.9), 35, matches
    else:
        dex = 0
        for i, match in enumerate(matches):
            val = cv2.matchShapes(match, quarter_template_contour, 2, 0.0)

            if val < minVal:
                dex = i
                minVal = val
        
        target_quarter_contour = matches[dex]
        
        cx, cy,radius = get_quarter_contour_info(target_quarter_contour)

    return cx, cy, radius, matches

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
        print("error: {}".format(e))
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
