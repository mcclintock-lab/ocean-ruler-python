import utils
import cv2
import numpy as np

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


    #gray_denoised = cv2.cvtColor(edged_img, cv2.COLOR_BGR2GRAY)
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
    #white_or_gray = utils.is_white_or_gray(input_image)
    white_or_gray = True
    target_contour, orig_contours, top_offset, left_offset = get_target_lobster_contour(input_image.copy(), lobster_template_contour, 0.02, white_or_gray, False, 150)
    if target_contour is None:
        target_contour, orig_contours = get_target_lobster_contour(input_image.copy(), lobster_template_contour, 0.005, white_or_gray, True, 300)

    ncols = len(input_image[0]) 
    nrows = len(input_image)
    img_area = nrows*ncols


    contours = np.array(target_contour)

    return contours, orig_contours, left_offset, top_offset

def get_square_image(input_image, use_opposite):
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

def get_target_square_contours(input_image, square_template_contour, white_or_gray, lower_percent_bounds, use_opposite):
    target_contour = None
    white_or_gray = True
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
    #gray_denoised = cv2.cvtColor(edged_img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(scale_img.copy(), 127,255,0)
    cnts = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    contours = cnts[1]
    print("number of contours: {}".format(len(contours)))

    ncols = len(input_image[0]) 
    nrows = len(input_image)
    img_area = nrows*ncols
    minVal = 100000000
    dex = 0
    for i, contour in enumerate(contours):
        try:
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)   
            contour_area = cv2.contourArea(contour)

            hull_perc = hull_area/img_area
            actual_perc = contour_area/img_area

            if actual_perc <= 0.3 and actual_perc > lower_percent_bounds:
                if(len(contour) == 0):
                    continue

                x,y,w,h = cv2.boundingRect(contour)
                #ditch cutting board borders around the outside
                if w < 0.2*ncols and h < 0.2*nrows:

                    val = cv2.matchShapes(contour, square_template_contour, 2, 0.0)
                    val=val*(1/contour_area)

                    if val < minVal:
                        dex = i
                        minVal = val

                    target_contour = contour
        except Exception as e:
            print("error: {}".format(e))
            continue
    #orig contours are returned for display/testing
    #cv2.drawContours(input_image, [target_contour], 0, (0,255,255),4)
    #utils.show_img("square contours", input_image)
    return target_contour, contours

def get_filtered_square_contours(scale_contours, target_contour, target_perc, img_area, check_squareness):
    matches = []
    for scontour in scale_contours:
        try:
            carea = cv2.contourArea(scontour)
            hull = cv2.convexHull(scontour,returnPoints=True)
            hullArea = cv2.contourArea(hull)

            perc = hullArea/img_area

            if perc <= 0.1 and perc >= target_perc:
                if check_squareness:
                    if utils.is_really_square(scontour):
                        matches.append(scontour)
                else:
                    matches.append(scontour)

        except Exception as e:
            continue

    return matches

def get_square_contour(input_image, lobster_contour, square_template_contour, look_for_shapes):

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
    print("first pass.....")
    white_or_gray = utils.is_white_or_gray(input_image)
    square_contour, scale_contours = get_target_square_contours(input_image, square_template_contour, white_or_gray, 0.005, False)
    

    if square_contour is None or len(square_contour) == 0:
        print("nothing on first pass, doing second")
        square_contour, scale_contours = get_target_square_contours(input_image, square_template_contour, white_or_gray,0.0005,False)
              
    return square_contour, scale_contours
