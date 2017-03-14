import cv2
import utils
import numpy as np
import matching

def get_color_image(orig_image, hue_offset, first_pass=True, is_bright = False,is_ruler=False):

    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2HSV)
    '''
    if len(image) > 500:
        mask = cv2.inRange(image, np.array([0,0,255]), np.array([300,0,255]))
        
        notmask = cv2.bitwise_not(mask)
        image = cv2.bitwise_and(orig_image,orig_image,mask=notmask)
    '''
    sat_offset =  10
    val_offset = 20

    #make this adjust to look for background with color?
    rows = len(orig_image)
    cols = len(orig_image[0])

    pts = utils.get_points(rows, cols, first_pass)

    #final_image = np.zeros((rows,cols,3), np.uint8)
    if not is_ruler:
        hue_offset = hue_offset+sat_offset

    else:
        hue_offset = hue_offset

    final_huemin = 400
    final_huemax = -1

    final_satmin = 400
    final_satmax = -1

    final_valmin = 400
    final_valmax = -1

    if utils.is_dark(image):
        #print "its dark!"
        sat_minus = hue_offset+sat_offset
        if is_ruler:
            val_minus = hue_offset+val_offset/2
        else:
            val_minus = hue_offset+val_offset/2

        sat_plus = hue_offset+sat_offset
        val_plus = hue_offset
    else:
        if utils.is_bright_background(image):
            #print "is bright!"
            sat_minus = hue_offset+sat_offset
            val_minus = hue_offset
            sat_plus = hue_offset+sat_offset

            if is_ruler:
                val_plus = hue_offset+val_offset
            else:
                val_plus = hue_offset
        else:  
            #print "not bright!"
            sat_minus = hue_offset+sat_offset
            val_minus = hue_offset+val_offset
            sat_plus = hue_offset+sat_offset*2
            val_plus = hue_offset+val_offset*2

    #fix this - figure out how to make it a mask of ones and pull out the right bits...
    for pt in pts:
        for i in range(0,3):
            for j in range(0,3):
                tgt_row = pt[0]+i
                tgt_col = pt[1]+j
                val = image[tgt_row,tgt_col]
                #print "h:{},s:{},v:{}".format(val[0],val[1],val[2])
                huemin = get_min(val[0]-hue_offset)
                satmin = get_min(val[1]-sat_minus)
                valmin = get_min(val[2]-val_minus)
                
                huemax = get_max(int(val[0])+hue_offset)
                satmax = get_max(int(val[1])+sat_plus)
                valmax = get_max(int(val[2])+val_plus)

                final_huemin = min(final_huemin, huemin)
                final_huemax = max(final_huemax, huemax)

                final_satmin = min(final_satmin, satmin)
                final_satmax = max(final_satmax, satmax)

                final_valmin = min(final_valmin, valmin)
                final_valmax = max(final_valmax, valmax)

    minrange = np.array([final_huemin, final_satmin, final_valmin])
    maxrange = np.array([final_huemax, final_satmax, final_valmax])


    #use original image so we get the non-masked values
    mask = cv2.inRange(image, minrange, maxrange)
    notmask = cv2.bitwise_not(mask)
    image = cv2.bitwise_and(image,image,mask=notmask)
    bgr = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    if False:
        print "{}".format(minrange)
        print "{}".format(maxrange)
        rows = len(image)
        cols = len(image[0])
        print "rows: {}, cols:{}".format(rows,cols)
        for pt in pts:
            print "drawing pt {}".format(pt)
            endpt = (pt[0]+2, pt[1]+2)
            print "drawing endpt {}".format(endpt)
            cv2.rectangle(image, (pt[1],pt[0]), (endpt[1],endpt[0]),(255,0,0),10)
        cv2.imshow("result {}".format(hue_offset), image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return bgr


def get_image_with_color_mask(input_image, thresh_val, blur_window, show_img,first_pass, is_ruler, use_adaptive):
    rows = len(input_image)
    cols = len(input_image[0])
    image = input_image

    is_bright = utils.is_bright_background(image)
    color_res = get_color_image(image, thresh_val+blur_window, first_pass=first_pass, is_bright=is_bright,is_ruler=is_ruler)
    
    #utils.show_img("color_res {};{}".format(thresh_val, blur_window),color_res)
    #maybe use this to see if we should threshold?
    gray = cv2.cvtColor(color_res, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (blur_window, blur_window), 0)
    
    if is_bright:
        #TODO: check for high variability/check patterns here?
        if is_ruler and use_adaptive:
            threshold_bw = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,7,3);
        else:
            retval, threshold_bw = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    else:
        #threshold_bw = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,13,3);
        retval, threshold_bw = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    if False:
        utils.show_img("thresh {};{}".format(thresh_val, blur_window),threshold_bw)

    return image, threshold_bw, color_res, rows

def do_color_image_match(input_image, template_contour, thresh_val, blur_window, showImg=False, 
    contour_color=None, is_ruler=False, use_gray_threshold=False, enclosing_contour=None, first_pass=True, 
    small_img=False, use_adaptive=False):
    #try the color image
    color_image, threshold_bw, color_img, mid_row = get_image_with_color_mask(input_image, thresh_val, 
        blur_window, showImg, first_pass, is_ruler, use_adaptive)
    rows = len(input_image)
    cols = len(input_image[0])

    if is_ruler:
        erode_iterations = 1
    else:
        if use_gray_threshold:
            erode_iterations = 1
        else:
            if not first_pass:
                erode_iterations = 1
            else:
                erode_iterations = 1

    edged = utils.find_edges(img=color_img, thresh_img=threshold_bw, use_gray=use_gray_threshold, showImg=False, 
        erode_iterations=erode_iterations,small_img=small_img)

    # find contours in the edge map
    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    cnts = cnts[1] #if imutils.is_cv2() else cnts[1]

    if is_ruler:
        contours= utils.get_large_edges(cnts)
        if False:
            cv2.drawContours(input_image, contours, -1, (0,255,0), 1)
            cv2.imshow("color image {}x{}".format(thresh_val, blur_window), input_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    else:
        contours, size = utils.get_largest_edge(cnts)

    if contours is None:
        return None, None, None,None,None

    smallest_combined = 10000000.0
    target_contour = None
    rval = 10000000
    aperc = 10000000
    adist = 10000000
    cdiff = 10000000

    if False:
        cv2.drawContours(input_image, contours, -1, (0,255,0), 3)
        cv2.imshow("{}x{}".format(thresh_val, blur_window), input_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    min_area = 0.25
    if small_img:
        min_area = 0.10
    theratio = 0
    ratio = 0

    for contour in contours:
        the_contour, result_val, area_perc, area_dist, centroid_diff = matching.sort_by_matching_shape(contour, template_contour, 
            False, input_image, is_ruler, first_pass)
        #ditch the outliers -- this is fine tuned later

        if the_contour is None:
            continue

        if small_img and is_ruler:
            if (area_perc < 0.25 or area_perc > 3.0):
                continue
        else:
            if (area_perc < 0.25 or area_perc > 2.0):
                continue


        area_perc_limit_for_ruler = 0.25

        if is_ruler:
            x,y,w,h = cv2.boundingRect(contour)
            tgt_ratio = 0.70
            ratio = float(w)/float(h)

            if ratio < tgt_ratio or 1.0/ratio < tgt_ratio:
                continue

            width_perc = float(h)/float(cols)
            if small_img:
                width_perc_max = 0.25
                area_perc_limit_for_ruler = 0.40
            else:
                width_perc_max = 0.10

            if width_perc > width_perc_max:
                continue

            #if its the ruler (quarter), check to see if its enclosed
            is_enclosed = utils.is_contour_enclosed(the_contour, enclosing_contour,first_pass)
            if is_enclosed:
                continue


        comb = result_val*area_dist*centroid_diff
        #comb = result_val

        if comb < smallest_combined:
            if (not is_ruler) or (is_ruler and area_perc > area_perc_limit_for_ruler):
                smallest_combined = comb
                rval = result_val
                aperc = area_perc
                adist = area_dist
                target_contour = the_contour
                cdiff = centroid_diff
                theratio = ratio

    if False:
        if is_ruler:
            #get rid of this
            if thresh_val == 10:
                '''
                templateHull = cv2.convexHull(template_contour)
                templateArea = cv2.contourArea(template_contour)
                targetHull = cv2.convexHull(target_contour)
                targetArea = cv2.contourArea(target_contour)
                hullArea = cv2.contourArea(targetHull)
                print "target area: {}, hullArea: {}, template area: {}".format(targetArea, hullArea, templateArea)
                '''
            utils.show_img_and_contour("big color img {}x{};val:{}".format(thresh_val, blur_window, rval), input_image, target_contour, enclosing_contour,0)

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