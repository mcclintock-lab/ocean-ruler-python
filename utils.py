import cv2
import time
import numpy as np
import color_images as ci
import pdb
ABALONE = "abalone"
RULER = "ruler"
QUARTER = "_quarter"


def print_time(msg, start_time):
    now = time.time()
    elapsed = now - start_time
    #print("{} time elapsed: {}".format(msg, elapsed))

def isLobster(fishery_type):
    return "lobster" in fishery_type

def isScallop(fishery_type):
    return "scallop" in fishery_type

def isFinfish(fishery_type):
    return "finfish" in fishery_type

def isAbalone(fishery_type):
    return "abalone" in fishery_type

def isQuarter(ref_object_type):
    return "quarter" in ref_object_type

def get_thumbnail(image_full):
    target_cols = 200.0

    orig_cols = len(image_full[0]) 
    orig_rows = len(image_full)
 
    target_rows = (float(orig_rows)/(float(orig_cols))*200.0)
    fx = float(target_cols/orig_cols)
    fy = float(target_rows/orig_rows)

    thumb = cv2.resize( image_full, (0,0), fx = fx, fy = fy)
    return thumb

def get_best_contour(shapes, lower_area, upper_area, which_one, enclosing_contour, retry, scaled_rows, scaled_cols, input_image=None, all_bets_are_off=False):
    if len(shapes) == 0:
        return None, None, None

    ab_by_combined = sorted(shapes, key=lambda shape: shape[5])
    ab_by_value =  sorted(shapes, key=lambda shape: shape[0])
    ab_by_dist = sorted(shapes, key=lambda shape: shape[2])
    
    i=0
    lowest_val = ab_by_value[0][0]
    lowest_combined = ab_by_combined[0][5]
    lowest_dist = ab_by_dist[0][2]
    lowest_cdiff = ab_by_dist[0][6]

    minValue = 100000000

    targetContour = None
    targetKey = ""
    x=0
    targetH = 0
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
        cdiff = values[6]


        combined = val*haus_dist*cdiff

        #check to see if the ruler contour is inside the abalone contour. if it is, its a bogus shape
        
        #drop contours that fill the image, like the cutting board edges
        if which_one == ABALONE:
            width_limit = 0.85
            height_limit = 0.85
        else:
            width_limit = 0.25
            height_limit = 0.25

        if area_perc > lower_area and area_perc < upper_area:

            x,y,w,h = cv2.boundingRect(contour)

            #get rid of the ones with big outlying streaks or edges
            hprop = float(h)/float(scaled_rows)
            wprop = float(w)/float(scaled_cols)

            
            if (combined < minValue):
                if all_bets_are_off or ((wprop < width_limit) and (hprop < height_limit)):
                    i+=1
                    if False:
                        print("{}-{} :: combined:{};  val:{}; haus_dist:{};area:{};wprop:{};hprop:{},width limit:{}".format(which_one,
                            contour_key,combined, val,haus_dist,area_perc, wprop, hprop,width_limit))
                
                    if contour_key.endswith(QUARTER):
                        contour_is_enclosed = False
                        if enclosing_contour is not None:
                            use_hull =  not retry
                            #utils.show_img_and_contour("enclosed contour", input_image, enclosing_contour, contour)
                            contour_is_enclosed = utils.is_contour_enclosed(contour, enclosing_contour, use_hull)

                            if contour_is_enclosed:
                                if not all_bets_are_off:
                                    #utils.show_img_and_contour("enclosed contour", input_image, enclosing_contour, contour)
                                    continue
                    minValue = combined
                    targetContour = contour
                    targetKey = contour_key



    if targetContour is not None and which_one != ABALONE:
        hull = cv2.convexHull(targetContour,returnPoints = False)
        defects = cv2.convexityDefects(targetContour,hull)
        dists = []
        for i in range(defects.shape[0]):
            s,e,f,d = defects[i,0]
            if i > 0:
                dists.append(d)
        
        #try to weight the ones that are far off circular
        defect_dist = np.mean(dists)
        minValue = minValue*defect_dist
        if not all_bets_are_off:

            if which_one != ABALONE and defect_dist > 2000:
                targetContour = None
                targetKey = None
                minValue = 1000000
            elif which_one == ABALONE and defect_dist > 5000:
                targetContour = None
                targetKey = None
                minValue = 1000000

    return targetContour, targetKey, minValue


def find_edges(img=None, thresh_img=None, use_gray=False, showImg=False, erode_iterations=1,small_img=False):
    # perform edge detection, then perform a dilation + erosion to
    # close gaps in between object edges

    if  use_gray:
        #quarter only
        di = 1
        edged_img = cv2.Canny(img, 60, 255)
    else:
        di = 1
        if erode_iterations == 1:
            #ruler
            edged_img = cv2.Canny(thresh_img, 20, 150)  
        else:
            edged_img = cv2.Canny(thresh_img, 20, 100)       

    edged_img = cv2.dilate(edged_img, np.ones((3,3), np.uint8), iterations=di)
    edged_img = cv2.erode(edged_img, None, iterations=erode_iterations)
    
    if False:
        show_img("canny edge", edged_img)

    return edged_img

def is_contour_enclosed(contour, enclosing_contour, use_hull, check_centroid):
    if enclosing_contour is None or len(enclosing_contour) == 0:
        return False

    try:
        if use_hull:
            hull = cv2.convexHull(enclosing_contour,returnPoints = True)
        else:
            hull = enclosing_contour

        if check_centroid:
            center = get_centroid(contour)
            enclosing_ellipse = cv2.fitEllipse(enclosing_contour)
            pts = cv2.boxPoints(enclosing_ellipse)
            topleft = pts[0]
            topright = pts[1]
            bottomright = pts[2]
            bottomleft = pts[3]

            xEnc = center[0] >= topleft[0] and center[0] <= topright[0]
            yEnc = center[1] >= topleft[1] and center[1] <= bottomleft[1]
            contour_is_enclosed = (xEnc and yEnc)
        else:
            extLeft = tuple(contour[contour[:, :, 0].argmin()][0])
            extRight = tuple(contour[contour[:, :, 0].argmax()][0])
            extTop = tuple(contour[contour[:, :, 1].argmin()][0])
            extBot = tuple(contour[contour[:, :, 1].argmax()][0])
            
            lIn = cv2.pointPolygonTest(hull,extLeft,False) >= 0
            rIn = cv2.pointPolygonTest(hull,extRight,False) >= 0
            tIn = cv2.pointPolygonTest(hull,extTop,False) >= 0
            bIn = cv2.pointPolygonTest(hull,extBot,False) >= 0

            contour_is_enclosed = (lIn or rIn or tIn or bIn)

        return contour_is_enclosed
    except Exception as e:
        return False

def is_really_round(contour):
    x,y,w,h = cv2.boundingRect(contour)
    
    w_v_h = float(w)/float(h)

    lim = 0.65
    is_round = (w_v_h >= lim and w_v_h <= (1.0/lim))
    return is_round

def is_really_square(contour):
    x,y,w,h = cv2.boundingRect(contour)
    
    w_v_h = float(w)/float(h)

    lim = 0.80
    is_square = (w_v_h >= lim and w_v_h <= (1.0/lim))
    return is_square

def get_large_edges(cnts):
    if len(cnts) == 0:
        return None, None
    return cnts


def get_gray_image(input_image, white_or_gray, use_opposite):
    if (not white_or_gray and not use_opposite) or (use_opposite and white_or_gray):
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
    return gray

def get_largest_edges(cnts):
    if len(cnts) == 0:
        return None, None
    try:
        max_size = 0
        targetDex = 0
        target_contours = []
        contours_only = []
        #include ties
        for i, contour in enumerate(cnts):
            try:
                hull = cv2.convexHull(contour)
                carea = cv2.contourArea(hull)   
                pair = [carea, contour, cv2.contourArea(contour)]
                #thull = cv2.convexHull(contour)
                #harea = cv2.contourArea(thull) 
                dex = 0  
                for inplace in target_contours:
                    if float(inplace[0]) > float(carea):
                        dex+=1 
                    else:
                        break
                target_contours.insert(dex, pair)
                contours_only.insert(dex, contour)
            except Exception:
                continue

    except Exception:
        return None, None

    return target_contours[:5], contours_only


def get_largest_contours(cnts, num_items):
    if len(cnts) == 0:
        return None, None
    try:
        max_size = 0
        targetDex = 0
        target_contours = []

        #include ties
        for i, contour in enumerate(cnts):
            try:
                hull = cv2.convexHull(contour)
                carea = cv2.contourArea(hull)   
                pair = [carea, contour, cv2.contourArea(contour)]
                #thull = cv2.convexHull(contour)
                #harea = cv2.contourArea(thull) 
                dex = 0  
                for inplace in target_contours:
                    if float(inplace[0]) > float(carea):
                        dex+=1 
                    else:
                        break
                target_contours.insert(dex, contour)
            except Exception as e:
                continue

    except Exception:
        return None, None


    if len(target_contours) < num_items:
        num_items = len(target_contours)

    return target_contours[:num_items]

def get_largest_edge(cnts):
    if len(cnts) == 0:
        return None, None
    try:
        max_size = 0
        targetDex = 0
        target_contours = []

        for i, contour in enumerate(cnts):
            if len(contour) == 0:
                continue
            try:
                carea = cv2.contourArea(contour)    
                #thull = cv2.convexHull(contour)
                #harea = cv2.contourArea(thull)    
                if carea > max_size:
                    max_size = carea
            except Exception:
                continue

        #include ties
        for i, contour in enumerate(cnts):
            try:
                carea = cv2.contourArea(contour)   
                #thull = cv2.convexHull(contour)
                #harea = cv2.contourArea(thull)    
                if carea >= max_size:

                    target_contours.append(contour)
            except Exception:
                continue

    except Exception:
        return None, None

    return target_contours, max_size


def show_img_and_contour(imageName, input_image, contour, template_contour,top_offset=0):
    try:
        if contour is not None:
            #cv2.drawContours(input_image, [contour], 0, (0,0,255), 3)
            cv2.drawContours(input_image, [contour], 0, (0,255,255),4)
            cv2.drawContours(input_image, [template_contour], 0, (255,0,0), 3)
            show_img(imageName, input_image)
    except Exception as err:
        print_time("couldn't draw image...{}".format(err),0)

def show_img(title, img, rescale_width=None):
    if rescale_width:
        img = scale_img(img, rescale_width)
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def show_all_imgs(titles, imgs, rescale_width=None):
    for i in range(len(titles)):
        if rescale_width:
            img = scale_img(imgs[i], rescale_width)
        else:
            img = imgs[i]
        cv2.imshow(titles[i], img)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def scale_img(img, width_pixels):
    orig_cols = len(img[0]) 
    orig_rows = len(img)

    target_cols = width_pixels
    
    target_rows = (float(orig_rows)/(float(orig_cols))*target_cols)
    fx = float(target_cols)/float(orig_cols)
    fy = float(target_rows)/float(orig_rows)

    scaled_image = cv2.resize( img, (0,0), fx = fx, fy = fy)
    return scaled_image


def get_centroid(contour):
    try:
        M = cv2.moments(contour)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    except Exception:
        cX,cY = 10000.0,10000.0

    return cX,cY

def is_dark(image):
    (mean_h_val, mean_s_val, mean_v_val) = get_mean_background_color(image)

    return (mean_s_val < 30 and mean_v_val <100)   

def is_light_background(image):
    (mean_h_val, mean_s_val, mean_v_val) = get_mean_background_color(image)

    return (mean_s_val < 25 and mean_v_val > 75)


def is_bright_background(image):
    (mean_h_val, mean_s_val, mean_v_val) = get_mean_background_color(image)

    return (mean_h_val < 30 and mean_s_val > 50 and mean_v_val < 60)


def reject_outliers(data, m=2):
    return data[abs(data - np.mean(data)) < m * np.std(data)]

def is_white_or_gray(input_image, imageIsClipped):
    #used for setting erode/dilate thresholds
    if imageIsClipped:
        offset = 0
    else:
        offset = 100
    mean_color = get_mean_background_color(input_image, offset)

    #low saturation and high value -- white or really light gray
    return mean_color[1] < 75 and mean_color[2] > 175

def is_dark_gray(input_image):
    #used for setting erode/dilate thresholds
    mean_color = get_mean_background_color(input_image)
    #low saturation and high value -- white or really light gray
    is_dark_gray = mean_color[0] < 120 and mean_color[1] < 50 and mean_color[2] < 75

    return is_dark_gray

def is_color(input_image):
    image = cv2.cvtColor(input_image, cv2.COLOR_BGR2HSV)
    (mean_h_val, mean_s_val, mean_v_val) = get_mean_background_color(image)

    return (mean_s_val > 75)

def is_background_similar_color(input_image):
    ab_h, ab_s, ab_v = get_mean_abalone_color(input_image)
    back_h, back_s, back_v = get_mean_background_color(input_image)

    diff_h = abs(ab_h - back_h)
    diff_s = abs(ab_s - back_s)
    diff_v = abs(ab_v - back_v)
    
    return diff_v

def get_mean_background_color(input_image,offset=0):
    image = cv2.cvtColor(input_image, cv2.COLOR_BGR2HSV)
    img_median = np.median(image)
    print("img median: {}".format(img_median))
    h_vals = []
    s_vals = []
    v_vals = []
    try:
        for i in range(20+offset,35+offset):
            for j in range(30+offset,40+offset):
                h_vals.append(image[i][j][0])
                s_vals.append(image[i][j][1])
                v_vals.append(image[i][j][2])

        rows = len(image)-1
        cols = len(image[0])-1

        for i in range(rows-(30+offset), rows-offset):
            for j in range(cols-(20+offset), cols-offset):
                h_vals.append(image[i][j][0])
                s_vals.append(image[i][j][1])
                v_vals.append(image[i][j][2])
    except Exception as e:
        print("error :{}".format(e))


    s_vals = reject_outliers(np.asarray(s_vals, dtype=int))
    h_vals = reject_outliers(np.asarray(h_vals,dtype=int))
    v_vals = reject_outliers(np.asarray(v_vals, dtype=int))

    mean_s_val = np.mean(s_vals)
    mean_v_val = np.mean(v_vals)
    mean_h_val = np.mean(h_vals) 
 
    return (mean_h_val, mean_s_val, mean_v_val)

#get rid of outlying points for determining background color - prevents
#shadows/black spots/dirt from throwing off white color...
def reject_outliers(data, m = 2.):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/mdev if mdev else 0.
    return data[s<m]

def get_points(rows, cols, first_pass):
    row_first = int(rows/8)
    row_mid = int(rows/2)
    row_last = int(rows*0.85)

    if first_pass:
        col_first = int(cols/8)
        col_last = int(cols*0.925)
    else:
        col_first = int(cols/8.25)
        col_last = int(cols*0.9)

    col_mid = int(cols/2)

    upper_left = (row_first, col_first)
    mid_left = (row_mid, col_first)
    mid_right = (row_mid, col_last)
    bottom_right = (row_last, col_last)
    bottom_left = (row_last, col_first)
    upper_right = (row_first, col_last)

    if first_pass:
        pts = [upper_left, upper_right, bottom_left, bottom_right, mid_left, mid_right]
    else:
        pts = [mid_left, bottom_left, bottom_right, mid_right]

    return pts

def get_mean_abalone_color(input_image):
    image = cv2.cvtColor(input_image, cv2.COLOR_BGR2HSV)
    h_vals = []
    s_vals = []
    v_vals = []

    rows = len(image)
    cols = len(image[0])

    for i in range(int((rows/2))-25, int((rows/2))+25):
        for j in range(int((cols/2))-25, int((cols/2))+25):
            h_vals.append(image[i][j][0])
            s_vals.append(image[i][j][1])
            v_vals.append(image[i][j][2])


    mean_s_val = np.mean(s_vals)
    mean_v_val = np.mean(v_vals)
    mean_h_val = np.mean(h_vals) 


    return mean_h_val, mean_s_val, mean_v_val

    