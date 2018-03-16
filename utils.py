import cv2
import numpy as np

ABALONE = "abalone"
RULER = "ruler"
QUARTER = "_quarter"

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
                            if which_one != ABALONE:
                                print("is contour enclosed? {}".format(contour_is_enclosed))
                            if contour_is_enclosed:
                                if not all_bets_are_off:
                                    print("throwing this one away...")
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
                print("ditching this one, defects are too high")
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
        print("empty enclosing...")
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
            #print("left:{};right:{};top:{};bottom:{}".format(extLeft, extRight, extTop, extBot))
            
            lIn = cv2.pointPolygonTest(hull,extLeft,False) >= 0
            rIn = cv2.pointPolygonTest(hull,extRight,False) >= 0
            tIn = cv2.pointPolygonTest(hull,extTop,False) >= 0
            bIn = cv2.pointPolygonTest(hull,extBot,False) >= 0

            #print("lIn:{};rIn:{};tIn:{};bIn:{}".format(lIn, rIn, tIn, bIn))
            contour_is_enclosed = (lIn or rIn or tIn or bIn)

        return contour_is_enclosed
    except Exception as e:
        print("getting enclosed barfed: {}".format(e))
        return False

def is_really_round(contour):
    x,y,w,h = cv2.boundingRect(contour)
    
    w_v_h = float(w)/float(h)

    lim = 0.65
    is_round = (w_v_h >= lim and w_v_h <= (1.0/lim))
    return is_round

def get_large_edges(cnts):
    if len(cnts) == 0:
        return None, None
    return cnts
def get_largest_edges(cnts):
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
                target_contours.insert(dex, pair)
            except Exception:
                continue

    except Exception:
        print("error getting largest edge")
        return None, None

    results = []
    #for x, pair in enumerate(target_contours):
        
        #results.append(pair[1])
    return target_contours[:5]


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
                print("couldn't get size")
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
        print("error getting largest edge")
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
        print("couldn't draw image...{}".format(err))

def show_img(title, img):
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def get_centroid(contour):
    try:
        M = cv2.moments(contour)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    except Exception:
        cX,cY = 10000.0,10000.0

    return cX,cY

def is_dark(image):
    h_vals = []
    s_vals = []
    v_vals = []
    for i in range(130,132):
        for j in range(131,133):
            h_vals.append(image[i][j][0])
            s_vals.append(image[i][j][1])
            v_vals.append(image[i][j][2])

    rows = len(image)
    cols = len(image[0])
    
    for i in range(rows-85, rows-83):
        for j in range(cols-85, cols-83):
            h_vals.append(image[i][j][0])
            s_vals.append(image[i][j][1])
            v_vals.append(image[i][j][2])


    mean_s_val = np.mean(s_vals)
    mean_v_val = np.mean(v_vals)
    mean_h_val = np.mean(h_vals) 
    #print "h{},s{},v{}".format(mean_h_val, mean_s_val, mean_v_val)
    return (mean_s_val < 30 and mean_v_val <100)   

def is_light_background(image):
    h_vals = []
    s_vals = []
    v_vals = []
    for i in range(130,132):
        for j in range(131,133):
            h_vals.append(image[i][j][0])
            s_vals.append(image[i][j][1])
            v_vals.append(image[i][j][2])

    rows = len(image)
    cols = len(image[0])
    
    for i in range(rows-85, rows-83):
        for j in range(cols-85, cols-83):
            h_vals.append(image[i][j][0])
            s_vals.append(image[i][j][1])
            v_vals.append(image[i][j][2])



    mean_s_val = np.median(s_vals)
    mean_v_val = np.median(v_vals)
    mean_h_val = np.median(h_vals) 
    #print "H:{}, S:{}, V:{}".format(mean_h_val, mean_s_val, mean_v_val)
    return (mean_s_val < 25 and mean_v_val > 75)

def is_bright_background(image):
    h_vals = []
    s_vals = []
    v_vals = []
    for i in range(130,132):
        for j in range(131,133):
            h_vals.append(image[i][j][0])
            s_vals.append(image[i][j][1])
            v_vals.append(image[i][j][2])

    rows = len(image)
    cols = len(image[0])
    
    for i in range(rows-85, rows-83):
        for j in range(cols-85, cols-83):
            h_vals.append(image[i][j][0])
            s_vals.append(image[i][j][1])
            v_vals.append(image[i][j][2])



    mean_s_val = np.median(s_vals)
    mean_v_val = np.median(v_vals)
    mean_h_val = np.median(h_vals) 
    
    return (mean_h_val < 30 and mean_s_val > 50 and mean_v_val < 60)
    #return mean_s_val > 75

def reject_outliers(data, m=2):
    return data[abs(data - np.mean(data)) < m * np.std(data)]

def is_white_or_gray(input_image):
    #used for setting erode/dilate thresholds
    mean_color = get_mean_background_color(input_image)
    #low saturation and high value -- white or really light gray
    return mean_color[1] < 75 and mean_color[2] > 200

def is_color(input_image):
    image = cv2.cvtColor(input_image, cv2.COLOR_BGR2HSV)
    h_vals = []
    s_vals = []
    v_vals = []
    for i in range(130,132):
        for j in range(131,133):
            h_vals.append(image[i][j][0])
            s_vals.append(image[i][j][1])
            v_vals.append(image[i][j][2])

    rows = len(image)
    cols = len(image[0])
    
    for i in range(rows-85, rows-83):
        for j in range(cols-85, cols-83):
            h_vals.append(image[i][j][0])
            s_vals.append(image[i][j][1])
            v_vals.append(image[i][j][2])

    mean_s_val = np.mean(s_vals)
    mean_v_val = np.mean(v_vals)
    mean_h_val = np.mean(h_vals) 
    #print "hsv::::   {}, {}, {}".format(mean_h_val, mean_s_val, mean_v_val)
    return (mean_s_val > 75)

def is_background_similar_color(input_image):
    ab_h, ab_s, ab_v = get_mean_abalone_color(input_image)
    back_h, back_s, back_v = get_mean_background_color(input_image)


    diff_h = abs(ab_h - back_h)
    diff_s = abs(ab_s - back_s)
    diff_v = abs(ab_v - back_v)
    
    return diff_v


def get_mean_background_color(input_image):
    image = cv2.cvtColor(input_image, cv2.COLOR_BGR2HSV)
    h_vals = []
    s_vals = []
    v_vals = []
    for i in range(130,145):
        for j in range(255,265):
            h_vals.append(image[i][j][0])
            s_vals.append(image[i][j][1])
            v_vals.append(image[i][j][2])

    rows = len(image)
    cols = len(image[0])
    
    for i in range(rows-100, rows-150):
        for j in range(cols-100, cols-150):
            h_vals.append(image[i][j][0])
            s_vals.append(image[i][j][1])
            v_vals.append(image[i][j][2])


    mean_s_val = np.mean(s_vals)
    mean_v_val = np.mean(v_vals)
    mean_h_val = np.mean(h_vals) 
    return mean_h_val, mean_s_val, mean_v_val

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