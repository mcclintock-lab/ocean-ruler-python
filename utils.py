import cv2
import numpy as np

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
            edged_img = cv2.Canny(thresh_img, 20, 60)       

    edged_img = cv2.dilate(edged_img, None, iterations=di)
    edged_img = cv2.erode(edged_img, None, iterations=erode_iterations)
    
    if showImg:
        show_img("result image", thresh_img)
        show_img("edged img ", edged_img)

    return edged_img

def is_contour_enclosed(contour, enclosing_contour):
    if enclosing_contour is None:
        return False

    try:
        hull = cv2.convexHull(enclosing_contour,returnPoints = True)

        extLeft = tuple(contour[contour[:, :, 0].argmin()][0])
        extRight = tuple(contour[contour[:, :, 0].argmax()][0])
        extTop = tuple(contour[contour[:, :, 1].argmin()][0])
        extBot = tuple(contour[contour[:, :, 1].argmax()][0])
        #print "left:{};right:{};top:{};bottom:{}".format(extLeft, extRight, extTop, extBot)
        
        lIn = cv2.pointPolygonTest(hull,extLeft,False) >= 0
        rIn = cv2.pointPolygonTest(hull,extRight,False) >= 0
        tIn = cv2.pointPolygonTest(hull,extTop,False) >= 0
        bIn = cv2.pointPolygonTest(hull,extBot,False) >= 0

        #print "lIn:{};rIn:{};tIn:{};bIn:{}".format(lIn, rIn, tIn, bIn)
        contour_is_enclosed = lIn or rIn or tIn or bIn
        return contour_is_enclosed
    except StandardError:
        return False

def get_large_edges(cnts):
    if len(cnts) == 0:
        return None, None
    return cnts

def get_largest_edge(cnts):
    if len(cnts) == 0:
        print "no edges..."
        return None, None
    try:
        max_size = 0
        targetDex = 0
        target_contours = []

        for i, contour in enumerate(cnts):
            if len(contour) == 0:
                continue

            carea = cv2.contourArea(contour)    
            #thull = cv2.convexHull(contour)
            #harea = cv2.contourArea(thull)        

            if carea > max_size:
                max_size = carea


        #include ties
        for i, contour in enumerate(cnts):
            carea = cv2.contourArea(contour)    
            #thull = cv2.convexHull(contour)
            #harea = cv2.contourArea(thull)    
            if carea >= max_size:
                target_contours.append(contour)

    except  StandardError, e:
        print "skipping contour with no points..."
        return None, None

    return target_contours, max_size


def show_img_and_contour(imageName, input_image, contour, template_contour,top_offset=0):
    try:
        if contour is not None:
            #cv2.drawContours(input_image, [contour], 0, (0,0,255), 3)
            cv2.fillPoly(input_image, [contour], (0,255,255))
            cv2.drawContours(input_image, [template_contour], 0, (255,0,0), 3)
            show_img(imageName, input_image)
    except StandardError, e:
        print "couldn't draw image..."

def show_img(title, img):
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def get_centroid(contour):
    try:
        M = cv2.moments(contour)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    except StandardError, e:
        cX,cY = 10000.0,10000.0

    return cX,cY

def is_bright_background(input_image):
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

    return (mean_s_val < 30 or mean_h_val < 30)

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

    print "hue vals: {}".format(mean_h_val)
    print "sat vals: {}".format(mean_s_val)
    print "val vals: {}".format(mean_v_val)
    return (mean_v_val > 50)