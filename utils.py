import cv2


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
    
    max_size = 0
    targetDex = 0
    target_contours = []

    for i, contour in enumerate(cnts):
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