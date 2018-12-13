import cv2
import utils
import numpy as np
import math
from scipy.spatial import distance

def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

def get_width_from_ruler(dB, rulerWidth):
    return (dB)/float(rulerWidth)

def get_corner_points(pre, contour):

    if pre == "Ruler":

        (cX, cY),radius = cv2.minEnclosingCircle(contour)
        center = (int(cX),int(cY))
        brect = cv2.boundingRect(contour)
       
        w = brect[2]
        h = brect[3]

        if abs(w-h) <= 1:
            radius = int(radius)
        else:
            radius = int(max(w,h)/2)-2

        tl = (cX-radius, cY-radius)
        tr = (cX+radius, cY-radius)
        bl = (cX-radius, cY+radius)
        br = (cX+radius, cY+radius)
        return tl, tr, bl, br
    else:
        return get_bounding_corner_points(contour)
   

def get_bounding_corner_points(contour):
    brect = cv2.boundingRect(contour)
    # unpack the ordered bounding box, then compute the midpoint
    # between the top-left and top-right coordinates, followed by
    # the midpoint between bottom-left and bottom-right coordinates
    x = brect[0]
    y=brect[1]
    y = brect[1]
    width=brect[2]
    height=brect[3]
    tl = (x, y+height)
    tr = (x+width, y+height)
    bl = (x,y)
    br = (x+width, y)
    corners = [tl, tr, br, bl]
    return tl, tr, bl, br

def drawLines(base_img, flipDrawing, startLinePoint, endLinePoint, drawHatches):
    cv2.line(base_img, startLinePoint, endLinePoint,
            (255, 0, 255), 1)
    if drawHatches:
        if flipDrawing:
            firstHatchStart = (int(startLinePoint[0]-50), int(startLinePoint[1]))
            firstHatchEnd = (int(startLinePoint[0]+50), int(startLinePoint[1]))
            secondHatchStart = (int(endLinePoint[0]-50), int(endLinePoint[1]))
            secondHatchEnd = (int(endLinePoint[0]+50), int(endLinePoint[1]))
        else:
            firstHatchStart = (int(startLinePoint[0]), int(startLinePoint[1]-50))
            firstHatchEnd = (int(startLinePoint[0]), int(startLinePoint[1]+50))
            secondHatchStart = (int(endLinePoint[0]), int(endLinePoint[1]-50))
            secondHatchEnd = (int(endLinePoint[0]), int(endLinePoint[1]+50))

        cv2.line(base_img, firstHatchStart, firstHatchEnd,
            (255, 0, 255), 1)

        cv2.line(base_img, secondHatchStart, secondHatchEnd,
            (255, 0, 255), 1)

def get_quarter_corners(quarterCenterX, quarterCenterY, quarterRadius):

    tl = (quarterCenterX-quarterRadius, quarterCenterY-quarterRadius)
    tr = (quarterCenterX+quarterRadius, quarterCenterY-quarterRadius)
    bl = (quarterCenterX-quarterRadius, quarterCenterY+quarterRadius)
    br = (quarterCenterX+quarterRadius, quarterCenterY+quarterRadius)
    return tl, tr, bl, br

def draw_target_contour_with_width(base_img, c, draw_text, flipDrawing, pixelsPerMetric, fisheryType):
    
    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])

    if flipDrawing:
        # compute the midpoint between the top-left and top-right points,
        # followed by the midpoint between the top-righ and bottom-right
        startLinePoint = extTop
        endLinePoint = extBot
        dBX = abs(startLinePoint[0] - endLinePoint[0])
        dBY = abs(startLinePoint[1] - endLinePoint[1])
        dB = get_distance(dBX, dBY)

        widthStartLinePoint = extLeft
        widthEbdLinePoint = extRight

        dBWidthX = abs(widthStartLinePoint[0] - widthEndLinePoint[0])
        dBWidthY = abs(widthStartLinePoint[1] - widthEndLinePoint[1])
        dBWidth = get_distance(dBWidthX, dBWidthY)
    else:
        startLinePoint = extLeft
        endLinePoint = extRight

        dBX = abs(startLinePoint[0] - endLinePoint[0])
        dBY = abs(startLinePoint[1] - endLinePoint[1])
        dB = get_distance(dBX, dBY)

        widthStartLinePoint = extTop
        widthEndLinePoint = extBot

        dBWidthX = abs(widthStartLinePoint[0] - widthEndLinePoint[0])
        dBWidthY = abs(widthStartLinePoint[1] - widthEndLinePoint[1])
        dBWidth = get_distance(dBWidthX, dBWidthY)

    drawLines(base_img, not flipDrawing, startLinePoint, endLinePoint, False)
    if True:
        cv2.circle(base_img, extLeft, 8, (0, 0, 255), -1)
        cv2.circle(base_img, extRight, 8, (0, 255, 0), -1)
        cv2.circle(base_img, extTop, 8, (255, 0, 0), -1)
        cv2.circle(base_img, extBot, 8, (255, 255, 0), -1)
        cv2.drawContours(base_img,[c],0,(255,0,0),2)
        drawLines(base_img, not flipDrawing, widthStartLinePoint, widthEndLinePoint, False)
    
    '''
    x,y,w,h = cv2.boundingRect(contour)
    cv2.rectangle(base_img,(x,y),(x+w,y+h),(255,255,0),3)
    
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    cv2.drawContours(base_img,[box],0,(0,191,255),2)
    '''
    '''
    ellipse = cv2.fitEllipse(contour)
    cv2.ellipse(base_img,ellipse,(0,0,255),4)
    '''
    # if the pixels per metric has not been initialized, then
    # compute it as the ratio of pixels to supplied metric
    # (in this case, inches)

    dimB = dB / pixelsPerMetric
    dimBWidth = dBWidth/pixelsPerMetric
    if draw_text:
        if fisheryType is None or len(fisheryType) == 0:
            fisheryType = "Abalone"
        else:
            fisheryType = fisheryType.capitalize()
        # draw the object sizes on the image
        cv2.putText(base_img, fisheryType,
            (endLinePoint[0]+10, endLinePoint[1]), cv2.FONT_HERSHEY_TRIPLEX,
            1, (255, 255, 255), 1, lineType=cv2.LINE_AA)

        cv2.putText(base_img, "{:.1f}in".format(dimB),
            (endLinePoint[0]+10, endLinePoint[1]+50), cv2.FONT_HERSHEY_TRIPLEX,
            1, (255, 255, 255), 1, lineType=cv2.LINE_AA)

    return dimB, dimBWidth, startLinePoint, endLinePoint, widthStartLinePoint, widthEndLinePoint

def get_distance(xdiff, ydiff):
    distance = math.sqrt(math.pow(float(xdiff),2)+math.pow(float(ydiff),2))
    return distance

def draw_target_contour(base_img, contour, draw_text, flipDrawing, pixelsPerMetric, fisheryType):
    
    tl, tr, bl, br = get_corner_points("Abalone", contour)

    #qtl, qtr, qbl, qbr = get_quarter_corners(quarterCenterX, quarterCenterY, quarterRadius)
    #print("quarter corners: {}, {}, {}, {}: radius: {}".format(qtl, qtr, qbl, qbr, quarterRadius))
    if flipDrawing:
        # compute the midpoint between the top-left and top-right points,
        # followed by the midpoint between the top-righ and bottom-right
        startLinePoint = midpoint(tl, tr)
        startLinePoint = (int(startLinePoint[0]), int(startLinePoint[1]))
        endLinePoint = midpoint(bl, br)
        endLinePoint = (int(endLinePoint[0]), int(endLinePoint[1]))
        dB = abs(startLinePoint[1] - endLinePoint[1])

        widthStartLinePoint = midpoint(tl, bl)
        widthStartLinePoint = (int(widthStartLinePoint[0]), int(widthStartLinePoint[1]))
        widthEndLinePoint = midpoint(tr, br)
        widthEndLinePoint = (int(widthEndLinePoint[0]), int(widthEndLinePoint[1]))
        dBWidth = abs(widthStartLinePoint[1] - widthEndLinePoint[1])
    else:
        # compute the midpoint between the top-left and top-right points,
        # followed by the midpoint between the top-righ and bottom-right
        startLinePoint = midpoint(tl, bl)
        startLinePoint = (int(startLinePoint[0]), int(startLinePoint[1]))
        endLinePoint = midpoint(tr, br)
        endLinePoint = (int(endLinePoint[0]), int(endLinePoint[1]))
        # compute the Euclidean distance between the midpoints
        dBX = abs(startLinePoint[0] - endLinePoint[0])
        dBY = abs(startLinePoint[1] - endLinePoint[1])
        dB = get_distance(dBX, dBY)
        print("x distance: {}, real distance: {}".format(dBX, dB))

        widthStartLinePoint = midpoint(tl, tr)
        widthStartLinePoint = (int(widthStartLinePoint[0]), int(widthStartLinePoint[1]))
        widthEndLinePoint = midpoint(bl, br)
        widthEndLinePoint = (int(widthEndLinePoint[0]), int(widthEndLinePoint[1]))
        dBWidthX = abs(widthStartLinePoint[0] - widthEndLinePoint[0])
        dBWidthY = abs(widthStartLinePoint[1] - widthEndLinePoint[1])
        dBWidth = get_distance(dBWidthX, dBWidthY)



    # draw the midpoints on the image
    cv2.circle(base_img, (int(startLinePoint[0]), int(startLinePoint[1])), 2, (255, 0, 0), -1)
    cv2.circle(base_img, (int(endLinePoint[0]), int(endLinePoint[1])), 2, (255, 0, 0), -1)
    drawLines(base_img, flipDrawing, startLinePoint, endLinePoint, False)


    dimB = dB / pixelsPerMetric
    dimBWidth = dBWidth/pixelsPerMetric
    if draw_text:
        if fisheryType is None or len(fisheryType) == 0:
            fisheryType = "Abalone"
        else:
            fisheryType = fisheryType.capitalize()
        # draw the object sizes on the image
        cv2.putText(base_img, fisheryType,
            (endLinePoint[0]+10, endLinePoint[1]), cv2.FONT_HERSHEY_TRIPLEX,
            1, (255, 255, 255), 1, lineType=cv2.LINE_AA)

        cv2.putText(base_img, "{:.1f}in".format(dimB),
            (endLinePoint[0]+10, endLinePoint[1]+50), cv2.FONT_HERSHEY_TRIPLEX,
            1, (255, 255, 255), 1, lineType=cv2.LINE_AA)

    return dimB, dimBWidth, startLinePoint, endLinePoint, widthStartLinePoint, widthEndLinePoint



def draw_quarter_contour(base_img, contour, draw_text, flipDrawing, quarterCenterX, quarterCenterY, refWidth, refObjectSize):
    
    #tl, tr, bl, br = get_corner_points("Abalone", contour)
    qtl, qtr, qbl, qbr = get_quarter_corners(quarterCenterX, quarterCenterY, refWidth/2)
    #print("quarter corners: {}, {}, {}, {}: radius: {}".format(qtl, qtr, qbl, qbr, quarterRadius))
    if flipDrawing:
        # compute the midpoint between the top-left and top-right points,
        # followed by the midpoint between the top-righ and bottom-right
        '''
        startLinePoint = midpoint(tl, tr)
        startLinePoint = (int(startLinePoint[0]), int(startLinePoint[1]))
        endLinePoint = midpoint(bl, br)
        endLinePoint = (int(endLinePoint[0]), int(endLinePoint[1]))
        dB = abs(startLinePoint[1] - endLinePoint[1])
        '''
        quarterStartLinePoint = midpoint(qtl, qtr)
        quarterStartLinePoint = (int(quarterStartLinePoint[0]), int(quarterStartLinePoint[1]))
        quarterEndLinePoint = midpoint(qbl, qbr)
        quarterEndLinePoint = (int(quarterEndLinePoint[0]), int(quarterEndLinePoint[1]))
        dB = abs(quarterStartLinePoint[1] - quarterEndLinePoint[1])

    else:
        # compute the midpoint between the top-left and top-right points,
        # followed by the midpoint between the top-righ and bottom-right
        '''
        startLinePoint = midpoint(tl, bl)
        startLinePoint = (int(startLinePoint[0]), int(startLinePoint[1]))
        endLinePoint = midpoint(tr, br)
        endLinePoint = (int(endLinePoint[0]), int(endLinePoint[1]))
        # compute the Euclidean distance between the midpoints
        dB = abs(startLinePoint[0] - endLinePoint[0])
        '''

        quarterStartLinePoint = midpoint(qtl, qbl)
        quarterStartLinePoint = (int(quarterStartLinePoint[0]), int(quarterStartLinePoint[1]))
        quarterEndLinePoint = midpoint(qtr, qbr)
        quarterEndLinePoint = (int(quarterEndLinePoint[0]), int(quarterEndLinePoint[1]))
        dB = abs(quarterStartLinePoint[0] - quarterEndLinePoint[0])

    # draw the midpoints on the image
    '''
    cv2.circle(base_img, (int(startLinePoint[0]), int(startLinePoint[1])), 2, (255, 0, 0), -1)
    cv2.circle(base_img, (int(endLinePoint[0]), int(endLinePoint[1])), 2, (255, 0, 0), -1)
    '''

    cv2.circle(base_img, (int(quarterStartLinePoint[0]), int(quarterStartLinePoint[1])), 2, (0, 255, 0), -1)
    cv2.circle(base_img, (int(quarterEndLinePoint[0]), int(quarterEndLinePoint[1])), 2, (0, 255, 0), -1)


    # draw lines between the midpoints
    #drawLines(base_img, flipDrawing, startLinePoint, endLinePoint)
    drawLines(base_img, flipDrawing, quarterStartLinePoint, quarterEndLinePoint, False)


    # if the pixels per metric has not been initialized, then
    # compute it as the ratio of pixels to supplied metric
    # (in this case, inches)

    pixelsPerMetric = get_width_from_ruler(refWidth, refObjectSize)
    dimB = dB / pixelsPerMetric

    if draw_text:
        # draw the quarter
        cv2.putText(base_img, "{}: {}in".format("U.S. Quarter",0.955),
            (quarterEndLinePoint[0]+10, quarterEndLinePoint[1]), cv2.FONT_HERSHEY_TRIPLEX,
            1, (255, 255, 255), 1,lineType=cv2.LINE_AA)

    return pixelsPerMetric, dimB, quarterStartLinePoint, quarterEndLinePoint

def draw_square_contour(base_img, contour, pixelsPerMetric, draw_text, flipDrawing, refObjectSize):
    tl, tr, bl, br = get_bounding_corner_points(contour)
    if flipDrawing:
        # compute the midpoint between the top-left and top-right points,
        # followed by the midpoint between the top-righ and bottom-right
        startLinePoint = midpoint(tl, tr)
        startLinePoint = (int(startLinePoint[0]), int(startLinePoint[1]))
        endLinePoint = midpoint(bl, br)
        endLinePoint = (int(endLinePoint[0]), int(endLinePoint[1]))
        dB = abs(startLinePoint[1] - endLinePoint[1])
    else:
        # compute the midpoint between the top-left and top-right points,
        # followed by the midpoint between the top-righ and bottom-right

        startLinePoint = midpoint(tl, bl)
        startLinePoint = (int(startLinePoint[0]), int(startLinePoint[1]))
        endLinePoint = midpoint(tr, br)
        endLinePoint = (int(endLinePoint[0]), int(endLinePoint[1]))
        dB = abs(startLinePoint[0] - endLinePoint[0])


    pixelsPerMetric = get_width_from_ruler(dB, refObjectSize)
    print("square width: {}".format(dB))
    print("pixels per inch: {}".format(pixelsPerMetric))
    dimB = dB / pixelsPerMetric

    cv2.circle(base_img, (int(startLinePoint[0]), int(startLinePoint[1])), 4, (255, 0, 0), -1)
    cv2.circle(base_img, (int(endLinePoint[0]), int(endLinePoint[1])), 4, (255, 0, 0), -1)

    # draw lines between the midpoints
    cv2.line(base_img, startLinePoint, endLinePoint,
        (255, 0, 255), 1)

    '''
    if flipDrawing:
        firstHatchStart = (int(startLinePoint[0]-50), int(startLinePoint[1]))
        firstHatchEnd = (int(startLinePoint[0]+50), int(startLinePoint[1]))
        secondHatchStart = (int(endLinePoint[0]-50), int(endLinePoint[1]))
        secondHatchEnd = (int(endLinePoint[0]+50), int(endLinePoint[1]))
    else:
        firstHatchStart = (int(startLinePoint[0]), int(startLinePoint[1]-50))
        firstHatchEnd = (int(startLinePoint[0]), int(startLinePoint[1]+50))
        secondHatchStart = (int(endLinePoint[0]), int(endLinePoint[1]-50))
        secondHatchEnd = (int(endLinePoint[0]), int(endLinePoint[1]+50))

    cv2.line(base_img, firstHatchStart, firstHatchEnd,
        (255, 0, 255), 1)

    cv2.line(base_img, secondHatchStart, secondHatchEnd,
        (255, 0, 255), 1)
    '''

    if draw_text:
        cv2.putText(base_img, "{}".format("2 in. Square",dimB),
            (endLinePoint[0]+10, endLinePoint[1]), cv2.FONT_HERSHEY_TRIPLEX,
            1, (255, 255, 255), 1,lineType=cv2.LINE_AA)
    return pixelsPerMetric, dimB, startLinePoint, endLinePoint

def draw_target_lobster_contour(base_img, contour, pixelsPerMetric, draw_text, left_offset, top_offset, full_contour):
    c = contour

    #farthest x/y points on the contour
    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBottom = tuple(c[c[:, :, 1].argmax()][0])

    rotRect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rotRect)

    rows,cols = base_img.shape[:2]
    [vx,vy,x,y] = cv2.fitLine(full_contour, cv2.DIST_L2,0,0.01,0.01)


    lefty = int((-x*vy/vx) + y)
    righty = int(((cols-x)*vy/vx)+y)
    b = (cols-1, righty)
    a = (0, lefty)
    slope = abs(float(b[1] - a[1])/float(b[0] - a[0]))

    #use the slope to determine which x/y points to use on the contour
    if slope < 1.0:
        startLinePoint = extLeft
        endLinePoint = extRight
    else:
        startLinePoint = extTop
        endLinePoint = extBottom

    #center of the contour
    M = cv2.moments(contour)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    dB = distance.euclidean(startLinePoint, endLinePoint)
    print("dB is {}".format(dB))
    cv2.circle(base_img, (int(startLinePoint[0]), int(startLinePoint[1])), 2, (255, 0, 255), -1)
    cv2.circle(base_img, (int(endLinePoint[0]), int(endLinePoint[1])), 2, (255, 0, 255), -1)



    dimB = dB / pixelsPerMetric
    if False:
        box = np.int0(box)
        cv2.drawContours(base_img,[box],0,(125,25,25),4, offset=(left_offset, top_offset))
        cv2.drawContours(base_img,[contour],0,(125,125,125),4, offset=(left_offset, top_offset))
        cv2.drawContours(base_img,[full_contour],0,(225,225,225),4, offset=(left_offset, top_offset))
        cv2.circle(base_img, (cX, cY), 10, (50, 50, 255), -1)
        cv2.line(base_img,a,b,(0,255,0),2)


    line = [a,b]
    startLinePoint, endLinePoint = get_contour_line_intersection(base_img, contour, line, startLinePoint, endLinePoint)

    print("startLinePoint after return: {}".format(startLinePoint))
    print("endLinePoint after return: {}".format(endLinePoint))

    #utils.show_img("intersection...", img3)
    if draw_text:
            # draw the object sizes on the image
            cv2.putText(base_img, "Lobster",
                (endLinePoint[0]+10, endLinePoint[1]), cv2.FONT_HERSHEY_TRIPLEX,
                1, (255, 255, 255), 1, lineType=cv2.LINE_AA)

            cv2.putText(base_img, "{:.1f}in".format(dimB),
                (endLinePoint[0]+10, endLinePoint[1]+50), cv2.FONT_HERSHEY_TRIPLEX,
                1, (255, 255, 255), 1, lineType=cv2.LINE_AA)

    # draw lines between the midpoints
    cv2.line(base_img, startLinePoint, endLinePoint,
        (255, 0, 255), 4)

    return dimB, startLinePoint, endLinePoint

def get_contour_line_intersection(base_img, contour, line, startLinePoint, endLinePoint):
        #find the union of the fitted line for the entire lobster contour
    # and the target contour
    
    # create an image filled with zeros, single-channel, same size as img.
    blank = np.zeros( base_img.shape[0:2] )

    # copy each of the contours (assuming there's just two) to its own image. 
    # Just fill with a '1'.
    img1 = cv2.drawContours( blank.copy(), [contour], 0, 1 )

    # make the line contour
    line_contour = np.array(line).reshape((-1,1,2)).astype(np.int32)

    #draw the line contour to turn it into a mask
    img2 = cv2.drawContours( blank.copy(), [line_contour], 0, 1 )

    #and AND them together
    imgI = np.logical_and(img1, img2)

    #see where they're true (the intersections)
    locations = np.argwhere(imgI)
    print("locations: {}".format(locations))
    print("startLinePoint: {}".format(startLinePoint))
    print("endLinePoint: {}".format(endLinePoint))

    if locations.any() and len(locations) >= 2:
        print("updating line points to intersection with full line...")
        
        for i, loc in enumerate(locations):
            cX = loc[1]
            cY = loc[0]
            if(i == 0):
                startLinePoint = (cX, cY)
            elif i == 1:
                endLinePoint = (cX, cY)
            
            if True:
                cv2.circle(base_img, (cX, cY), 12, (50, 50, 255), -1)

    print("startLinePoint after update: {}".format(startLinePoint))
    print("endLinePoint after update: {}".format(endLinePoint))
    return startLinePoint, endLinePoint
def draw_lobster_contour(base_img, contour, pixelsPerMetric, draw_text, flipDrawing, rulerWidth, left_offset, top_offset, full_contour):
    #center (x,y), (width, height), angle of rotation 
    
    rotRect = cv2.minAreaRect(contour)
    width = rotRect[1][0]
    height = rotRect[1][1]
    rotAngle = abs(rotRect[2])
    verts = cv2.boxPoints(rotRect)
    print("rot angle: {}".format(rotAngle))
    if rotAngle > 45:
        tl = verts[2]
        tr = verts[3]
        br = verts[0]

        bl = verts[1]
    else:
        tl = verts[1]
        tr = verts[2]
        br = verts[3]
        bl = verts[0]

    centerPoint = rotRect[0]
    print("rotRect is {}".format(centerPoint))
    print("-------------rotAngle is: {}".format(rotAngle))
    if rotAngle < 45:
        #width is longer side
        #shows as x
        print("long side is width")
    else:
        #height is longer side
        #shows as y
        print("x is short side")
    box = cv2.boxPoints(rotRect)

    #convert from floats to int
    box = np.int0(box)
    cv2.drawContours(base_img,[box],0,(25,25,25),1, offset=(left_offset, top_offset))
    
    flipLine = (rotAngle > 45 and width > height) or (rotAngle < 45 and width < height)
    '''
    a = abs(tl[0] - tr[0])
    b = abs(tl[1] - tr[1])
    dB = math.sqrt(math.pow(a,2)+math.pow(b,2))
    '''

    if width <= height:
        dB = width
    else:
        dB = height
    #tl, tr, bl, br = get_bounding_corner_points(contour)
    tl = (tl[0]+left_offset, tl[1]+top_offset)
    tr = (tr[0]+left_offset, tr[1]+top_offset)
    bl = (bl[0]+left_offset, bl[1]+top_offset)
    br = (br[0]+left_offset, br[1]+top_offset)
    
    if flipLine:
        # compute the midpoint between the top-left and top-right points,
        # followed by the midpoint between the top-righ and bottom-right
        startLinePoint = midpoint(tl, tr)
        startLinePoint = (int(startLinePoint[0]), int(startLinePoint[1]))
        endLinePoint = midpoint(bl, br)
        endLinePoint = (int(endLinePoint[0]), int(endLinePoint[1]))

        #dB = abs(startLinePoint[1] - endLinePoint[1])
    else:
        # compute the midpoint between the top-left and top-right points,
        # followed by the midpoint between the top-righ and bottom-right

        startLinePoint = midpoint(tl, bl)
        startLinePoint = (int(startLinePoint[0]), int(startLinePoint[1]))
        endLinePoint = midpoint(tr, br)
        endLinePoint = (int(endLinePoint[0]), int(endLinePoint[1]))
        # compute the Euclidean distance between the midpoints
        #dB = abs(startLinePoint[0] - endLinePoint[0])
  

    # draw the midpoints on the image
    '''
    top_left_point = (int(tl[0]), int(tl[1]))
    cv2.circle(base_img, top_left_point, 16, (255,0,0), -1) #blue
    cv2.putText(base_img, "topleft",
        top_left_point, cv2.FONT_HERSHEY_TRIPLEX,
        1, (255, 255, 255), 1, lineType=cv2.LINE_AA)
    top_right_point = (int(tr[0]), int(tr[1]))
    cv2.circle(base_img, top_right_point, 16, (50, 50, 50), -1)#gray
    cv2.putText(base_img, "topright",
        top_right_point, cv2.FONT_HERSHEY_TRIPLEX,
        1, (255, 255, 255), 1, lineType=cv2.LINE_AA)

    bottom_right_point = (int(br[0]), int(br[1]))
    cv2.circle(base_img, bottom_right_point, 16, (255, 255, 255), -1) #white
    cv2.putText(base_img, "bottomright",
        bottom_right_point, cv2.FONT_HERSHEY_TRIPLEX,
        1, (255, 255, 255), 1, lineType=cv2.LINE_AA)

    bottom_left_point = (int(bl[0]), int(bl[1]))
    cv2.circle(base_img, bottom_left_point, 16, (0, 0, 0), -1) #black
    cv2.putText(base_img, "bottomleft",
        bottom_left_point, cv2.FONT_HERSHEY_TRIPLEX,
        1, (255, 255, 255), 1, lineType=cv2.LINE_AA)
    '''
    cv2.circle(base_img, (int(startLinePoint[0]), int(startLinePoint[1])), 4, (255, 0, 255), -1)
    cv2.circle(base_img, (int(endLinePoint[0]), int(endLinePoint[1])), 4, (255, 0, 255), -1)

    # draw lines between the midpoints
    cv2.line(base_img, startLinePoint, endLinePoint,
        (255, 0, 255), 4)

    '''
    if flipDrawing:
        firstHatchStart = (int(startLinePoint[0]-50), int(startLinePoint[1]))
        firstHatchEnd = (int(startLinePoint[0]+50), int(startLinePoint[1]))
        secondHatchStart = (int(endLinePoint[0]-50), int(endLinePoint[1]))
        secondHatchEnd = (int(endLinePoint[0]+50), int(endLinePoint[1]))
    else:
        firstHatchStart = (int(startLinePoint[0]), int(startLinePoint[1]-50))
        firstHatchEnd = (int(startLinePoint[0]), int(startLinePoint[1]+50))
        secondHatchStart = (int(endLinePoint[0]), int(endLinePoint[1]-50))
        secondHatchEnd = (int(endLinePoint[0]), int(endLinePoint[1]+50))

    cv2.line(base_img, firstHatchStart, firstHatchEnd,
        (255, 0, 255), 1)

    cv2.line(base_img, secondHatchStart, secondHatchEnd,
        (255, 0, 255), 1)
    '''

    rows,cols = base_img.shape[:2]
    [vx,vy,x,y] = cv2.fitLine(full_contour, cv2.DIST_L2,0,0.01,0.01)
    print("vx: {}".format(vx))
    print("vy: {}".format(vy))
    print("vy/vx: {}".format(vy/vx))
    lefty = int((-x*vy/vx) + y)
    righty = int(((cols-x)*vy/vx)+y)
    b = (cols-1, righty)
    a = (0, lefty)
    slope = abs(float(b[1] - a[1])/float(b[0] - a[0]))
    print("SLOPE:::: {}".format(slope))


    dimB = dB / pixelsPerMetric
    if True:
        cv2.drawContours(base_img,[box],0,(25,25,25),1, offset=(left_offset, top_offset))
        cv2.drawContours(base_img,[contour],0,(125,125,125),4, offset=(left_offset, top_offset))
        cv2.drawContours(base_img,[full_contour],0,(225,225,225),4, offset=(left_offset, top_offset))

        cv2.line(base_img,a,b,(0,255,0),2)

    if draw_text:
            # draw the object sizes on the image
            cv2.putText(base_img, "Lobster",
                (endLinePoint[0]+10, endLinePoint[1]), cv2.FONT_HERSHEY_TRIPLEX,
                1, (255, 255, 255), 1, lineType=cv2.LINE_AA)

            cv2.putText(base_img, "{:.1f}in".format(dimB),
                (endLinePoint[0]+10, endLinePoint[1]+50), cv2.FONT_HERSHEY_TRIPLEX,
                1, (255, 255, 255), 1, lineType=cv2.LINE_AA)

    return dimB, startLinePoint, endLinePoint

def draw_contour(base_img, contour, pixelsPerMetric, pre, draw_text, flipDrawing, rulerWidth):

    if pre == "Square":
        tl, tr, bl, br = get_bounding_corner_points(contour)
    else:
        tl, tr, bl, br = get_corner_points(pre, contour)

    
    if flipDrawing:
        # compute the midpoint between the top-left and top-right points,
        # followed by the midpoint between the top-righ and bottom-right
        startLinePoint = midpoint(tl, tr)
        startLinePoint = (int(startLinePoint[0]), int(startLinePoint[1]))
        endLinePoint = midpoint(bl, br)
        endLinePoint = (int(endLinePoint[0]), int(endLinePoint[1]))
        dB = abs(startLinePoint[1] - endLinePoint[1])
    else:
        # compute the midpoint between the top-left and top-right points,
        # followed by the midpoint between the top-righ and bottom-right
        startLinePoint = midpoint(tl, bl)
        startLinePoint = (int(startLinePoint[0]), int(startLinePoint[1]))
        endLinePoint = midpoint(tr, br)
        endLinePoint = (int(endLinePoint[0]), int(endLinePoint[1]))
        # compute the Euclidean distance between the midpoints

        dB = abs(startLinePoint[0] - endLinePoint[0])
  

    # draw the midpoints on the image
    cv2.circle(base_img, (int(startLinePoint[0]), int(startLinePoint[1])), 2, (255, 0, 0), -1)
    cv2.circle(base_img, (int(endLinePoint[0]), int(endLinePoint[1])), 2, (255, 0, 0), -1)

    # draw lines between the midpoints


    cv2.line(base_img, startLinePoint, endLinePoint,
        (255, 0, 255), 1)

    if flipDrawing:
        firstHatchStart = (int(startLinePoint[0]-50), int(startLinePoint[1]))
        firstHatchEnd = (int(startLinePoint[0]+50), int(startLinePoint[1]))
        secondHatchStart = (int(endLinePoint[0]-50), int(endLinePoint[1]))
        secondHatchEnd = (int(endLinePoint[0]+50), int(endLinePoint[1]))
    else:
        firstHatchStart = (int(startLinePoint[0]), int(startLinePoint[1]-50))
        firstHatchEnd = (int(startLinePoint[0]), int(startLinePoint[1]+50))
        secondHatchStart = (int(endLinePoint[0]), int(endLinePoint[1]-50))
        secondHatchEnd = (int(endLinePoint[0]), int(endLinePoint[1]+50))

    cv2.line(base_img, firstHatchStart, firstHatchEnd,
        (255, 0, 255), 1)

    cv2.line(base_img, secondHatchStart, secondHatchEnd,
        (255, 0, 255), 1)


    # if the pixels per metric has not been initialized, then
    # compute it as the ratio of pixels to supplied metric
    # (in this case, inches)
    if pixelsPerMetric is None:
        pixelsPerMetric = get_width_from_ruler(dB, rulerWidth)
        


    dimB = dB / pixelsPerMetric
    print("db: {}, pixels per: {}; dim b: {}".format(dB, pixelsPerMetric, dimB))
    if draw_text:
        if pre == "Ruler" or pre == "Ellipse":
                # draw the object sizes on the image
            cv2.putText(base_img, "{}: {}in".format("U.S. Quarter",dimB),
                (endLinePoint[0]+10, endLinePoint[1]), cv2.FONT_HERSHEY_TRIPLEX,
                1, (255, 255, 255), 1,lineType=cv2.LINE_AA)
        elif pre == "Square":
            cv2.putText(base_img, "{}: {}in".format("2 in. Square",dimB),
                (endLinePoint[0]+10, endLinePoint[1]), cv2.FONT_HERSHEY_TRIPLEX,
                1, (255, 255, 255), 1,lineType=cv2.LINE_AA)
        else:
            # draw the object sizes on the image
            cv2.putText(base_img, "{}".format(pre),
                (endLinePoint[0]+10, endLinePoint[1]), cv2.FONT_HERSHEY_TRIPLEX,
                1, (255, 255, 255), 1, lineType=cv2.LINE_AA)

            cv2.putText(base_img, "{:.1f}in".format(dimB),
                (endLinePoint[0]+10, endLinePoint[1]+50), cv2.FONT_HERSHEY_TRIPLEX,
                1, (255, 255, 255), 1, lineType=cv2.LINE_AA)

    return pixelsPerMetric, dimB, startLinePoint, endLinePoint