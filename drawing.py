import cv2
import utils
import numpy as np
import math
from scipy.spatial import distance
from imutils import perspective
import constants
import depth_adjuster

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
   
def get_square_points(c):

    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])

    tl = (extLeft[0], extTop[1])
    tr = (extRight[0],extTop[1])
    bl = (extLeft[0],extBot[1])
    br = (extRight[0],extBot[1])
   
    return tl, tr, bl, br

def get_bounding_corner_points(contour):
    brect = cv2.boundingRect(contour)
    #brect = cv2.boxPoints(rect)
    #brect = np.int0(box)
    #print("box: {}".format(brect))
    
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
        widthEndLinePoint = extRight

        dBWidthX = abs(widthStartLinePoint[0] - widthEndLinePoint[0])
        dBWidthY = abs(widthStartLinePoint[1] - widthEndLinePoint[1])
        dBWidth = get_distance(dBWidthX, dBWidthY)
    else:
        startLinePoint = extLeft
        endLinePoint = extRight

        dBX = abs(startLinePoint[0] - endLinePoint[0])
        dBY = abs(startLinePoint[1] - endLinePoint[1])
        dB = get_distance(dBX, dBY)
        if constants.isScallop(fisheryType):
            x,y,w,h = cv2.boundingRect(c)
            cv2.rectangle(base_img,(x,y),(x+w,y+h),(255,255,0),3)
            widthStartLinePoint = extTop
            widthEndLinePoint = (extTop[0],extTop[1]+h)
        else:
            widthStartLinePoint = extTop
            widthEndLinePoint = extBot

        dBWidthX = abs(widthStartLinePoint[0] - widthEndLinePoint[0])
        dBWidthY = abs(widthStartLinePoint[1] - widthEndLinePoint[1])
        dBWidth = get_distance(dBWidthX, dBWidthY)

    drawLines(base_img, not flipDrawing, startLinePoint, endLinePoint, False)
    if True:
        cv2.circle(base_img, widthStartLinePoint, 8, (0, 0, 255), -1)
        cv2.circle(base_img, widthEndLinePoint, 8, (0, 255, 0), -1)
        cv2.drawContours(base_img,[c],0,(255,0,0),2)
        drawLines(base_img, not flipDrawing, widthStartLinePoint, widthEndLinePoint, False)
    
    #this is the rotated box
    if False:
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(base_img,[box],0,(0,191,255),8)
    
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

        if(constants.isScallop(fisheryType)):
            unitStr = "cm"
            widthToShow = dimBWidth
        else:
            unitStr = "in"
            widthToShow = dimB
        
        cv2.putText(base_img, "{:.1f}{}".format(widthToShow,unitStr),
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
        dBX = abs(startLinePoint[1] - endLinePoint[1])
        dBY = abs(startLinePoint[0] - endLinePoint[0])
        dB = get_distance(dBX, dBY)

        widthStartLinePoint = midpoint(tl, bl)
        widthStartLinePoint = (int(widthStartLinePoint[0]), int(widthStartLinePoint[1]))
        widthEndLinePoint = midpoint(tr, br)
        widthEndLinePoint = (int(widthEndLinePoint[0]), int(widthEndLinePoint[1]))
        dBWidthX = abs(widthStartLinePoint[1] - widthEndLinePoint[1])
        dBWidthY = abs(widthStartLinePoint[0] - widthEndLinePoint[0])
        dBWidth = get_distance(dBWidthX, dBWidthY)
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
        quarterStartLinePoint = midpoint(qtl, qtr)
        quarterStartLinePoint = (int(quarterStartLinePoint[0]), int(quarterStartLinePoint[1]))
        quarterEndLinePoint = midpoint(qbl, qbr)
        quarterEndLinePoint = (int(quarterEndLinePoint[0]), int(quarterEndLinePoint[1]))
        dB = abs(quarterStartLinePoint[1] - quarterEndLinePoint[1])

    else:
        # compute the midpoint between the top-left and top-right points,
        # followed by the midpoint between the top-righ and bottom-right

        quarterStartLinePoint = midpoint(qtl, qbl)
        quarterStartLinePoint = (int(quarterStartLinePoint[0]), int(quarterStartLinePoint[1]))
        quarterEndLinePoint = midpoint(qtr, qbr)
        quarterEndLinePoint = (int(quarterEndLinePoint[0]), int(quarterEndLinePoint[1]))
        dB = abs(quarterStartLinePoint[0] - quarterEndLinePoint[0])

    # draw the midpoints on the image


    cv2.circle(base_img, (int(quarterStartLinePoint[0]), int(quarterStartLinePoint[1])), 3, (0, 255, 0), -1)
    cv2.circle(base_img, (int(quarterEndLinePoint[0]), int(quarterEndLinePoint[1])), 3, (0, 255, 0), -1)
    #cv2.circle(base_img, (int(quarterCenterX), int(quarterCenterY)), int(refWidth/2), (255, 255, 0), 2)

    # draw lines between the midpoints
    #drawLines(base_img, flipDrawing, startLinePoint, endLinePoint)
    drawLines(base_img, flipDrawing, quarterStartLinePoint, quarterEndLinePoint, False)


    # if the pixels per metric has not been initialized, then
    # compute it as the ratio of pixels to supplied metric
    # (in this case, inches)

    #slight variations based on zoom level...

    #compensating for distance between abalone and quarter on board
    # the farther away it is (smaller the quarter) the less it compensates,
    # since the percentage diff between quarter distance and abalone edge (from camera)
    # is smaller...    
    multiplier = depth_adjuster.get_multiplier_from_db(fishery_type, dB)

    pixelsPerMetric = get_width_from_ruler(refWidth, refObjectSize)
    pixelsPerMetric = pixelsPerMetric*multiplier
    dimB = dB / pixelsPerMetric

    if draw_text:
        # draw the quarter
        cv2.putText(base_img, "{}: {}in".format("U.S. Quarter",0.955),
            (quarterEndLinePoint[0]+10, quarterEndLinePoint[1]), cv2.FONT_HERSHEY_TRIPLEX,
            1, (255, 255, 255), 1,lineType=cv2.LINE_AA)

    return pixelsPerMetric, dimB, quarterStartLinePoint, quarterEndLinePoint

def draw_square_contour(base_img, contour, pixelsPerMetric, draw_text, flipDrawing, refObjectSize, refObjectUnits, 
        fishery_type):

    
    #tl, tr, bl, br = get_bounding_corner_points(contour)
    #print("tl: {}".format(tl))
    #TODO: instead of using the bounding box, intersect this line and the contour line, use those points instead
    #tl, tr, bl, br = get_square_points(contour)
    minRect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(minRect)
    
    box = np.array(box, dtype="int")
    
    cv2.drawContours(base_img, [box],0,(125,0,0),3)
    box = perspective.order_points(box)

    tl = tuple(box[0])
    tr = tuple(box[1])
    br = tuple(box[2])
    bl = tuple(box[3])
    
    startLinePoint = midpoint(tl, bl)
    startLinePoint = (int(startLinePoint[0])+2, int(startLinePoint[1]))
    endLinePoint = midpoint(tr, br)
    endLinePoint = (int(endLinePoint[0])-2, int(endLinePoint[1]))
    #cornerPoints = get_square_corners(base_img, contour)

    startLinePoint, endLinePoint = get_contour_line_intersection(base_img, contour, [startLinePoint, endLinePoint], startLinePoint, endLinePoint)

    dBX = abs(startLinePoint[0] - endLinePoint[0])
    dBY =  abs(startLinePoint[1] - endLinePoint[1])
    dB = get_distance(dBX, dBY)
    print("dB: {}".format(dB))
    #compensating for distance between abalone and quarter on board
    # the farther away it is (smaller the quarter) the less it compensates,
    # since the percentage diff between quarter distance and abalone edge (from camera)
    # is smaller...    

    multiplier = depth_adjuster.get_multiplier_from_db(fishery_type, dB)

    pixelsPerMetric = get_width_from_ruler(dB, refObjectSize)
   
    pixelsPerMetric = pixelsPerMetric*multiplier
    dimB = dB / pixelsPerMetric
    
    if True:
        cv2.drawContours(base_img, [contour],0,(125,0,0),1)
        cv2.circle(base_img, tl, 4, (255, 0, 0), -1)
        cv2.circle(base_img, tr, 4, (255, 0, 0), -1)
        cv2.circle(base_img, bl, 4, (255, 0, 0), -1)
        cv2.circle(base_img, br, 4, (255, 0, 0), -1)

    cv2.circle(base_img, (int(startLinePoint[0]), int(startLinePoint[1])), 1, (255, 0, 0), -1)
    cv2.circle(base_img, (int(endLinePoint[0]), int(endLinePoint[1])), 1, (255, 0, 0), -1)

    # draw lines between the midpoints
    cv2.line(base_img, startLinePoint, endLinePoint,
        (255, 0, 255), 1)

    if draw_text:
        cv2.putText(base_img, "{}{} Square".format(refObjectSize, refObjectUnits),
            (endLinePoint[0]+10, endLinePoint[1]), cv2.FONT_HERSHEY_TRIPLEX,
            1, (255, 255, 255), 1,lineType=cv2.LINE_AA)
    return pixelsPerMetric, dimB, startLinePoint, endLinePoint

def draw_target_finfish_contour(base_img, contour, pixelsPerMetric, draw_text, left_offset, top_offset):
    
    rotRect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rotRect)
    box = np.int0(box)

    startLinePoint, endLinePoint = get_middle_of_min_rect(base_img, contour)

    dB = distance.euclidean(startLinePoint, endLinePoint)
   
    cv2.circle(base_img, (int(startLinePoint[0]), int(startLinePoint[1])), 2, (255, 0, 255), -1)
    cv2.circle(base_img, (int(endLinePoint[0]), int(endLinePoint[1])), 2, (255, 0, 255), -1)

    dimB = dB / pixelsPerMetric
    if True:
        #box = np.int0(box)
        #cv2.drawContours(base_img,[box],0,(125,225,225),4, offset=(left_offset, top_offset))
        #cv2.drawContours(base_img,[contour],0,(125,125,125),4, offset=(left_offset, top_offset))
        cv2.circle(base_img, (startLinePoint[0], startLinePoint[1]), 10, (50, 50, 255), -1)
        cv2.circle(base_img, (endLinePoint[0], endLinePoint[1]), 10, (50, 50, 255), -1)
        cv2.line(base_img,startLinePoint,endLinePoint,(0,255,0),2)

    #utils.show_img("intersection...", img3)
    if draw_text:
            # draw the object sizes on the image
            cv2.putText(base_img, "Finfish",
                (endLinePoint[0]+10, endLinePoint[1]), cv2.FONT_HERSHEY_TRIPLEX,
                1, (255, 255, 255), 1, lineType=cv2.LINE_AA)

            cv2.putText(base_img, "{:.2f}in".format(dimB),
                (endLinePoint[0]+10, endLinePoint[1]+50), cv2.FONT_HERSHEY_TRIPLEX,
                1, (255, 255, 255), 1, lineType=cv2.LINE_AA)

    # draw lines between the midpoints
    cv2.line(base_img, startLinePoint, endLinePoint,
        (255, 0, 255), 4)

    return dimB, startLinePoint, endLinePoint

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
    #[vx,vy,x,y] = cv2.fitLine(full_contour, cv2.DIST_L2,0,0.01,0.01)
    ellipse = cv2.fitEllipse(full_contour)
    (ex,ey),(eMA,ema),eAngle = cv2.fitEllipse(full_contour)
    #trimmed_lobster_contour = contour_utils.trim_lobster_contour(full_contour, (ex,ey), (eMA, ema), eAngle)
    
    isHorizontal = 45.0 <= eAngle <= 135.0
   
    poly = cv2.ellipse2Poly((int(ellipse[0][0]), int(ellipse[0][1])), (int(ellipse[1][0] / 2), int(ellipse[1][1] / 2)), int(ellipse[2]), 0, 360, 5)
    cv2.ellipse(base_img, ellipse, (255,0,255), 2,cv2.LINE_AA)
    [vx,vy,x,y] = cv2.fitLine(poly, cv2.DIST_L2,0,0.01,0.01)

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
   
    cv2.circle(base_img, (int(startLinePoint[0]), int(startLinePoint[1])), 2, (255, 0, 255), -1)
    cv2.circle(base_img, (int(endLinePoint[0]), int(endLinePoint[1])), 2, (255, 0, 255), -1)

    dimB = dB / pixelsPerMetric
    if True:
        box = np.int0(box)
        cv2.drawContours(base_img,[box],0,(125,225,225),4, offset=(left_offset, top_offset))
        cv2.drawContours(base_img,[contour],0,(125,125,125),4, offset=(left_offset, top_offset))
        cv2.drawContours(base_img,[full_contour],0,(225,225,225),4, offset=(left_offset, top_offset))
        cv2.circle(base_img, (cX, cY), 10, (50, 50, 255), -1)
        cv2.line(base_img,a,b,(0,255,0),2)


    line = [a,b]
    startLinePoint, endLinePoint = get_contour_line_intersection(base_img, contour, line, startLinePoint, endLinePoint)

    #utils.show_img("intersection...", img3)
    if draw_text:
            # draw the object sizes on the image
            cv2.putText(base_img, "Lobster",
                (endLinePoint[0]+10, endLinePoint[1]), cv2.FONT_HERSHEY_TRIPLEX,
                1, (255, 255, 255), 1, lineType=cv2.LINE_AA)

            cv2.putText(base_img, "{:.2f}in".format(dimB),
                (endLinePoint[0]+10, endLinePoint[1]+50), cv2.FONT_HERSHEY_TRIPLEX,
                1, (255, 255, 255), 1, lineType=cv2.LINE_AA)

    # draw lines between the midpoints
    cv2.line(base_img, startLinePoint, endLinePoint,
        (255, 0, 255), 4)

    return dimB, startLinePoint, endLinePoint

def get_middle_of_min_rect(base_img, contour):
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    (x,y,) , (width, height),angle = rect
    rotAngle = abs(angle)
    [tl, tr, br, bl] = clockwise_points(box)
    #check the rot angle because the width/height in the rectangle changes based on rotation
    if rotAngle > 45:
        width = rect[1][1]
        height = rect[1][0]
    else:
        width = rect[1][0]
        height = rect[1][1]
    
    #if the fish is vertical instead of horizontal, flip the points
    if width >= height:
        leftPoint = midpoint(tl, bl)
        rightPoint = midpoint(tr,br)
    else:
        leftPoint = midpoint(tl, tr)
        rightPoint = midpoint(bl, br)
    startPoint = (int(leftPoint[0]), int(leftPoint[1]))
    endPoint = (int(rightPoint[0]), int(rightPoint[1]))
    return startPoint, endPoint

def get_far_points(base_img, c):
    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBottom = tuple(c[c[:, :, 1].argmax()][0])
    rect = cv2.minAreaRect(c)
    width = rect[1][0]
    height = rect[1][1]
    if width >= height:
        return extLeft, extRight
    else:
        return extTop, extBottom
        
def get_contour_rect_intersection(base_img, contour):
        #find the union of the fitted line for the entire lobster contour
    # and the target contour
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    rect = cv2.boundingRect(contour)


    # create an image filled with zeros, single-channel, same size as img.
    blank = np.zeros( base_img.shape[0:2] )

    # copy each of the contours (assuming there's just two) to its own image. 
    # Just fill with a '1'.
    img1 = cv2.drawContours( blank.copy(), [contour], 0, 2 )

    x,y,w,h = rect
    img2 = cv2.rectangle(base_img,(x,y),(x+w,y+h),(0,255,0),2)
    #img2 = cv2.drawContours(blank.copy(),[box],0, 5)
    utils.show_img("bounding", img2)

    #and AND them together
    imgI = np.logical_and(img1, img2)
    locations = np.argwhere(imgI)

    '''
    if len(locations) > 2:
        locations = remove_duplicates(locations)
    '''
    if locations is not None and len(locations) >= 2:
        
        if True:
            for i, loc in enumerate(locations):
                cX = loc[1]
                cY = loc[0]

                cv2.circle(base_img, (cX, cY), 12, (125,125,125), -1)
        width = rect[1][0]
        height = rect[1][1]
        
        xMinDex = np.argmin(locations[:,1:])
        xMaxDex = np.argmax(locations[:,1:])
        
        if width >= height:
            leftMost = locations[xMinDex]
            rightMost = locations[xMaxDex]


            startLinePoint = (leftMost[1], leftMost[0])
            endLinePoint = (rightMost[1],rightMost[0])
            if True:
                cv2.circle(base_img, startLinePoint, 12, (255,0,0), -1)
                cv2.circle(base_img, endLinePoint, 12, (255,0,0), -1)
        else:
            print("its taller")
        
        color =  (50, 50, 255)
        for i, loc in enumerate(locations):
            cX = loc[1]
            cY = loc[0]
            color = (i*30,i*30,i*25)
            if(i == 0):
                startLinePoint = (cX, cY)
            elif i == 1:
                endLinePoint = (cX, cY)

        
    elif len(locations) == 2:
        startLinePoint = (locations[0][1],locations[0][1])
        endLinePoint = (locations[1][1], locations[1][0])
    else:
        print("no start or end line point")
        startLinePoint = 0
        endLinePoint = 0


    cv2.circle(base_img, startLinePoint, 10,(150,0,5),-1)
    cv2.circle(base_img,endLinePoint, 10, (0,150,5),-1)
    return startLinePoint, endLinePoint    


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
    img2 = cv2.drawContours( blank.copy(), [line_contour], 0, 1,2 )

    #and AND them together
    imgI = np.logical_and(img1, img2)

    #see where they're true (the intersections)
    locations = np.argwhere(imgI)

    if len(locations) > 2:
        locations = remove_duplicates(locations)

    if locations is not None and len(locations) >= 2:
        color =  (50, 50, 255)
        for i, loc in enumerate(locations):
            cX = loc[1]
            cY = loc[0]
            color = (i*30,i*30,i*25)
            if(i == 0):
                startLinePoint = (cX, cY)
            elif i == 1:
                endLinePoint = (cX, cY)
            
            if True:
                cv2.circle(base_img, (cX, cY), 12, (125,125,125), -1)

    return startLinePoint, endLinePoint

def is_blank(location):
    return location[0] == -1 and location[1] == -1

def close_enough(orig_value, new_value):
    if is_blank(new_value):
        return False
    offset = 10
    if (orig_value[0]-offset <= new_value[0] <= orig_value[0]+offset) and (orig_value[1]-offset <= new_value[1] <= orig_value[1]+offset):
        return True
    else:
        return False

def remove_duplicates(locations):
    for i, loc in enumerate(locations):
        if is_blank(loc):
            continue
        nextDex = i+1
        for j in range(nextDex, len(locations)):
            curr_loc = locations[j]
            if close_enough(loc, curr_loc):
                locations[j] = [-1,-1]
    
    results = []
    for loc in locations:
        if not is_blank(loc):
            results.append(loc)

    return results


def get_square_corners(base_img, contour):
    rotRect = cv2.minAreaRect(contour)

    box = cv2.boxPoints(rotRect)
    box = np.array(box, dtype="int")
    rect = clockwise_points(box)
    colors = ((0, 0, 255), (240, 0, 159), (255, 0, 0), (255, 255, 0))
    for ((x, y), color) in zip(rect, colors):
	    cv2.circle(base_img, (int(x), int(y)), 5, color, -1)

    cv2.drawContours(base_img, [box], -1, (220, 255, 225), 1)

    return rect
    
#start in topleft
def clockwise_points(pts):
	sortedPts = pts[np.argsort(pts[:, 0]), :]

	left = sortedPts[:2, :]
	right = sortedPts[2:, :]

	left = left[np.argsort(left[:, 1]), :]
	(tl, bl) = left
	euclideanDist = distance.cdist(tl[np.newaxis], right, "euclidean")[0]
	(br, tr) = right[np.argsort(euclideanDist)[::-1], :]

	return [tl, tr, br, bl]

def draw_lobster_contour(base_img, contour, pixelsPerMetric, draw_text, flipDrawing, rulerWidth, left_offset, top_offset, full_contour):
    #center (x,y), (width, height), angle of rotation 
    
    rotRect = cv2.minAreaRect(contour)
    width = rotRect[1][0]
    height = rotRect[1][1]
    rotAngle = abs(rotRect[2])
    verts = cv2.boxPoints(rotRect)
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

 
    box = cv2.boxPoints(rotRect)

    #convert from floats to int
    box = np.int0(box)
    cv2.drawContours(base_img,[box],0,(25,25,25),1, offset=(left_offset, top_offset))
    
    flipLine = (rotAngle > 45 and width > height) or (rotAngle < 45 and width < height)

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

    cv2.circle(base_img, (int(startLinePoint[0]), int(startLinePoint[1])), 4, (255, 0, 255), -1)
    cv2.circle(base_img, (int(endLinePoint[0]), int(endLinePoint[1])), 4, (255, 0, 255), -1)

    # draw lines between the midpoints
    cv2.line(base_img, startLinePoint, endLinePoint,
        (255, 0, 255), 4)


    rows,cols = base_img.shape[:2]
    [vx,vy,x,y] = cv2.fitLine(full_contour, cv2.DIST_L2,0,0.01,0.01)

    lefty = int((-x*vy/vx) + y)
    righty = int(((cols-x)*vy/vx)+y)
    b = (cols-1, righty)
    a = (0, lefty)
    slope = abs(float(b[1] - a[1])/float(b[0] - a[0]))


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