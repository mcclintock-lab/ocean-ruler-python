import cv2
import utils
import numpy as np
import math
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

def drawLines(base_img, flipDrawing, startLinePoint, endLinePoint):
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

def get_quarter_corners(quarterCenterX, quarterCenterY, quarterRadius):

    tl = (quarterCenterX-quarterRadius, quarterCenterY-quarterRadius)
    tr = (quarterCenterX+quarterRadius, quarterCenterY-quarterRadius)
    bl = (quarterCenterX-quarterRadius, quarterCenterY+quarterRadius)
    br = (quarterCenterX+quarterRadius, quarterCenterY+quarterRadius)
    return tl, tr, bl, br


def draw_target_contour(base_img, contour, draw_text, flipDrawing, pixelsPerMetric):
    
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

        '''
        quarterStartLinePoint = midpoint(qtl, qtr)
        quarterStartLinePoint = (int(quarterStartLinePoint[0]), int(quarterStartLinePoint[1]))
        quarterEndLinePoint = midpoint(qbl, qbr)
        quarterEndLinePoint = (int(quarterEndLinePoint[0]), int(quarterEndLinePoint[1]))
        '''
    else:
        # compute the midpoint between the top-left and top-right points,
        # followed by the midpoint between the top-righ and bottom-right
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
        '''

    # draw the midpoints on the image
    cv2.circle(base_img, (int(startLinePoint[0]), int(startLinePoint[1])), 2, (255, 0, 0), -1)
    cv2.circle(base_img, (int(endLinePoint[0]), int(endLinePoint[1])), 2, (255, 0, 0), -1)

    '''
    cv2.circle(base_img, (int(quarterStartLinePoint[0]), int(quarterStartLinePoint[1])), 2, (0, 255, 0), -1)
    cv2.circle(base_img, (int(quarterEndLinePoint[0]), int(quarterEndLinePoint[1])), 2, (0, 255, 0), -1)
    '''

    # draw lines between the midpoints
    drawLines(base_img, flipDrawing, startLinePoint, endLinePoint)
    #drawLines(base_img, flipDrawing, quarterStartLinePoint, quarterEndLinePoint)


    # if the pixels per metric has not been initialized, then
    # compute it as the ratio of pixels to supplied metric
    # (in this case, inches)

    dimB = dB / pixelsPerMetric

    if draw_text:

        # draw the object sizes on the image
        cv2.putText(base_img, "Abalone",
            (endLinePoint[0]+10, endLinePoint[1]), cv2.FONT_HERSHEY_TRIPLEX,
            1, (255, 255, 255), 1, lineType=cv2.LINE_AA)

        cv2.putText(base_img, "{:.1f}in".format(dimB),
            (endLinePoint[0]+10, endLinePoint[1]+50), cv2.FONT_HERSHEY_TRIPLEX,
            1, (255, 255, 255), 1, lineType=cv2.LINE_AA)

    return dimB, startLinePoint, endLinePoint



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
    drawLines(base_img, flipDrawing, quarterStartLinePoint, quarterEndLinePoint)


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

def draw_lobster_contour(base_img, contour, pixelsPerMetric, draw_text, flipDrawing, rulerWidth, left_offset, top_offset):
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

    if rotRect < 45:
        #width is longer side
        #shows as x
        print("long side is width")
    else:
        #height is longer side
        #shows as y
        print("x is short side")
    box = cv2.boxPoints(rotRect)
    print("rotRect: {}".format(rotRect))
    print("box: {}".format(box))

    #convert from floats to int
    box = np.int0(box)
    cv2.drawContours(base_img,[box],0,(25,25,25),1, offset=(left_offset, top_offset))
    
    print("width: ", width)
    print("height: ", height)

    print("----------->>>>> tl:{}, tr:{}, br:{}, bl:{}".format(tl, tr, br, bl))
    #calculate hypotenuse
    a = abs(tl[0] - tr[0])
    b = abs(tl[1] - tr[1])
    dB = math.sqrt(math.pow(a,2)+math.pow(b,2))

    #tl, tr, bl, br = get_bounding_corner_points(contour)
    tl = (tl[0]+left_offset, tl[1]+top_offset)
    tr = (tr[0]+left_offset, tr[1]+top_offset)
    bl = (bl[0]+left_offset, bl[1]+top_offset)
    br = (br[0]+left_offset, br[1]+top_offset)
    print("----------->>>>> tl:{}, tr:{}, br:{}, bl:{}".format(tl, tr, br, bl))
    flipLine = rotAngle > 45 and width > height
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
    print("a size is {}".format(a))
    print("b side is {}".format(b))
    print("hypotenuse is:::::: {}".format(dB))
    print("pixels per metrix is: {}".format(pixelsPerMetric))
    dimB = dB / pixelsPerMetric
    
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