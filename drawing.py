import cv2
import utils
import numpy as np

def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

def get_width_from_ruler(dB, rulerWidth):
    return (dB)/float(rulerWidth)

def get_corner_points(pre, contour):
    print("pre is {}".format(pre))
    if pre == "Ruler":
        '''
        cX, cY = utils.get_centroid(contour)
        quarter_ellipse = cv2.fitEllipse(contour)
        pts = cv2.boxPoints(quarter_ellipse)
        size = quarter_ellipse[1]
        print("size: {}".format(size))


        w = int(size[0])
        h = int(size[1])
        center = np.array([cX, cY])
        radius = (w/2)-2
        qcon = np.squeeze(contour)

        xmin = center[0]-((int(w/2)+2))
        xmax = center[0]+((int(w/2))+2)
        for pt in qcon:
            if pt[0] < xmin:
                pt[0] = xmin
            elif pt[0] > xmax:
                pt[0] = xmax
        
                print("width: {}, height: {}".format(w, h))
        #for testing
        brect = cv2.boundingRect(contour)
        bwidth=brect[2]
        bheight=brect[3]
        print("bounding width: {}, bounding height: {}".format(bwidth, bheight))
        width = w
        height = h
        tl = (cX-radius, cY-radius)
        tr = (cX+radius, cY-radius)
        bl = (cX-radius, cY+radius)
        br = (cX+radius, cY+radius)
        '''
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
        
    else:

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

def draw_both_contours(base_img, contour, draw_text, flipDrawing, quarterCenterX, quarterCenterY, quarterRadius):
    rulerWidth = 0.955
    tl, tr, bl, br = get_corner_points("Abalone", contour)
    qtl, qtr, qbl, qbr = get_quarter_corners(quarterCenterX, quarterCenterY, quarterRadius)
    print("quarter corners: {}, {}, {}, {}: radius: {}".format(qtl, qtr, qbl, qbr, quarterRadius))
    if flipDrawing:
        # compute the midpoint between the top-left and top-right points,
        # followed by the midpoint between the top-righ and bottom-right
        startLinePoint = midpoint(tl, tr)
        startLinePoint = (int(startLinePoint[0]), int(startLinePoint[1]))
        endLinePoint = midpoint(bl, br)
        endLinePoint = (int(endLinePoint[0]), int(endLinePoint[1]))
        dB = abs(startLinePoint[1] - endLinePoint[1])

        quarterStartLinePoint = midpoint(qtl, qtr)
        quarterStartLinePoint = (int(quarterStartLinePoint[0]), int(quarterStartLinePoint[1]))
        quarterEndLinePoint = midpoint(qbl, qbr)
        quarterEndLinePoint = (int(quarterEndLinePoint[0]), int(quarterEndLinePoint[1]))



    else:
        # compute the midpoint between the top-left and top-right points,
        # followed by the midpoint between the top-righ and bottom-right
        startLinePoint = midpoint(tl, bl)
        startLinePoint = (int(startLinePoint[0]), int(startLinePoint[1]))
        endLinePoint = midpoint(tr, br)
        endLinePoint = (int(endLinePoint[0]), int(endLinePoint[1]))
        # compute the Euclidean distance between the midpoints
        dB = abs(startLinePoint[0] - endLinePoint[0])
    

        quarterStartLinePoint = midpoint(qtl, qbl)
        quarterStartLinePoint = (int(quarterStartLinePoint[0]), int(quarterStartLinePoint[1]))
        quarterEndLinePoint = midpoint(qtr, qbr)
        quarterEndLinePoint = (int(quarterEndLinePoint[0]), int(quarterEndLinePoint[1]))

    # draw the midpoints on the image
    cv2.circle(base_img, (int(startLinePoint[0]), int(startLinePoint[1])), 2, (255, 0, 0), -1)
    cv2.circle(base_img, (int(endLinePoint[0]), int(endLinePoint[1])), 2, (255, 0, 0), -1)

    cv2.circle(base_img, (int(quarterStartLinePoint[0]), int(quarterStartLinePoint[1])), 2, (0, 255, 0), -1)
    cv2.circle(base_img, (int(quarterEndLinePoint[0]), int(quarterEndLinePoint[1])), 2, (0, 255, 0), -1)


    # draw lines between the midpoints


    drawLines(base_img, flipDrawing, startLinePoint, endLinePoint)
    drawLines(base_img, flipDrawing, quarterStartLinePoint, quarterEndLinePoint)


    # if the pixels per metric has not been initialized, then
    # compute it as the ratio of pixels to supplied metric
    # (in this case, inches)

    pixelsPerMetric = get_width_from_ruler(quarterRadius*2, rulerWidth)
    dimB = dB / pixelsPerMetric

    print("db: {}, pixels per: {}; dim b: {}".format(dB, pixelsPerMetric, dimB))
    if draw_text:

        # draw the quarter
        cv2.putText(base_img, "{}: {}in".format("U.S. Quarter",0.955),
            (quarterEndLinePoint[0]+10, quarterEndLinePoint[1]), cv2.FONT_HERSHEY_TRIPLEX,
            1, (255, 255, 255), 1,lineType=cv2.LINE_AA)
        

        # draw the object sizes on the image
        cv2.putText(base_img, "Abalone",
            (endLinePoint[0]+10, endLinePoint[1]), cv2.FONT_HERSHEY_TRIPLEX,
            1, (255, 255, 255), 1, lineType=cv2.LINE_AA)

        cv2.putText(base_img, "{:.1f}in".format(dimB),
            (endLinePoint[0]+10, endLinePoint[1]+50), cv2.FONT_HERSHEY_TRIPLEX,
            1, (255, 255, 255), 1, lineType=cv2.LINE_AA)

    return pixelsPerMetric, dimB, startLinePoint, endLinePoint, quarterStartLinePoint, quarterEndLinePoint



def draw_contour(base_img, contour, pixelsPerMetric, pre, draw_text, flipDrawing):
    rulerWidth = 0.955
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
        else:
            # draw the object sizes on the image
            cv2.putText(base_img, "{}".format(pre),
                (endLinePoint[0]+10, endLinePoint[1]), cv2.FONT_HERSHEY_TRIPLEX,
                1, (255, 255, 255), 1, lineType=cv2.LINE_AA)

            cv2.putText(base_img, "{:.1f}in".format(dimB),
                (endLinePoint[0]+10, endLinePoint[1]+50), cv2.FONT_HERSHEY_TRIPLEX,
                1, (255, 255, 255), 1, lineType=cv2.LINE_AA)

    return pixelsPerMetric, dimB, startLinePoint, endLinePoint