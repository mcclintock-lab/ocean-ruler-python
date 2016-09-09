# import the necessary packages
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2
import csv
import os

def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

def get_ruler_shape(template_contours):
    size = 10000000
    targetDex = 0
    for i, con in enumerate(template_contours):
        carea = cv2.contourArea(con)
        print("area: ", carea, "for index: ", i)
        if carea < size:
            size = carea
            targetDex = i

    print("ruler target dex is ", targetDex, " with size ", size)
    return template_contours[targetDex]

def get_abalone_shape(template_contours):
    size = 0
    targetDex = 0
    for i, con in enumerate(template_contours):
        carea = cv2.contourArea(con)
        if carea > size:
            size = carea
            targetDex = i

    print("abalone target dex is ", targetDex, " with size ", size)
    return template_contours[targetDex]

def sort_by_matching_shape(contours, template_contours):
    abalone_shape = get_abalone_shape(template_contours)
    ruler_shape = get_ruler_shape(template_contours)
    rulerDex = 0
    abaloneDex = 0
    minRulerVal = 1000
    minAbaloneVal = 1000
    for i, image_contour in enumerate(contours):
        ab_val = cv2.matchShapes(image_contour,abalone_shape,1,0.0)
        print ("ab val: ", ab_val, " for dex: ", i)
        if ab_val < minAbaloneVal:
            minAbaloneVal = ab_val
            abaloneDex = i

        ruler_val = cv2.matchShapes(image_contour,ruler_shape,1,0.0)
        print("ruler val: ", ruler_val, " for dex ", i)
        if ruler_val < minRulerVal:
            minRulerVal = ruler_val
            rulerDex = i

    print("abalone dex: ", abaloneDex, "ruler dex: ", rulerDex)
    return [contours[rulerDex], contours[abaloneDex]]

        
def sort_contours(cnts, method="left-to-right"):
    # initialize the reverse flag and sort index
    reverse = False
    i = 0
 
    # handle if we need to sort in reverse
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
 
    # handle if we are sorting against the y-coordinate rather than
    # the x-coordinate of the bounding box
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1
 
    # construct the list of bounding boxes and sort them from top to
    # bottom
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
        key=lambda b:b[1][i], reverse=reverse))
 
    # return the list of sorted contours and bounding boxes
    print("counts::::::: ", cnts[0:2])
    return (cnts, boundingBoxes)

def get_width_from_ruler(dB, rulerWidth):
    return dB/float(rulerWidth)

def get_edges(cnts, orig):
    if orig:
        return cnts
    else:
        abalone = 0
        ruler = 0
        max_size = 0
        for i, contour in enumerate(cnts):
            carea = cv2.contourArea(contour)
            print("area for contour: ", carea, "for index: ", i)
            if carea > max_size:
                max_size = carea
                abalone = i
        max_size = 0
        for i,contour in enumerate(cnts):
            carea = cv2.contourArea(contour)
            if carea > max_size and i != abalone:
                max_size = carea
                ruler = i

        abalone_and_ruler = [cnts[ruler], cnts[abalone]]
        return abalone_and_ruler

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument('image', metavar='fp', nargs='+', help='file names')
ap.add_argument("-w", "--width", required=False,
    help="width of ruler. defaults to 8.0 inches")
args = vars(ap.parse_args())

showResults = False
imageName = args['image'][0]
imageParts = imageName.split()
if(len(imageParts) > 1):
    imageName = "{} {}".format(imageParts[0], imageParts[1])
print("imageName: ", imageName)

rulerWidth = args["width"]
if not rulerWidth:
    print("no ruler width, default to 8.0")
    rulerWidth = 8.0

# load the image, convert it to grayscale, and blur it slightly
image_full = cv2.imread(imageName)
if(len(image_full) > 500):
    image = cv2.resize( image_full, (0,0), fx = 0.25, fy = 0.25)
else:
    image = image_full

rows = len(image)
cols = len(image[0])    

hist = cv2.calcHist([image],[0],None,[10],[0,256])
print(hist)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

gray = cv2.GaussianBlur(gray, (7, 7), 0)
#cv2.imshow("Image", gray)
#cv2.waitKey(0)

#assumes the abalone is centered
mid_row_start = int(rows/2) - 70
mid_col_start = int(cols/2) - 70

mid_row_end = mid_row_start+100
mid_col_end = mid_col_start+100


mid_patch = gray[mid_row_start:mid_row_end, mid_col_start:mid_col_end]
mn = np.mean(mid_patch) 


retval, thresh1 = cv2.threshold(gray,mn,255,cv2.THRESH_BINARY)
#cv2.imshow("Image", thresh1)
#cv2.waitKey(0)


template = cv2.imread("../template.jpg")
#ret, temp_thresh = cv2.threshold(template, 1, 1,0)
#cv2.imshow("temp", temp_thresh)
#cv2.waitKey(0)

template_edged = cv2.Canny(template, 50, 100)
#edged = cv2.dilate(edged, None, iterations=1)
#cv2.imshow("Image", template_edged)
#cv2.waitKey(0)

#edged = cv2.erode(edged, None, iterations=1)
#cv2.imshow("Image", edged)
#cv2.waitKey(0)
template_shapes = cv2.findContours(template_edged, 2,1)
template_contours = template_shapes[0] if imutils.is_cv2() else template_shapes[1]
#template_abalone_and_ruler = get_edges(template_contours, False)


# perform edge detection, then perform a dilation + erosion to
# close gaps in between object edges
edged = cv2.Canny(thresh1, 50, 100)
edged = cv2.dilate(edged, None, iterations=1)
#cv2.imshow("Image", edged)
#cv2.waitKey(0)

edged = cv2.erode(edged, None, iterations=1)
#cv2.imshow("Image", edged)
#cv2.waitKey(0)


# find contours in the edge map
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]



# sort the contours from left-to-right and initialize the
# 'pixels per metric' calibration variable
final_contours = None
final_contours = get_edges(cnts, False)

#(template_contours, _) = contours.sort_contours(template_contours)
pixelsPerMetric = None

abalone_and_ruler = sort_by_matching_shape(final_contours, template_contours)
template_abalone_and_ruler = [get_ruler_shape(template_contours), get_abalone_shape(template_contours)]


# loop over the contours individually
for i, con in enumerate(abalone_and_ruler):
    carea = cv2.contourArea(con)

    # compute the rotated bounding box of the contour
    orig = image.copy()
    box = cv2.minAreaRect(con)
    brect = cv2.boundingRect(con)


    #box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
    box = cv2.boxPoints(box)
    box = np.array(box, dtype="int")
    brect_arr = np.array(brect, dtype="int")


    # order the points in the contour such that they appear
    # in top-left, top-right, bottom-right, and bottom-left
    # order, then draw the outline of the rotated bounding
    # box
    #box = perspective.order_points(box)
    #cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)

    # loop over the original points and draw them
    #for (x, y) in box:
    #   cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)

    # unpack the ordered bounding box, then compute the midpoint
    # between the top-left and top-right coordinates, followed by
    # the midpoint between bottom-left and bottom-right coordinates
    x = brect[0]
    y=brect[1]
    width=brect[2]
    height=brect[3]
    tl = (x, y+height)
    tr = (x+width, y+height)
    bl = (x,y)
    br = (x+width, y)
    corners = [tl, tr, br, bl]

    print("box: ", [box.astype("int")])
    print("corners: ", [corners])
    #cv2.drawContours(orig, [corners], -1, (0, 255, 0), 2)

    # loop over the original points and draw them
    #for (x, y) in corners:
    #   cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)

    (tltrX, tltrY) = midpoint(tl, tr)
    (blbrX, blbrY) = midpoint(bl, br)

    # compute the midpoint between the top-left and top-right points,
    # followed by the midpoint between the top-righ and bottom-right
    (tlblX, tlblY) = midpoint(tl, bl)
    (trbrX, trbrY) = midpoint(tr, br)

    # draw the midpoints on the image
    #cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
    #cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
    cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
    cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)

    # draw lines between the midpoints
    #cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
    #   (255, 0, 255), 2)
    cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
        (255, 0, 255), 2)

    # compute the Euclidean distance between the midpoints
    dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
    dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

    print("dA: ", dA, " dB: ", dB)
    # if the pixels per metric has not been initialized, then
    # compute it as the ratio of pixels to supplied metric
    # (in this case, inches)
    if pixelsPerMetric is None:
        pixelsPerMetric = get_width_from_ruler(dB, rulerWidth)

    # compute the size of the object
    dimA = dA / pixelsPerMetric
    dimB = dB / pixelsPerMetric

    # draw the object sizes on the image
    #cv2.putText(orig, "{:.1f}in".format(dimA),
    #       (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
    #   0.65, (255, 255, 255), 2)
    if i == 0:
        pre = "Ruler"
    else:
        pre = "Abalone"
    cv2.putText(orig, "{}: {:.1f}in".format(pre, dimB),
        (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
        0.65, (255, 255, 255), 2)
    
    if showResults:
        cv2.imshow("orig", orig)
        cv2.waitKey(0)
    else:
        out_file = "../blue_data.csv"
        delimeter = ","
        quotechar = '|'
        all_rows = {}
        if os.path.exists(out_file):
            print("out exist")
            with open(out_file, 'rb') as csvfile:
                print("closed:   ", csvfile.closed)
                csvreader = csv.reader(csvfile, delimiter=delimeter, quotechar='|')
                print("reader: ", csvreader)
                try:
                    for row in csvreader:
                        name = row[0]
                        size = row[1]
                        all_rows[name] = size
                except StandardError, e:
                    print("problem here: {}".format(e))


        all_rows[imageName] = dimB

        with open(out_file, 'wb') as csvfile:
            writer = csv.writer(csvfile, delimiter=delimeter, quotechar=quotechar, quoting=csv.QUOTE_MINIMAL)
            for name, size in all_rows.items():
                writer.writerow([name, size])

        # show the output image
        #plt.imshow(orig, interpolation = 'bicubic')
        #plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
        
        #plt.show()
        #cv2.imshow("orig", orig)
        #cv2.waitKey(0)