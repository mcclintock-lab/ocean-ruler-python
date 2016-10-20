import cv2


def find_edges(gray, threshhold, use_gray, showImg):
    # perform edge detection, then perform a dilation + erosion to
    # close gaps in between object edges

    if  use_gray:
         edged_img = cv2.Canny(gray, 60, 255)
    else:
        #goes down this path for quarter, too
        edged_img = cv2.Canny(threshhold, 50, 100)       

    edged_img = cv2.dilate(edged_img, None, iterations=1)
    edged_img = cv2.erode(edged_img, None, iterations=1)

    return edged_img

def get_largest_edge(cnts):
    if len(cnts) == 0:
        return None, None
    
    max_size = 0
    targetDex = 0
    target_contours = []
    for i, contour in enumerate(cnts):
        carea = cv2.contourArea(contour)
        if carea > max_size:
            max_size = carea

    for i, contour in enumerate(cnts):
        carea = cv2.contourArea(contour)
        if carea == max_size:
            target_contours.append(contour)

    return target_contours, max_size


def show_img_and_contour(imageName, input_image, contour, template_contour,top_offset=0):
    try:
        
        cv2.drawContours(input_image, [contour], 0, (0,0,255), 3)
        cv2.drawContours(input_image, [template_contour], 0, (255,0,0), 3)
        show_img(imageName, input_image)
    except StandardError, e:
        print "couldn't draw image..."

def show_img(title, img):
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()