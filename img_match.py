import numpy as np
import cv2

#my files
import utils


def do_image_matching(input_image):
    #show_img("half ruler", ruler_image.copy())
    lobster_img =cv2.imread('lobster_head.png')
    lob_height,lob_width = lobster_img.shape[:2]
    img_height, img_width = input_image.shape[:2]
    print "lob img h: {}, lob width: {}, img h: {}, img w:{}".format(lob_height, lob_width, img_height, img_width)

    if img_height > img_width:
        #template_image = scale_template(input_image, lobster_img)
        template_image = lobster_img
        #same orientation as template
        utils.show_img("origal img", lobster_img)
        ruler_mask, ruler_top_offset_x, ruler_top_offset_y = match_all_templates(input_image.copy(), lobster_img,True)

    else:
        #rotate
        rotImg = lobster_img.copy()
        rows = len(rotImg)
        cols = len(rotImg[0])
        print "rows: {}, cols: {}".format(rows, cols)
        M = cv2.getRotationMatrix2D((rows/3.6,cols/2),90,1)
        rotated_img = cv2.warpAffine(rotImg,M,(rows,cols))
        #utils.show_img("rot 90", rotated_img)

        #scaled_template = scale_template(input_image, rotated_img)
        scaled_template = rotated_img
        #utils.show_img("scaled", scaled_template)
        match_all_templates(input_image.copy(), scaled_template,True)


        M = cv2.getRotationMatrix2D((rows/1.95,cols/1.1),-90,1)
        rotated_img = cv2.warpAffine(rotImg,M,(rows,cols))
        #utils.show_img("rot -90", rotated_img)
        match_all_templates(input_image.copy(), scaled_template,True)


def scale_template(input_img, template_img):
    orig_cols = len(input_img[0])
    orig_rows = len(input_img)
    orig_ratio = orig_cols/orig_rows

    template_cols = len(template_img[0])
    template_rows = len(template_img)

    fx = float(orig_cols)/float(template_cols)
    fy = float(orig_rows)/float(template_rows)

    scaled_image = cv2.resize(template_img, (0,0), fx = fx, fy = fy)
    return scaled_image

def match_all_templates(input_image, template_image, showImg):
    h,w = template_image.shape[:2]
    res = cv2.matchTemplate(input_image,template_image,cv2.TM_CCOEFF_NORMED)
    threshold = 0.5
    loc = np.where( res >= threshold)
    print "loc: {}".format(loc[::-1])

    for pt in zip(*loc[::-1]):
        print "pt: {}".format(pt)
        cv2.rectangle(input_image, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)

    if showImg:
        utils.show_img("all results", input_image)

def match_template(input_image, template_image, showImg):

    h,w = template_image.shape[:2]
    
    res = cv2.matchTemplate(input_image,template_image,cv2.TM_SQDIFF)
    utils.show_img("results", res)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    print "min val: {}, max_val: {}, min loc: {},, max_ loc: {}".format(min_val, max_val, min_loc, max_loc)
    #SQDIFF and SQ_DIFF_NORMED use top_left is min_loc, otherwise top_left is max_loc
    top_left = max_loc
    x = top_left[0]
    y = top_left[1]
    bottom_right = (top_left[0] + w, top_left[1]+h)
    if showImg:
        print "top: {}, bottom:{}".format(top_left, bottom_right)
    cv2.rectangle(input_image,top_left, bottom_right, 255, 2)

    y,x = np.unravel_index(res.argmax(), res.shape)
    if showImg:
        print "x: {}, y:{}".format(x, y)
    mask = np.zeros(input_image.shape,np.uint8)
    mask[x:x+w, y:y+h] = input_image[x:x+w, y:y+h]
    if showImg:
        cv2.namedWindow("template", cv2.WINDOW_NORMAL)
        cv2.imshow("template", input_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return mask, x,y