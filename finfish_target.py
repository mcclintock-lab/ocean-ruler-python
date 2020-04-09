import utils
import templates
import cv2
import contour_utils
import constants
import numpy as np

def get_dynamic_target_contour(input_image, clipped_image, fishery_type, orig_cols, orig_rows, ml_path, 
                                is_square_ref, x_offset, y_offset, ml_mask=None,
                                clipped_full_image=None, edge_contour=None):

    isWhiteOrGray = utils.is_white_or_gray(clipped_image, False) 
    print("is white or gray: {}".format(isWhiteOrGray))
    finfish_template_contour = templates.get_template_contour(orig_cols, orig_rows, ml_path+"images/finfish.png")

    gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 100, 255, 0)
    print("fishery type finfish....")

    target_contour, orig_contours = contour_utils.get_target_finfish_contour(input_image.copy(), clipped_image, 
                                                                            finfish_template_contour, 
                                                                            is_square_ref_object=is_square_ref,isWhiteOrGray=True, edge_of_mask=edge_contour)
    

    target_contour = contour_utils.offset_contour(target_contour, x_offset, y_offset)

    if False:
        rect = cv2.minAreaRect(target_contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(input_image,[target_contour],0,(0,191,255),8)
        #cv2.drawContours(rescaled_image, [target_contour], 0, (255,0,0),5)
        #cv2.drawContours(tmpimg, [ref_object_template_contour], -1, (0,255,0),10)
        utils.show_img("clipped Image from thresholding...", input_image)

    return target_contour, None, y_offset, x_offset