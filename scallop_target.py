import utils
import templates
import cv2
import contour_utils
import constants


def get_dynamic_target_contour(input_image, clipped_image, fishery_type, orig_cols, orig_rows, ml_path, 
                                is_square_ref, x_offset, y_offset, ml_mask=None,
                                clipped_full_image=None, edge_contour=None):
    print("here: {}".format(input_image))
    isWhiteOrGray = utils.is_white_or_gray(input_image, False) 
    

    small_abalone_template_contour = templates.get_template_contour(orig_cols, orig_rows, 
        ml_path+"images/abalone_only_2x.png")

    gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 100, 255, 0)

    target_contour, orig_contours = contour_utils.get_target_contour(clipped_image, input_image.copy(), 
                                                                        small_abalone_template_contour, 
                                                                        is_square_ref, (constants.isAbalone(fishery_type)), True, fishery_type)
    target_contour = contour_utils.offset_contour(target_contour, x_offset, y_offset)
    
    if False:
        cv2.drawContours(input_image, [target_contour], 0, (255,0,0),5)
        
        utils.show_img("clipped Image from thresholding...", input_image)

    return target_contour, None, 0, 0