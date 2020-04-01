import utils
import templates
import cv2
import contour_utils
import constants

    
#
def get_dynamic_target_contour(input_image, clipped_image, fishery_type, orig_cols, orig_rows, ml_path, 
                                is_square_ref, x_offset, y_offset, ml_mask=None,
                                clipped_full_image=None, edge_contour=None):
    target_full_contour = None

    if ml_mask is not None and ml_mask.any():
        print("it has a mask")
        target_contour = ml_mask

        full_lobster_contour, orig_full_contours = contour_utils.get_target_full_lobster_contour(clipped_full_image)
        #todo: this should be xfulloffset, yfullofset 
        full_lobster_contour = contour_utils.offset_contour(full_lobster_contour, x_offset, y_offset)
        
        target_full_contour = full_lobster_contour

        top_offset = left_offset = 0
        
    else:
        print("NO MASK")
        small_lobster_template_contour = templates.get_template_contour(orig_cols, orig_rows, ml_path+"lobster_templates/full_lobster_right.png")
        target_contour, orig_contours, top_offset, left_offset = contour_utils.get_lobster_contour(input_image.copy(), small_lobster_template_contour)

    return target_contour, target_full_contour, top_offset, left_offset