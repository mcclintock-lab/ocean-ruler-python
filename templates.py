import cv2

def rescale(orig_cols, orig_rows, template_img):
    template_cols = len(template_img[0])
    template_rows = len(template_img)

    fx = float(orig_cols)/float(template_cols)
    fy = float(orig_rows)/float(template_rows)

    scaled_image = cv2.resize(template_img, (0,0), fx = fx, fy = fy)
    return scaled_image
    
def get_template_contours(rescaled_image):
    row_offset = 30
    col_offset = 30

    orig_cols = len(rescaled_image[0]) 
    orig_rows = len(rescaled_image)

    #by default, using the big abalone template
    abalone_template = cv2.imread("images/big_abalone_only_2x.png")
    rescaled_ab_template = rescale(orig_cols, orig_rows, abalone_template)
    abalone_template = rescaled_ab_template[30:len(rescaled_ab_template),30:len(rescaled_ab_template[0])-30]

    small_abalone_template = cv2.imread("images/abalone_only_2x.png")
    rescaled_small_ab_template = rescale(orig_cols, orig_rows, small_abalone_template)
    small_abalone_template = rescaled_small_ab_template[30:len(rescaled_small_ab_template),30:len(rescaled_small_ab_template[0])-30]

    quarter_only = cv2.imread("images/quarter_template_1280.png")
    quarter_only = quarter_only[30:len(quarter_only),30:len(quarter_only[0])-30]

    template_edged = cv2.Canny(abalone_template, 15, 100)
    small_template_edged = cv2.Canny(small_abalone_template, 15, 100)
    quarter_only_edged = cv2.Canny(quarter_only, 15,100)

    edged_img = cv2.dilate(template_edged, None, iterations=1)
    small_edged_img = cv2.dilate(small_template_edged, None, iterations=1)
    quarter_edged_img = cv2.dilate(quarter_only_edged, None, iterations=1)

    im2, abalone_shapes, hierarchy = cv2.findContours(edged_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    abalone_shape = abalone_shapes[1]

    small_im, small_abalone_shapes, small_hierarchy = cv2.findContours(small_edged_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    small_abalone_shape = small_abalone_shapes[1]

    quarter_e, quarter_shapes, hierarchy2 = cv2.findContours(quarter_edged_img,  cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    quarter_shape = quarter_shapes[0] 

    return abalone_shape, small_abalone_shape,quarter_shape