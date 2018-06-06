import cv2
import numpy as np
import sys
import time
import threading
import logging
import math
import base64
import time
import json
import time

import utils
import file_utils
import color_images as ci
import templates
import drawing
import uploads
import contour_utils
import constants


ABALONE = "abalone"
RULER = "ruler"
QUARTER = "_quarter"
SQUARE = "square"
_start_time = time.time()

def get_scaled_image(image_full):
    target_cols = 1280.0
    #target_rows = 960.0

    orig_cols = len(image_full[0]) 
    orig_rows = len(image_full)
    print("orig cols: {}, orig_rows: {}".format(orig_cols, orig_rows))
    #orig_cols > 1920

    if(orig_cols > 1920):
        target_cols = 1280
    elif(orig_cols < 1280):
        target_cols = 1280
    else:
        target_cols = 1280
    
    target_rows = (float(orig_rows)/(float(orig_cols))*target_cols)
    fx = float(target_cols)/float(orig_cols)
    fy = float(target_rows)/float(orig_rows)

    print("{}, {}".format(fx, fy))
    scaled_image = cv2.resize( image_full, (0,0), fx = fx, fy = fy)
    
    rows = int(len(scaled_image))
    cols = int(len(scaled_image[0]))

    #scaled_image = scaled_image[30:rows,50:cols-50]

    #return image_full, orig_rows-30, orig_cols-100
    return scaled_image, rows, cols

def get_color_image():
    #all the work
    thresh_val = 30
    blur_window = 5
    contour_color=(0,0,255)
    is_ruler=False
    use_gray_threshold=False
    enclosing_contour=None
    first_pass=False
    is_small=False
    use_adaptive=False
    input_image = rescaled_image.copy()
    color_image, threshold_bw, color_img, mid_row = ci.get_image_with_color_mask(input_image, thresh_val, 
        blur_window, False, first_pass, is_ruler, use_adaptive)


def trim_quarter(quarter_contour):
    centroidX, centroidY, qell, quarter_ellipse = utils.get_quarter_contour_and_center(quarter_contour)

    if qell is not None and len(qell) > 0:

        return quarter_ellipse
    else:
        return quarter_contour



def find_length(is_deployed, req):
    utils.print_time( "start", _start_time)


    bestRulerContour = None
    bestAbaloneContour = None

    if is_deployed:
        utils.print_time("its deployed....", _start_time)
        #user info
        name = req[u'username']
        email = req[u'email']
        uuid = req[u'uuid']
        locCode = req[u'locCode']
        picDate = req[u'picDate']
        rating = '-1'
        notes = 'none'


        utils.print_time("cv2 is loaded? ...checking", _start_time)
        val = cv2.COLOR_BGR2HSV
        print("val is {}".format(val))
        #img info
        utils.print_time("about to get image", _start_time)

        img_str = req[u'base64Image']

        try:
            img_data = base64.b64decode(img_str)
            utils.print_time("im data is ok? {}".format(img_data is not None), _start_time)
        except Exception, e:
            print("boom!! could read img_str ::: {}".format(e))

        tmp_filename = '/tmp/ab_length_{}.png'.format(time.time()) 

        with open(tmp_filename, 'wb') as f:
            f.write(img_data)
        print("wrote ok...")

        imageName = tmp_filename
        image_full = cv2.imread(imageName)

        thumb = utils.get_thumbnail(image_full)
        utils.print_time("image got!", _start_time)
        showResults = False
        out_file = None
        try:
            utils.print_time("getting new ones...", _start_time)
            fishery_type = req[u'fisheryType']
            if fishery_type is None or len(fishery_type) == 0:
                fishery_type = constants.ABALONE

            ref_object = req[u'refObject']
            if ref_object is None or len(ref_object) == 0:
                ref_object = constants.QUARTER

            ref_object_size = req[u'refObjectSize']
            if ref_object_size is None or len(ref_object_size) == 0:
                ref_object_size = constants.QUARTER_SIZE_IN

            ref_object_units = req[u'refObjectUnits']
            if ref_object_units is None or len(ref_object_units) == 0:
                ref_object_units = constants.INCHES

            minSize = req[u'minSize']
            if minSize is None:
                minSize = constants.MIN_SIZE
            else:
                try:
                    minSize = float(minSize)
                except Exception, e:
                    minSize = constants.MIN_SIZE

            maxSize = req[u'maxSize']
            if maxSize is None:
                maxSize = constants.MAX_SIZE
            else:
                try:
                    maxSize = float(maxSize)
                except Exception, e:
                    maxSize = constants.MAX_SIZE
        except Exception, e:
            utils.print_time("error getting args: {}".format(e), _start_time)
            fishery_type="abalone"
            ref_object="quarter"
            ref_object_size=0.955
            ref_object_units="inches"
        print("fishery type: {}, ref object:{}, size: {}, units:{} ".format(fishery_type, ref_object, ref_object_size, ref_object_units))
    else:
        (imageName, showResults, out_file, fishery_type, ref_object, ref_object_size, ref_object_units, minSize, maxSize) = file_utils.read_args()
        shouldIgnore = file_utils.shouldIgnore(imageName)

        if shouldIgnore:
            print("IGNORING THIS ONE!!!!")
            return

        image_full = cv2.imread(imageName)
        thumb = utils.get_thumbnail(image_full)
        img_data = cv2.imencode('.png', image_full)[1].tostring()
        thumb_str = cv2.imencode('.png', thumb)[1].tostring()
        rating = '-1'
        notes = 'none'
        name = 'DUploadTestv2'
        email = 'foo@bar.c'
        uuid = 'b412c020-3254-430a-a108-243113f9fde5'
        locCode = "S88 Bodega Head"
        picDate = int(time.time()*1000);

    print("fishery type: {}, ref object: {}, ref size: {}, ref units: {}".format(fishery_type, ref_object, ref_object_size, ref_object_units, minSize, maxSize))
    try:
        rescaled_image, pixelsPerMetric, abaloneLength, left_point, right_point, left_ruler_point, right_ruler_point, minSize, maxSize = execute(imageName, image_full, showResults, is_deployed, 
                        fishery_type, ref_object, ref_object_size, ref_object_units, minSize, maxSize)
            
        if is_mac():
            file_utils.read_write_simple_csv(out_file, imageName, abaloneLength)

        rows = len(rescaled_image)
        cols = len(rescaled_image[0])
        orig_rows = len(image_full)
        orig_cols = len(image_full[0])
        
        #if is_deployed:
        if is_deployed:
            utils.print_time("calculations done, uploading now", _start_time)
            uploads.upload_worker(rescaled_image, thumb, img_data, name, email, uuid, locCode, picDate, abaloneLength, rating, notes,
                left_point[0], left_point[1],right_point[0], right_point[1], 
                left_ruler_point[0], left_ruler_point[1], right_ruler_point[0],right_ruler_point[1], fishery_type, ref_object_size, ref_object_size, ref_object_units, 
                orig_cols, orig_rows)

        rval =  {
                    "start_x":str(left_point[0]), "start_y":str(left_point[1]), 
                    "end_x":str(right_point[0]), "end_y":str(right_point[1]), 
                    "length":str(abaloneLength),
                    "width":str(cols),"height":str(rows),
                    "quarter_start_x":str(left_ruler_point[0]),
                    "quarter_start_y":str(left_ruler_point[1]),
                    "quarter_end_x":str(right_ruler_point[0]),
                    "quarter_end_y":str(right_ruler_point[1]),
                    "uuid":str(uuid),
                    "ref_object":str(ref_object), "ref_object_size":str(ref_object_size),
                    "ref_object_units":str(ref_object_units), "orig_width":orig_cols, "orig_height":orig_rows,
                    "fishery_type":str(fishery_type)
                }
        utils.print_time("total time", _start_time)
    except StandardError, e:
        utils.print_time("big bombout....: {}".format(e), _start_time)

    jsonVal = json.dumps(rval)
    #print(jsonVal)
    return jsonVal

def execute(imageName, image_full, showResults, is_deployed, fishery_type, ref_object, ref_object_size, ref_object_units, minSize, maxSize):
    #width of US quarter in inches
    orig_cols = len(image_full[0]) 
    orig_rows = len(image_full)
    #if its vertical, flip it 90
    if orig_cols < orig_rows:
        img = cv2.transpose(image_full)  
        img = cv2.flip(img, 0)
        image_full = img.copy()
        orig_cols = len(image_full[0])
        orig_rows = len(image_full)
    
    rescaled_image, scaled_rows, scaled_cols = get_scaled_image(image_full)

    orig_cols = len(rescaled_image[0]) 
    orig_rows = len(rescaled_image)

    #get the arget contour for the appropriate fishery
    ref_object_contour = None
    all_square_contours = None

    is_square_ref = (ref_object == constants.SQUARE)

    if fishery_type == constants.ABALONE:
        print("abalone")
        #abalone_template_contour = templates.get_template_contour(orig_cols, orig_rows,"images/big_abalone_only_2x.png")
        small_abalone_template_contour = templates.get_template_contour(orig_cols, orig_rows, "images/abalone_only_2x.png")
        target_contour, orig_contours = contour_utils.get_target_contour(rescaled_image.copy(), small_abalone_template_contour, is_square_ref)

    elif fishery_type == constants.LOBSTER:
        print("lobster")
        small_lobster_template_contour = templates.get_template_contour(orig_cols, orig_rows, "lobster_templates/full_lobster_right.png")
        target_contour, orig_contours, top_offset, left_offset = contour_utils.get_lobster_contour(rescaled_image.copy(), small_lobster_template_contour)

    else:
        print("everything else")
        tmpimg =rescaled_image.copy()
        small_abalone_template_contour = templates.get_template_contour(orig_cols, orig_rows, "images/abalone_only_2x.png")
        utils.print_time("done getting template: ", _start_time);

        target_contour, orig_contours = contour_utils.get_target_contour(rescaled_image.copy(), small_abalone_template_contour, is_square_ref)
        if False:
            cv2.drawContours(tmpimg, [target_contour], -1, (100,100,100),8)
            utils.show_img("ref object", tmpimg)

    utils.print_time("done getting {} contours".format(fishery_type), _start_time)

    if ref_object == constants.QUARTER:
        if ref_object_units is None or ref_object_units == constants.INCHES:
            ref_object_size = constants.QUARTER_SIZE_IN
        else:
            ref_object_size = constants.QUARTER_SIZE_MM

        ref_object_template_contour = templates.get_template_contour(orig_cols, orig_rows, "images/quarter_template_1280.png")
        refObjectCenterX, refObjectCenterY, refRadius, matches = contour_utils.get_quarter_dimensions(rescaled_image.copy(), target_contour, ref_object_template_contour, False)    
    else:
        tmpimg =rescaled_image.copy()
        ref_object_template_contour = templates.get_template_contour(orig_cols, orig_rows, "lobster_templates/square_templates_2inch.png")
        ref_object_contour, all_square_contours = contour_utils.get_square_contour(tmpimg, target_contour, ref_object_template_contour, _start_time)
        
        if False:
            cv2.drawContours(tmpimg, all_square_contours, -1, (255,200,200),5)
            #cv2.drawContours(tmpimg, [ref_object_contour],-1,(0,0,255),10)
            #cv2.drawContours(tmpimg, [ref_object_template_contour], -1, (0,255,0),10)
            utils.show_img("ref object", tmpimg)

    utils.print_time("ref object contours done", _start_time)

    showText = showResults and not is_deployed
    flipDrawing = orig_rows/orig_cols > 1.2

    new_drawing = rescaled_image.copy()
    if ref_object == constants.QUARTER:
        pixelsPerMetric, targetLength, left_ref_object_point, right_ref_object_point = drawing.draw_quarter_contour(new_drawing, 
            target_contour,showText, flipDrawing, refObjectCenterX, refObjectCenterY, refRadius*2, ref_object_size)
    else:
        pixelsPerMetric, targetLength,left_ref_object_point, right_ref_object_point = drawing.draw_square_contour(new_drawing, 
            ref_object_contour, None, True, flipDrawing, float(ref_object_size))

    utils.print_time("drew ref object contour", _start_time)

    if fishery_type == constants.LOBSTER:
        targetLength, left_point, right_point = drawing.draw_lobster_contour(new_drawing, 
            target_contour, pixelsPerMetric, True, flipDrawing, ref_object_size, top_offset, left_offset)
    else:
        targetLength, left_point, right_point = drawing.draw_target_contour(new_drawing, 
            target_contour, showText, flipDrawing, pixelsPerMetric, fishery_type)    

    utils.print_time("done drawing target contours", _start_time)

    if not is_deployed and showResults:
        #cv2.circle(new_drawing,(quarterCenterX, quarterCenterY),quarterRadius,(0,255,0),4)

        utils.show_img("Final Measurements", new_drawing)

    return rescaled_image, pixelsPerMetric, targetLength, left_point, right_point, left_ref_object_point, right_ref_object_point, minSize, maxSize
    

def lambda_handler(event, context):
    try:
        _start_time = time.time()
        ab_length = find_length(True, event)
    except StandardError:
        ab_length = "Unknown"
        
    return ab_length

def is_mac():
    os_name = sys.platform
    return os_name == "darwin"

def run_program():
    os_name = sys.platform
    if os_name == "darwin":
        res = find_length(False, None)
    else:
        res = find_length(False, None)


if __name__ == "__main__":
    run_program()

