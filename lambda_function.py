import cv2
import numpy as np
import sys
import time
import zipfile
import threading
import logging
import math
import base64
import time
import json
import time
import os

import utils
import file_utils
import color_images as ci
import templates
import drawing
import uploads
import contour_utils
import constants
import csv


ABALONE = "abalone"
RULER = "ruler"
QUARTER = "_quarter"
SQUARE = "square"
_start_time = time.time()
DELIM = ","
QUOTECHAR = '|'

def get_scaled_image(image_full):
    target_cols = 1280.0
    #target_rows = 960.0

    orig_cols = len(image_full[0]) 
    orig_rows = len(image_full)
    print("------------>>>>>>orig cols: {}, orig_rows: {}".format(orig_cols, orig_rows))
    #orig_cols > 1920

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
    print("opencv version::::: {} ".format(cv2.__version__))
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
        except Exception:
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
                original_filename = req[u'originalFilename']
            if original_filename is None or len(original_filename) == 0:
                original_filename = "Unknown"
                original_size = req[u'originalSize']
            if original_size is None or len(original_size) == 0:
                original_size = 0.0

            ref_object = req[u'refObject']
            if ref_object is None or len(ref_object) == 0:
                ref_object = constants.QUARTER

            ref_object_size = req[u'refObjectSize']
            if ref_object_size is None or len(ref_object_size) == 0:
                ref_object_size = constants.QUARTER_SIZE_IN

            ref_object_units = req[u'refObjectUnits']
            if ref_object_units is None or len(ref_object_units) == 0:
                ref_object_units = constants.INCHES


        except Exception:
            utils.print_time("error getting args: {}".format(e), _start_time)
            fishery_type="abalone"
            ref_object="quarter"
            ref_object_size=0.955
            ref_object_units="inches"
        print("fishery type: {}, ref object:{}, size: {}, units:{} ".format(fishery_type, ref_object, ref_object_size, ref_object_units))
    else:
        print("here!!!!")
        (imageName, showResults, out_file, fishery_type, ref_object, ref_object_size, ref_object_units) = file_utils.read_args()
        shouldIgnore = file_utils.shouldIgnore(imageName)

        if shouldIgnore:
            print("IGNORING THIS ONE!!!!")
            return
        original_filename = imageName
        original_size = 8.522
        image_full = cv2.imread(imageName)
        thumb = utils.get_thumbnail(image_full)
        img_data = cv2.imencode('.png', image_full)[1].tostring()
        thumb_str = cv2.imencode('.png', thumb)[1].tostring()
        rating = '-1'
        notes = 'none'
        name = 'DUploadTestv2'
        email = 'foo@bar.c'
        uuid = 'a412c020-3254-430a-a108-243113f9fde5'
        locCode = "S88 Bodega Head"
        picDate = int(time.time()*1000);

    rval = {}
    print("fishery type: {}, ref object: {}, ref size: {}, ref units: {}".format(fishery_type, ref_object, ref_object_size, ref_object_units))
    try:
        rescaled_image, pixelsPerMetric, abaloneLength, left_point, right_point, left_ruler_point, right_ruler_point = execute(imageName, image_full, None, showResults, is_deployed, 
                        fishery_type, ref_object, ref_object_size, ref_object_units)
        print("execute........")

        if is_mac() and out_file is not None:
            print("writing real file! {}".format(out_file))
            file_utils.read_write_simple_csv(out_file, imageName, abaloneLength)

        rows = len(rescaled_image)
        cols = len(rescaled_image[0])
        orig_rows = len(image_full)
        orig_cols = len(image_full[0])
        presigned_url = ""
        #if is_deployed:
        if False:
            utils.print_time("starting upload", _start_time)
            dynamo_name = 'ocean-ruler-test';
            s3_bucket_name = 'ocean-ruler-test';
            presigned_url = uploads.upload_worker(rescaled_image, thumb, img_data, name, email, uuid, locCode, picDate, abaloneLength, rating, notes,
                left_point[0], left_point[1],right_point[0], right_point[1], 
                left_ruler_point[0], left_ruler_point[1], right_ruler_point[0],right_ruler_point[1], fishery_type, ref_object_size, ref_object_size, ref_object_units, 
                orig_cols, orig_rows, dynamo_name, s3_bucket_name, original_filename, original_size)

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
                    "fishery_type":str(fishery_type), "presigned_url":presigned_url, "original_filename":str(original_filename), "original_size":str(original_size)
                }

        utils.print_time("total time after upload", _start_time)
    except Exception as e:
        utils.print_time("big bombout....: {}".format(e), _start_time)

    jsonVal = json.dumps(rval)
    print(jsonVal)
    return jsonVal

def runFromML(imageName, maskImageName, username, email, uuid, ref_object, ref_object_units, ref_object_size, locCode, fishery_type, original_filename, original_size):
    try: 
        original_filename = imageName

        image_full = cv2.imread(imageName)

        mask_image = cv2.imread(maskImageName)
        thumb = utils.get_thumbnail(image_full)
        img_data = cv2.imencode('.png', image_full)[1].tostring()
        thumb_str = cv2.imencode('.png', thumb)[1].tostring()
        rating = '-1'
        notes = 'none'

        picDate = int(time.time()*1000)
        showResults = True

        is_deployed = False

        rescaled_image, pixelsPerMetric, abaloneLength, left_point, right_point, left_ruler_point, right_ruler_point = execute(imageName, image_full, mask_image, showResults, is_deployed, 
                        fishery_type, ref_object, ref_object_size, ref_object_units)
        print("execute........")

        if False:
            print("writing real file! {}".format(out_file))
            file_utils.read_write_simple_csv(out_file, imageName, abaloneLength)

        rows = len(rescaled_image)
        cols = len(rescaled_image[0])
        orig_rows = len(image_full)
        orig_cols = len(image_full[0])
        print("orig colsxrows: {}x{}".format(orig_cols, orig_rows))
        presigned_url = ""
        #if is_deployed:
        if False:
            utils.print_time("starting upload", _start_time)
            dynamo_name = 'ocean-ruler-test';
            s3_bucket_name = 'ocean-ruler-test';
            presigned_url = uploads.upload_worker(rescaled_image, thumb, img_data, name, email, uuid, locCode, picDate, abaloneLength, rating, notes,
                left_point[0], left_point[1],right_point[0], right_point[1], 
                left_ruler_point[0], left_ruler_point[1], right_ruler_point[0],right_ruler_point[1], fishery_type, ref_object_size, ref_object_size, ref_object_units, 
                orig_cols, orig_rows, dynamo_name, s3_bucket_name, original_filename, original_size)

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
                    "fishery_type":str(fishery_type), "presigned_url":presigned_url, "original_filename":str(original_filename), "original_size":str(original_size)
                }

        utils.print_time("total time after upload", _start_time)
    except Exception as e:
        utils.print_time("big bombout....: {}".format(e), _start_time)
        rval={"big bombout":str(e)}
    jsonVal = json.dumps(rval)
    print(jsonVal)
    return jsonVal

def readClippingBounds(rescaled_image):
    clippingFile = "machine_learning/ml_output.csv"

    name=""
    startX = 0
    endX = 1280
    startY = 0
    endY = 960
    maxWidth = 224
    cols = len(rescaled_image[0])
    rows = len(rescaled_image)

    print("cols: {}, rows:{}".format(cols, rows))
    with open(clippingFile, 'rU') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=DELIM, quotechar=QUOTECHAR)

        try:
            for row in csvreader:
                name = row[0]
                startX = row[1]
                endX = row[2]
                startY = row[3]
                endY = row[4]
                maxWidth = row[5]

        except Exception as err:
            print("something went wrong reading clipping file: {}".format(err))

    xScale = int(float(cols)/224.0)
    yScale = int(float(rows)/224.0)

    xScale = xScale*(float(cols)/float(rows))
    print("xscale: {}, yscale:{}".format(xScale, yScale))
    startX = float(startX)*xScale
    endX = float(endX)*xScale

    startY = float(startY)*yScale
    endY = float(endY)*yScale

    return name, int(startX), int(endX), int(startY), int(endY), maxWidth

def getClippingBoundsFromMask(mask_image, rescaled_image, orig_cols, orig_rows, real_cols, real_rows):

    #STILL NOT CLIPPING RIGHT FOR ROTATED!
    new_cols = orig_cols
    new_rows = orig_rows
    if orig_cols < orig_rows:
        print("flipping mask image")
        img = cv2.transpose(mask_image)  
        img = cv2.flip(img, 0)
        new_cols = orig_rows
        new_rows = orig_cols

    print('new cols: {}, new rows: {}; real cols: {}, real rows: {}'.format(new_cols, new_rows, real_cols, real_rows))
    
    '''
    if real_cols < 1280:
        new_cols = real_cols
        new_rows = real_rows
    
    '''
    #utils.show_img("mask ", mask)
    rescaled_mask = templates.rescale(new_cols, new_rows, mask_image)

    #(thresh, im_bw) = cv2.threshold(rescaled_mask, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    #utils.show_img(thresh)

    mask_edged = cv2.Canny(rescaled_mask, 128, 255)
    #utils.show_img("mask:", rescaled_mask)
    edged_img = cv2.dilate(mask_edged, None, iterations=1)

    im2, target_shapes, hierarchy = cv2.findContours(edged_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    print("trying to get target shape: ")
    target_shape = target_shapes[0]

    if False:
        #cv2.drawContours(input_image, [contour], 0, (0,0,255), 3)
        cv2.drawContours(rescaled_image, [target_shape], 0, (0,255,255),10)
        cv2.imshow("clipped from mask", rescaled_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return target_shape

def getClippedImage(rescaled_image, clippingShape, image_width, image_height):
    x,y,w,h = cv2.boundingRect(clippingShape)
    print(" clipped!!!!!!!!!!  image width: {}, height:{}".format(image_width, image_height))
    if image_width < image_height:

        newX = y
        newY = x
        newWidth = h
        newHeight = w
    else:
        newX = x
        newY = y
        newWidth = w
        newHeight = h
    print("x:{},y:{},w:{},h:{}".format(x,y,w,h))
    clippedImage = rescaled_image[newY:newY+newHeight,newX:newX+newWidth]
    
    utils.show_img("clipped!", clippedImage)
    return clippedImage, x, y

def offset_contour(contour, x, y, image_width, image_height):
    '''
    print("................orig_cols: {}".format(orig_cols))
    if int(orig_cols) < 1280:
        print("..............orig cols are small, scaling down...")
        x = x*(orig_cols/1280)
        y = y*(orig_rows/960)
    '''
    if image_width < image_height:
        newX = y
        newY = x
    else:
        newX = x
        newY = y
    for points in contour:
        ndims = points.ndim
        if ndims > 1:
            for point in points:
                point[0] = point[0]+newX
                point[1] = point[1]+newX
        else:
            #for point in points:
            #print('point: {}'.format(point))
            points[0] = points[0]+newX
            points[1] = points[1]+newY
    return contour

def execute(imageName, image_full, mask_image, showResults, is_deployed, fishery_type, ref_object, ref_object_size, ref_object_units):
    mlPath = os.environ['ML_PATH']+"/../"
    #width of US quarter in inches
    orig_cols = len(image_full[0]) 
    orig_rows = len(image_full)
    
    #for calculation and storage, do everything in inches for consistency, then convert on displays
    print("orig ref object size: ", ref_object_size);

    if ref_object_units == constants.MM:
        ref_object_size = float(ref_object_size)/constants.INCHES_TO_MM;

    print("ref object after conversion: ", ref_object_size);
    

    image_height, image_width, channels = image_full.shape
    if orig_cols < orig_rows:
        img = cv2.transpose(image_full)  
        img = cv2.flip(img, 0)
        image_full = img.copy()
        orig_cols = len(image_full[0])
        orig_rows = len(image_full)
    

    rescaled_image, scaled_rows, scaled_cols = get_scaled_image(image_full)
    

    clipped_image = None

    if mask_image is not None:
        if fishery_type == constants.LOBSTER:
            print("getting scale stuff")
            
            orig_cols = len(rescaled_image[0]) 
            orig_rows = len(rescaled_image)
            mlMask = getClippingBoundsFromMask(mask_image, rescaled_image, orig_cols, orig_rows, orig_cols, orig_rows)
        else:
            print("clipping image...., {}x{}".format(orig_cols, orig_rows))
            mlMask = getClippingBoundsFromMask(mask_image, rescaled_image, scaled_cols, scaled_rows, orig_cols, orig_rows)
            print("now here....")
            clippedImage, xOffset, yOffset = getClippedImage(rescaled_image, mlMask, image_width, image_height)

    print("here?")
    #get the arget contour for the appropriate fishery
    ref_object_contour = None
    all_square_contours = None

    is_square_ref = (ref_object == constants.SQUARE)

    print("getting ab stuff")
    if fishery_type == constants.ABALONE and mask_image is not None:
        print("abalone")
        #abalone_template_contour = templates.get_template_contour(orig_cols, orig_rows,"images/big_abalone_only_2x.png")
        small_abalone_template_contour = templates.get_template_contour(orig_cols, orig_rows, mlPath+"images/abalone_only_2x.png")
        target_contour, orig_contours = contour_utils.get_target_contour(clippedImage, small_abalone_template_contour, is_square_ref)
        target_contour = offset_contour(target_contour, xOffset, yOffset, image_width, image_height)
        if False:
            cv2.drawContours(clippedImage, [target_contour], 0, (255,200,200),7)
            cv2.drawContours(clippedImage, orig_contours,-1,(0,0,255),2)
            #cv2.drawContours(tmpimg, [ref_object_template_contour], -1, (0,255,0),10)
            utils.show_img("clipped Image with contours", clippedImage)
            
    elif fishery_type == constants.LOBSTER:
        print("lobster")
        if mlMask is not None and mlMask.any():
            print("setting contour to mask")
            target_contour = mlMask
            top_offset = left_offset = 0
        else:
            print("no mask, doing normal")
            small_lobster_template_contour = templates.get_template_contour(orig_cols, orig_rows, mlPath+"lobster_templates/full_lobster_right.png")
            target_contour, orig_contours, top_offset, left_offset = contour_utils.get_lobster_contour(rescaled_image.copy(), small_lobster_template_contour)
        print("done getting lobster...")
    else:
        print("everything else")
        tmpimg =rescaled_image.copy()
        small_abalone_template_contour = templates.get_template_contour(orig_cols, orig_rows, mlPath+"images/abalone_only_2x.png")
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

        ref_object_template_contour = templates.get_template_contour(orig_cols, orig_rows, mlPath+"images/quarter_template_1280.png")
        refObjectCenterX, refObjectCenterY, refRadius, matches = contour_utils.get_quarter_dimensions(rescaled_image.copy(), target_contour, ref_object_template_contour, False)    
    else:
        print("getting squares ")
        tmpimg =rescaled_image.copy()
        templatePath = mlPath+"lobster_templates/square_templates_2inch.png"
        print("template path: {}".format(templatePath))
        ref_object_template_contour = templates.get_template_contour(orig_cols, orig_rows, templatePath)
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

    return rescaled_image, pixelsPerMetric, targetLength, left_point, right_point, left_ref_object_point, right_ref_object_point
    

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

