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
import random
import pdb

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
ML_IMAGE_SIZE = 320

def get_scaled_image(image_full):
    target_cols = 1280.0
    #target_rows = 960.0

    orig_cols = len(image_full[0]) 
    orig_rows = len(image_full)

    #orig_cols > 1920

    target_cols = 1280
    
    target_rows = (float(orig_rows)/(float(orig_cols))*target_cols)
    fx = float(target_cols)/float(orig_cols)
    fy = float(target_rows)/float(orig_rows)


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



def runFromML(imageName, maskImageName, fullMaskName, username, email, uuid, ref_object, 
              ref_object_units, ref_object_size, locCode, fishery_type, original_filename, original_size, extraMaskName,showResults=False):
    try: 
        #original_filename = imageName

        image_full = cv2.imread(imageName)

        mask_image = cv2.imread(maskImageName)
        thumb = utils.get_thumbnail(image_full)

        extra_mask_image = cv2.imread(extraMaskName)
        img_data = cv2.imencode('.png', image_full)[1].tostring()
        thumb_str = cv2.imencode('.png', thumb)[1].tostring()
        rating = '-1'
        notes = 'none'

        picDate = int(time.time()*1000)

        is_deployed = False
        if constants.isLobster(fishery_type) and fullMaskName != None and fullMaskName != "":
            print("reading {}".format(fullMaskName))
            full_mask_image = cv2.imread(fullMaskName)
        else:
            full_mask_image = None
            
        rescaled_image, targetLength, targetWidth, left_point, right_point, width_left_point, width_right_point, left_ruler_point, right_ruler_point = execute(imageName, 
                        image_full, mask_image, full_mask_image, 
                        showResults, is_deployed, 
                        fishery_type, ref_object, ref_object_size, ref_object_units, extra_mask_image)

        
        if False:
            file_utils.read_write_simple_csv("data_scallop_1219.csv", imageName, targetLength, ref_object_units)

        rows = len(rescaled_image)
        cols = len(rescaled_image[0])
        orig_rows = len(image_full)
        orig_cols = len(image_full[0])

        presigned_url = ""
        #if is_deployed:
        print("original filename: {}".format(original_filename))
        
        if True:
            dynamo_name = 'ocean-ruler-test';
            s3_bucket_name = 'ocean-ruler-test';
            print("")
            presigned_url = uploads.upload_worker(rescaled_image, thumb, img_data, username, email, uuid, locCode, picDate, targetLength, rating, notes,
                left_point[0], left_point[1],right_point[0], right_point[1], 
                left_ruler_point[0], left_ruler_point[1], right_ruler_point[0],right_ruler_point[1], fishery_type, ref_object, ref_object_size, ref_object_units, 
                orig_cols, orig_rows, dynamo_name, s3_bucket_name, original_filename, original_size, targetWidth, width_left_point[0], width_left_point[1], width_right_point[0], width_right_point[1])


        rval =  {
                    "start_x":str(left_point[0]), "start_y":str(left_point[1]), 
                    "end_x":str(right_point[0]), "end_y":str(right_point[1]), 
                    "length":str(targetLength),
                    "width":str(cols),"height":str(rows),
                    "quarter_start_x":str(left_ruler_point[0]),
                    "quarter_start_y":str(left_ruler_point[1]),
                    "quarter_end_x":str(right_ruler_point[0]),
                    "quarter_end_y":str(right_ruler_point[1]),
                    "uuid":str(uuid),
                    "ref_object":str(ref_object), "ref_object_size":str(ref_object_size),
                    "ref_object_units":str(ref_object_units), "orig_width":orig_cols, "orig_height":orig_rows,
                    "fishery_type":str(fishery_type), "presigned_url":presigned_url, "original_filename":str(original_filename), "original_size":str(original_size),
                    "width_in_inches": str(targetWidth),
                    "target_width_start_x": str(width_left_point[0]),
                    "target_width_start_y":str(width_left_point[1]),
                    "target_width_end_x":str(width_right_point[0]),
                    "target_width_end_y":str(width_right_point[1]),
                    "newwidth": str(targetWidth),
                    "target_width_new_start_x": str(width_left_point[0]),
                    "target_width_new_start_y":str(width_left_point[1]),
                    "target_width_new_end_x":str(width_right_point[0]),
                    "target_width_new_end_y":str(width_right_point[1]),
                }


        utils.print_time("total time after upload", _start_time)
    except Exception as e:
        utils.print_time("big bombout....: {}".format(e), _start_time)
        rval={"big bombout":str(e)}
    jsonVal = json.dumps(rval)
    return jsonVal

def readClippingBounds(rescaled_image):
    clippingFile = "machine_learning/ml_output.csv"

    name=""
    startX = 0
    endX = 1280
    startY = 0
    endY = 960
    maxWidth = ML_IMAGE_SIZE
    cols = len(rescaled_image[0])
    rows = len(rescaled_image)


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

    xScale = int(float(cols)/ML_IMAGE_SIZE)
    yScale = int(float(rows)/ML_IMAGE_SIZE)

    xScale = xScale*(float(cols)/float(rows))

    startX = float(startX)*xScale
    endX = float(endX)*xScale

    startY = float(startY)*yScale
    endY = float(endY)*yScale

    return name, int(startX), int(endX), int(startY), int(endY), maxWidth

def getClippingBoundsFromMask(mask_image, rescaled_image, orig_cols, orig_rows, allShapes=False):

    rescaled_mask = templates.rescale(orig_cols, orig_rows, mask_image)
    gray = cv2.cvtColor(rescaled_mask, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 127, 255, 0)

    
    im2, target_shapes, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    target_shapes = sorted(target_shapes, key=lambda shape: cv2.contourArea(shape), reverse=True)

    if False:
        #cv2.drawContours(input_image, [contour], 0, (0,0,255), 3)
        cv2.drawContours(rescaled_image, target_shapes, -1, (0,255,255),1)
        cv2.imshow("clipped from mask", rescaled_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if allShapes:
        return target_shapes
    else:
        return target_shapes[0]

def getAllClippedImages(rescaled_image, clippingShapes, target_contour):
    shapes = []
    for cs in clippingShapes:
        #area = cv2.contourArea(target_contour)
        #print('area: {}'.format(area))
        clippedImage, x, y = getClippedImage(rescaled_image, cs, target_contour)
        shapes.append([clippedImage,x,y])

    return shapes

def getClippedImage(rescaled_image, clippingShape, target_contour=None):
    x,y,w,h = cv2.boundingRect(clippingShape)
    newX = x
    newY = y
    newWidth = w
    newHeight = h

    img_to_clip = rescaled_image

    #if target_contour is not None:
    #    img_to_clip = cv2.drawContours( rescaled_image, [target_contour], 0, 1 )
        

    clippedImage = img_to_clip[newY:newY+newHeight,newX:newX+newWidth]

    #if target_contour is not None:
    #    utils.show_img("masked with target", img_to_clip)
    #    utils.show_img("clipped: ", clippedImage)
    return clippedImage, x, y


def rotate_img(img):
    if img is not None:
        img = cv2.transpose(img)
        img = cv2.flip(img, 0)
        img = img.copy()
    return img

def execute(imageName, image_full, mask_image, full_mask_image, showResults, is_deployed, fishery_type, ref_object, ref_object_size, ref_object_units, ro_mask_image=None):
    mlPath = os.environ['ML_PATH']+"/../"
    #width of US quarter in inches
    orig_cols = len(image_full[0]) 
    orig_rows = len(image_full)
    
    divisor = 1.0
    '''
    #for calculation and storage, do everything in inches for consistency, then convert on displays
    if ref_object_units == constants.MM:
        divisor = constants.INCHES_TO_MM
    elif ref_object_units == constants.CM:
        divisor = constants.INCHES_TO_CM
    '''
    ref_object_size = float(ref_object_size)/divisor

    image_height, image_width, channels = image_full.shape
    origCellCount = image_height*image_width
    if orig_cols < orig_rows:
        #rotate the image. for simplicity, always do landscape
        img = rotate_img(image_full)
        image_full = img.copy()
        orig_cols = len(image_full[0])
        orig_rows = len(image_full)
        #rotate the masks
        mask_image = rotate_img(mask_image)
        full_mask_image = rotate_img(full_mask_image)
        ro_mask_image = rotate_img(ro_mask_image)

    rescaled_image, scaled_rows, scaled_cols = get_scaled_image(image_full)    
    clipped_image = None


    mlFullMask = None
    clippedFullImage = None
    if mask_image is not None:
        if constants.isLobster(fishery_type):
            print("its lobster")
            orig_cols = len(rescaled_image[0]) 
            orig_rows = len(rescaled_image)
            mlMask = getClippingBoundsFromMask(mask_image, rescaled_image, orig_cols, orig_rows)
            mlFullMask = getClippingBoundsFromMask(full_mask_image, rescaled_image, orig_cols, orig_rows)
            clippedFullImage, xFullOffset, yFullOffset = getClippedImage(rescaled_image, mlFullMask)
        else:
            print("not lobster, its {}".format(fishery_type))
            mlMask = getClippingBoundsFromMask(mask_image, rescaled_image, scaled_cols, scaled_rows)
  
            clippedImage, xOffset, yOffset = getClippedImage(rescaled_image, mlMask)
 
    #get the arget contour for the appropriate fishery
    ref_object_contour = None
    all_square_contours = None
    is_square_ref = (ref_object == constants.SQUARE)

    if (constants.isAbalone(fishery_type) or constants.isScallop(fishery_type)) and mask_image is not None:
        isWhiteOrGray = utils.is_white_or_gray(rescaled_image.copy(), False) 
        print('its ab or scallop')
        #abalone_template_contour = templates.get_template_contour(orig_cols, orig_rows,"images/big_abalone_only_2x.png")
        small_abalone_template_contour = templates.get_template_contour(orig_cols, orig_rows, mlPath+"images/abalone_only_2x.png")
        target_contour, orig_contours = contour_utils.get_target_contour(clippedImage, small_abalone_template_contour, 
                                                                            is_square_ref, (constants.isAbalone(fishery_type)), isWhiteOrGray)
        target_contour = contour_utils.offset_contour(target_contour, xOffset, yOffset)
        
        if False:
            cv2.drawContours(clippedImage, [target_contour], 0, (255,0,0),5)
            #cv2.drawContours(tmpimg, [ref_object_template_contour], -1, (0,255,0),10)
            utils.show_img("clipped Image with contours", clippedImage)
            
    elif constants.isLobster(fishery_type):

        if mlMask is not None and mlMask.any():
            target_contour = mlMask

            print("getting contours for lobster")
            target_full_contour, orig_full_contours = contour_utils.get_target_full_lobster_contour(clippedFullImage)
            target_full_contour = contour_utils.offset_contour(target_full_contour, xFullOffset, yFullOffset)

            if False:

                tmpimg = rescaled_image.copy()
                cv2.drawContours(tmpimg, [target_contour], -1,(0,0,255),2)
                cv2.drawContours(tmpimg, [target_full_contour], -1, (100,100,200),8)
                #for i, oc in enumerate(orig_full_contours):
                #    col = (random.randint(0,254), random.randint(0,254), random.randint(0,254))
                #    cv2.drawContours(tmpimg, [oc], -1,col,4)
                utils.show_img("lobster full", tmpimg)

            top_offset = left_offset = 0
            
        else:
            small_lobster_template_contour = templates.get_template_contour(orig_cols, orig_rows, mlPath+"lobster_templates/full_lobster_right.png")
            target_contour, orig_contours, top_offset, left_offset = contour_utils.get_lobster_contour(rescaled_image.copy(), small_lobster_template_contour)

    else:
        print("getting abalone with no mask")
        tmpimg =rescaled_image.copy()
        small_abalone_template_contour = templates.get_template_contour(orig_cols, orig_rows, mlPath+"images/abalone_only_2x.png")
        utils.print_time("done getting template: ", _start_time);
        isWhiteOrGray = utils.is_white_or_gray(rescaled_image.copy(), False) 
        target_contour, orig_contours = contour_utils.get_target_contour(rescaled_image.copy(), small_abalone_template_contour, 
                                                                            is_square_ref, (constants.isAbalone(fishery_type)), isWhiteOrGray)
        if False:
            cv2.drawContours(tmpimg, [target_contour], -1, (100,100,100),8)
            utils.show_img("ref object", tmpimg)

    utils.print_time("done getting {} contours".format(fishery_type), _start_time)

    if ref_object == constants.QUARTER:
        print("getting quarter stuff...")
        if ref_object_units is None or ref_object_units == constants.INCHES:
            ref_object_size = constants.QUARTER_SIZE_IN
        else:
            ref_object_size = constants.QUARTER_SIZE_MM

        if ro_mask_image is not None:

            roMasks = getClippingBoundsFromMask(ro_mask_image, rescaled_image, scaled_cols, scaled_rows, True)
            clippedImages = getAllClippedImages(rescaled_image, roMasks, target_contour)

            rescaled_masked = cv2.drawContours(rescaled_image.copy(),[target_contour],-1,(0,0,0),-1)
            roMaskedMasks = getClippingBoundsFromMask(ro_mask_image, rescaled_masked, scaled_cols, scaled_rows, True)
            clippedMaskedImages = getAllClippedImages(rescaled_masked.copy(), roMaskedMasks, target_contour)
            
        else:
            clippedImages = [rescaled_image.copy()]

        print("length shapes: {}".format(len(clippedImages)))
        ref_object_template_contour = templates.get_template_contour(orig_cols, orig_rows, mlPath+"images/quarter_template_1280.png")
        print("checking is white or gray")
        isWhiteOrGray = utils.is_white_or_gray(rescaled_image.copy(), False)   
        print("is white or gray? {}".format(isWhiteOrGray))
        refObjectCenterX, refObjectCenterY, refRadius, matches = contour_utils.get_best_quarter_dimensions(clippedImages, clippedMaskedImages,
                                                                     target_contour, ref_object_template_contour, False, origCellCount, isWhiteOrGray)    
    else:
        tmpimg =rescaled_image.copy()
        templatePath = mlPath+"lobster_templates/square_templates_2inch.png"

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
        pixelsPerMetric, quarterSize, left_ref_object_point, right_ref_object_point = drawing.draw_quarter_contour(new_drawing, 
            target_contour,showText, flipDrawing, refObjectCenterX, refObjectCenterY, refRadius*2, ref_object_size)
    else:
        pixelsPerMetric, squareSize,left_ref_object_point, right_ref_object_point = drawing.draw_square_contour(new_drawing, 
            ref_object_contour, None, True, flipDrawing, float(ref_object_size))

    utils.print_time("drew ref object contour", _start_time)

    if constants.isLobster(fishery_type):
        print("ignoring width for lobster...")
        targetLength, left_point, right_point = drawing.draw_target_lobster_contour(new_drawing, target_contour, pixelsPerMetric, True, left_offset, top_offset, target_full_contour)
        #targetLength, left_point, right_point = drawing.draw_lobster_contour(new_drawing, 
        #    target_contour, pixelsPerMetric, True, flipDrawing, ref_object_size, top_offset, left_offset, target_full_contour)  
        
        #targetLength, targetWidth, left_point, right_point, width_left_point, width_right_point = drawing.draw_target_contour_with_width(new_drawing, 
        #    target_contour, showText, flipDrawing, pixelsPerMetric, fishery_type)
        #if showResults: 
        #    ellipse = cv2.fitEllipse(target_contour)
        #    cv2.ellipse(new_drawing,ellipse,(0,255,0),2)
        targetWidth = 0
        width_left_point = (0,0)
        width_right_point = (0,0)
    elif constants.isScallop(fishery_type):
        print("doing target with width for scallop")
        targetLength, targetWidth, left_point, right_point, width_left_point, width_right_point = drawing.draw_target_contour_with_width(new_drawing, 
            target_contour, showText, flipDrawing, pixelsPerMetric, fishery_type)  
    else:
        targetLength, targetWidth, left_point, right_point, width_left_point, width_right_point = drawing.draw_target_contour(new_drawing, 
            target_contour, showText, flipDrawing, pixelsPerMetric, fishery_type)    

    utils.print_time("done drawing target contours", _start_time)

    if not is_deployed and showResults:
        #cv2.circle(new_drawing,(quarterCenterX, quarterCenterY),quarterRadius,(0,255,0),4)

        utils.show_img("Final Measurements", new_drawing)

    return rescaled_image, targetLength, targetWidth, left_point, right_point, width_left_point, width_right_point, left_ref_object_point, right_ref_object_point
    

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

