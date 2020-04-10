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
import targets
import ast
import PIL

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

    return scaled_image, rows, cols


def trim_quarter(quarter_contour):
    centroidX, centroidY, qell, quarter_ellipse = utils.get_quarter_contour_and_center(quarter_contour)

    if qell is not None and len(qell) > 0:

        return quarter_ellipse
    else:
        return quarter_contour



def runFromML(imageName, maskImageName, fullMaskName, username, email, uuid, ref_object, 
              ref_object_units, ref_object_size, locCode, fishery_type, original_filename, 
              original_size, extraMaskName,showResults=False, measurementDirection="length"):
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
        if constants.isLobster(fishery_type) or constants.isFinfish(fishery_type) and fullMaskName != None and fullMaskName != "":
            full_mask_image = cv2.imread(fullMaskName)
        else:
            full_mask_image = None
        print("about to start executing.....")
        rescaled_image, targetLength, targetWidth, left_point, right_point, width_left_point, width_right_point, left_ruler_point, right_ruler_point, whichTechnique = execute(imageName, 
                        image_full, mask_image, full_mask_image, 
                        showResults, is_deployed, 
                        fishery_type, ref_object, ref_object_size, ref_object_units, 
                        extra_mask_image, measurementDirection)


        if False:
            file_utils.read_write_simple_csv("data_finfish_516.csv", imageName, targetLength, ref_object_units, "mm", whichTechnique)

        rows = len(rescaled_image)
        cols = len(rescaled_image[0])
        orig_rows = len(image_full)
        orig_cols = len(image_full[0])

        presigned_url = ""
        #if is_deployed:
        
        
        if True:
            dynamo_name = 'ocean-ruler-test';
            s3_bucket_name = 'ocean-ruler-test';
          
            presigned_url = uploads.upload_worker(username, email, uuid, locCode, picDate, targetLength, rating, notes,
                left_point[0], left_point[1],right_point[0], right_point[1], 
                left_ruler_point[0], left_ruler_point[1], right_ruler_point[0],right_ruler_point[1], fishery_type, ref_object, ref_object_size, ref_object_units, 
                orig_cols, orig_rows, dynamo_name, s3_bucket_name, original_filename, 
                original_size, targetWidth, width_left_point[0], width_left_point[1], 
                width_right_point[0], width_right_point[1], measurementDirection)


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
                    "measurement_direction":str(measurementDirection)
                }


    except Exception as e:
        utils.print_time("big bombout....: {}".format(e), _start_time)
        rval={"big bombout":str(e)}
        if True:
            file_utils.read_write_error("error.csv", imageName, str(e))
            
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

def getClippingBoundsFromMask(mask_image, rescaled_image, orig_cols, orig_rows, allShapes=False,useCircle=False):

    if useCircle:
        gray = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 127, 255, 0)
        print("getting contours")
        im2, target_shapes, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        target_shapes = sorted(target_shapes, key=lambda shape: cv2.contourArea(shape), reverse=True)
        target_contour = target_shapes[0]

        print("getting circle")
        #add enclosing envelope and *and* them together
        (x,y),radius = cv2.minEnclosingCircle(target_contour)

        blank = np.zeros( mask_image.shape[0:2] )

        # copy each of the contours (assuming there's just two) to its own image. 
        # Just fill with a '1'.
        #circleImg = cv2.drawContours( blank.copy(), [target_contour], 0, 1 )
        
        circleImg = cv2.circle(blank, (int(x),int(y)), int(radius), (255, 255, 255), -1)
        rescaled_mask = templates.rescale(orig_cols, orig_rows, circleImg)
        rescaled_mask[rescaled_mask > 0] = 1
        rescaled_mask = 255 * rescaled_mask # Now scale by 255

        rescaled_mask = rescaled_mask.astype(np.uint8)
        print("finding second contours")
        im2, target_shapes, hierarchy = cv2.findContours(rescaled_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        
        target_shapes = sorted(target_shapes, key=lambda shape: cv2.contourArea(shape), reverse=True)
        target_contour = target_shapes[0]

    else:

        rescaled_mask = templates.rescale(orig_cols, orig_rows, mask_image)
        gray = cv2.cvtColor(rescaled_mask, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 127, 255, 0)
        print("finding non circle contours")
        im2, target_shapes, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        target_shapes = sorted(target_shapes, key=lambda shape: cv2.contourArea(shape), reverse=True)
        target_contour = target_shapes[0]

    if False:
        print("showing now...")
        #cv2.drawContours(input_image, [contour], 0, (0,0,255), 3)
        targ = rescaled_image.copy()
        cv2.drawContours(targ, [target_contour], -1, (0,255,255),4)
        cv2.imshow("clipped from mask", targ)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    
    
    if allShapes:
        return target_shapes   
    else:
        return target_contour

def getAllClippedImages(rescaled_image, clippingShapes, target_contour, fishery_type):
    shapes = []
    for cs in clippingShapes:
        #area = cv2.contourArea(target_contour)
        #print('area: {}'.format(area))
        clippedImage, x, y = getClippedImage(rescaled_image, cs, target_contour, fishery_type)
        shapes.append([clippedImage,x,y])

    return shapes

def getClippedImage(rescaled_image, clippingShape, target_contour=None, fishery_type=None):
    x,y,w,h = cv2.boundingRect(clippingShape)
    newX = x
    newY = y
    newWidth = w
    newHeight = h

    img_to_clip = rescaled_image

    #if target_contour is not None:
    #    img_to_clip = cv2.drawContours( rescaled_image, [target_contour], 0, 1 )
        

    clippedImage = img_to_clip[newY:newY+newHeight,newX:newX+newWidth]
    if utils.isFinfish(fishery_type):
        origImg = rescaled_image.copy()
        origMask = np.ones(origImg.shape[:2], dtype="uint8") *255
        mask = origMask.copy()
        cv2.drawContours(mask, [clippingShape], -1, 0, -1)
        invertMask = cv2.bitwise_not(mask)
        clippedMaskImage = cv2.bitwise_and(origImg, origImg, mask=invertMask)
        clippedImage = clippedMaskImage[newY:newY+newHeight,newX:newX+newWidth]

        if False:
            utils.show_img("mask before fill", mask)
            utils.show_img("mask after fill", mask)
            #utils.show_img("mask: ", mask)
            #utils.show_img("clipped: ", clippedImage)
            
            utils.show_img("invert mask: ", invertMask)
            
            utils.show_img("inverted mask: ", clippedMaskImage)
    #if target_contour is not None:
    #    utils.show_img("masked with target", img_to_clip)
    
    return clippedImage, x, y


def rotate_img(img):
    print("doing nothing now...")
    '''
    if img is not None:

        img = cv2.transpose(img)
        img = cv2.flip(img, 0)
        img = img.copy()
    '''
    return img

def get_clipped_quarter_image(input_image,full_mask_image, target_contour):
    if target_contour is not None:
        orig_cols = len(input_image[0]) 
        orig_rows = len(input_image)

        #clippingShape = getClippingBoundsFromMask(full_mask_image, input_image, orig_cols, orig_rows, allShapes=False, useCircle=False)
        clippingShape = target_contour
        clippedMaskImage = input_image.copy()
        
        cv2.fillPoly(clippedMaskImage, [clippingShape], 0)
 
        ca = 0
        cols = len(input_image[0]) 
        rows = len(input_image)
        scaled_image = clippedMaskImage[ca:rows-ca,ca:cols-ca]
    else:
        ca = 0
        cols = len(input_image[0]) 
        rows = len(input_image)
        scaled_image = input_image[ca:rows-ca,ca:cols-ca]
    

    return [[scaled_image,ca,ca]]


def execute(imageName, image_full, mask_image, full_mask_image, showResults, 
            is_deployed, fishery_type, ref_object, ref_object_size, ref_object_units, 
            ro_mask_image=None,measurementDirection="length"):
    mlPath = os.environ['ML_PATH']+"/../"
    #width of US quarter in inches
    orig_cols = len(image_full[0]) 
    orig_rows = len(image_full)
    whichTechnique = ""
    divisor = 1.0

    ref_object_size = float(ref_object_size)/divisor

    image_height, image_width, channels = image_full.shape
    origCellCount = image_height*image_width
    '''
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
    '''

    print("getting scaled image...")
    rescaled_image, scaled_rows, scaled_cols = get_scaled_image(image_full)    

    clipped_image = None

    mlFullMask = None
    clippedFullImage = None
    xFullOffset = None
    yFullOffset = None
    xOffset = 0
    yOffset = 0
    clippedImage = None

    if mask_image is not None:
        if constants.isLobster(fishery_type):
            orig_cols = len(rescaled_image[0]) 
            orig_rows = len(rescaled_image)
            mlMask = getClippingBoundsFromMask(mask_image, rescaled_image, orig_cols, orig_rows, allShapes=False, useCircle=True)
            mlFullMask = getClippingBoundsFromMask(full_mask_image, rescaled_image, orig_cols, orig_rows, allShapes=False, useCircle=False)
            clippedFullImage, xFullOffset, yFullOffset = getClippedImage(rescaled_image, mlFullMask, None, fishery_type)
        else:
            print("getting clipping mask...")
            mlMask = getClippingBoundsFromMask(mask_image, rescaled_image, scaled_cols, scaled_rows,allShapes=False, useCircle=False)
            print("now getting clipped image")
            
            clippedImage, xOffset, yOffset = getClippedImage(rescaled_image, mlMask, None, fishery_type)
            
    edge_contour = contour_utils.offset_contour(mlMask, -xOffset, -yOffset)
    if constants.isLobster(fishery_type) and xFullOffset is not None:
        xOffset = xFullOffset
        yOffset = yFullOffset

    #get the contour for the appropriate fishery
    ref_object_contour = None
    all_square_contours = None
    is_square_ref = (ref_object == constants.SQUARE)
    target_file = targets.get_target_file(fishery_type)
   
    if target_file is not None and mask_image is not None:
        print("target file: {}".format(target_file))
        
        import_expr = 'import {} as targ'.format(target_file)

        exec(import_expr)
        target_method = 'targ.get_dynamic_target_contour'
        #only the lobster returns the full contour for now...
        target_contour, target_full_contour, top_offset, left_offset = eval(target_method)(rescaled_image.copy(), clippedImage, 
                                                    fishery_type, orig_cols, orig_rows, mlPath, is_square_ref, xOffset, yOffset, ml_mask=mlMask, 
                                                    clipped_full_image=clippedFullImage, edge_contour=edge_contour)

    else:
        print("couldn't find the fishery type, using default")
        tmpimg =rescaled_image.copy()
        small_abalone_template_contour = templates.get_template_contour(orig_cols, orig_rows, mlPath+"images/abalone_only_2x.png")
        isWhiteOrGray = utils.is_white_or_gray(rescaled_image.copy(), False) 
        
        #target_contour, orig_contours = contour_utils.get_target_contour(rescaled_image.copy(), rescaled_image.copy(), small_abalone_template_contour, 
        #                                                                    is_square_ref, (constants.isAbalone(fishery_type)), isWhiteOrGray, fishery_type)
        target_contour, orig_contours = contour_utils.get_target_contour(clippedImage, rescaled_image.copy(), small_abalone_template_contour, 
                                                                                is_square_ref, (constants.isAbalone(fishery_type)), True, fishery_type)
        if target_contour is None:
            print("could not find a contour with the clipped image...")
            target_contour, orig_contours = contour_utils.get_target_contour(rescaled_image.copy(), 
                                            rescaled_image.copy(), small_abalone_template_contour, 
                                            is_square_ref, (constants.isAbalone(fishery_type)), isWhiteOrGray, fishery_type)
        else:
            target_contour = contour_utils.offset_contour(target_contour, xOffset, yOffset)


    if ref_object == constants.QUARTER:
        ref_object_size = constants.QUARTER_SIZE_CM
        #experiment with ditching quarter mask and just clipping image
        clippedImages = get_clipped_quarter_image(rescaled_image.copy(), full_mask_image, target_contour)
        simpleQuarterImage =  get_clipped_quarter_image(rescaled_image.copy(), None, None)

        ref_object_template_contour = templates.get_template_contour(orig_cols, orig_rows, mlPath+"images/quarter_template_1280.png")
         
        isWhiteOrGray = True
        original_size = scaled_rows*scaled_cols
        print("trying to get quarter dimension...")
        refObjectCenterX, refObjectCenterY, refRadius, matches, whichTechnique = contour_utils.get_best_quarter_dimensions(clippedImages, simpleQuarterImage,
                                                                     target_contour, ref_object_template_contour, False, origCellCount, isWhiteOrGray, original_size=original_size)    
    else:

        tmpimg =rescaled_image.copy()
        templatePath = mlPath+"lobster_templates/square_templates_2inch.png"

        ref_object_template_contour = templates.get_template_contour(orig_cols, orig_rows, templatePath)
        ref_object_contour, all_square_contours = contour_utils.get_square_contour(tmpimg, target_contour, ref_object_template_contour, _start_time)


    showText = showResults and not is_deployed
    flipDrawing = orig_rows/orig_cols > 1.2

    new_drawing = rescaled_image.copy()
    if ref_object == constants.QUARTER:
        
        pixelsPerMetric, quarterSize, left_ref_object_point, right_ref_object_point = drawing.draw_quarter_contour(new_drawing, 
            target_contour,showText, flipDrawing, refObjectCenterX, refObjectCenterY, (refRadius*2)-1, ref_object_size)
    else:
        pixelsPerMetric, squareSize,left_ref_object_point, right_ref_object_point = drawing.draw_square_contour(new_drawing, 
            ref_object_contour, None, True, flipDrawing, float(ref_object_size), ref_object_units, fishery_type)


    drawWidth = False
    print("measurement direction passed in is {}".format(measurementDirection))
    if measurementDirection is None:
        drawWidth = constants.isScallop(fishery_type)
    else:
        drawWidth = (measurementDirection == constants.WIDTH_MEASUREMENT)
    
    
    if constants.isLobster(fishery_type):
        targetLength, left_point, right_point = drawing.draw_target_lobster_contour(new_drawing, target_contour, pixelsPerMetric, True, left_offset, top_offset, target_full_contour)

        targetWidth = 0
        width_left_point = (0,0)
        width_right_point = (0,0)
    elif drawWidth:
        targetLength, targetWidth, left_point, right_point, width_left_point, width_right_point = drawing.draw_target_contour_with_width(new_drawing, 
            target_contour, showText, flipDrawing, pixelsPerMetric, fishery_type)  

    elif constants.isFinfish(fishery_type):
        targetLength, left_point, right_point = drawing.draw_target_finfish_contour(new_drawing, 
            target_contour, pixelsPerMetric, showText, 0, 0) 
        
        targetWidth = 0
        width_left_point = (0,0)
        width_right_point = (0,0)
    else:
        
        targetLength, targetWidth, left_point, right_point, width_left_point, width_right_point = drawing.draw_target_contour(new_drawing, 
            target_contour, showText, flipDrawing, pixelsPerMetric, fishery_type) 
                

    utils.print_time("done drawing target contours", _start_time)

    if not is_deployed and showResults:
        utils.show_img("Final Measurements for {}".format(imageName), new_drawing)
        write_new_image(imageName, new_drawing)
    else:
        if not is_deployed:
            write_new_image(imageName, new_drawing)


    return rescaled_image, targetLength, targetWidth, left_point, right_point, width_left_point, width_right_point, left_ref_object_point, right_ref_object_point, whichTechnique
    
def write_new_image(imageName, image):
    basename = os.path.basename(imageName)
    path = os.path.dirname(imageName)
 
    out_path = os.path.join(path, "output")
    if not os.path.exists(out_path):
        os.mkdir(out_path)

    out_name = os.path.join(out_path, basename)
    cv2.imwrite(out_name, image)


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

