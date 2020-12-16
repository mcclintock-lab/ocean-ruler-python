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
    """ Scale the image so they're all the same size
        image_full = the input image
    """

    orig_cols = len(image_full[0]) 
    orig_rows = len(image_full)

    target_cols = 1280
    
    target_rows = (float(orig_rows)/(float(orig_cols))*target_cols)
    fx = float(target_cols)/float(orig_cols)
    fy = float(target_rows)/float(orig_rows)


    scaled_image = cv2.resize( image_full, (0,0), fx = fx, fy = fy)
    
    rows = int(len(scaled_image))
    cols = int(len(scaled_image[0]))

    return scaled_image, rows, cols



def runFromML(imageName, maskImageName, fullMaskName, username, email, uuid, ref_object, 
              ref_object_units, ref_object_size, locCode, fishery_type, original_filename, 
              original_size, extraMaskName,showResults=False, measurementDirection="length"):
    """ Launch point from ocean-ruler-server/index.js


    """
    try: 
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


        rows = len(rescaled_image)
        cols = len(rescaled_image[0])
        orig_rows = len(image_full)
        orig_cols = len(image_full[0])

        presigned_url = ""
        #if is_deployed:
        
        if True:
            dynamo_name = 'ocean-ruler-main';
            s3_bucket_name = 'oceanruler-tnc-images';
          
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


def getClippingBoundsFromMask(mask_image, rescaled_image, orig_cols, orig_rows, useCircle=False):
    """ Gets the bounds of the clipping mask from the machine learning output. 
        mask_image: The output image from machine learning
        rescaled_image: The input image from the user (rescaled)
        orig_cols, orig_rows: the number of rows/cols in the original image 
        useCircle: For lobster, contour has circle around it instead of box (because it's matching
        on the carapace)

    """
    if useCircle:
        gray = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 127, 255, 0)
        im2, target_shapes, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        target_shapes = sorted(target_shapes, key=lambda shape: cv2.contourArea(shape), reverse=True)
        target_contour = target_shapes[0]

        (x,y),radius = cv2.minEnclosingCircle(target_contour)

        blank = np.zeros( mask_image.shape[0:2] )

        circleImg = cv2.circle(blank, (int(x),int(y)), int(radius), (255, 255, 255), -1)
        rescaled_mask = templates.rescale(orig_cols, orig_rows, circleImg)
        rescaled_mask[rescaled_mask > 0] = 1
        rescaled_mask = 255 * rescaled_mask # Now scale by 255

        rescaled_mask = rescaled_mask.astype(np.uint8)
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

    

    return target_contour


def getClippedImage(rescaled_image, clippingShape,fishery_type=None):
    """ Clip the image based on the clipping shape
        rescaled_image: The input image
        clippingShape: The mask shape from cv
        fishery_type: finfish are clipped slightly differently, so checking for the type. 

    """
    x,y,w,h = cv2.boundingRect(clippingShape)
    newX = x
    newY = y
    newWidth = w
    newHeight = h

    img_to_clip = rescaled_image

    clippedImage = img_to_clip[newY:newY+newHeight,newX:newX+newWidth]
    if utils.isFinfish(fishery_type):
        #for finfish, apply a mask before clipping. This helps get rid of some
        #of the noise in the messy finfish photos
        origImg = rescaled_image.copy()
        origMask = np.ones(origImg.shape[:2], dtype="uint8") *255
        mask = origMask.copy()
        cv2.drawContours(mask, [clippingShape], -1, 0, -1)
        invertMask = cv2.bitwise_not(mask)
        clippedMaskImage = cv2.bitwise_and(origImg, origImg, mask=invertMask)
        clippedImage = clippedMaskImage[newY:newY+newHeight,newX:newX+newWidth]

    return clippedImage, x, y


def get_clipped_quarter_image(input_image,full_mask_image, target_contour):
    """ Quarter image is clipped based on the input mask and target contour


    """
    if target_contour is not None:
        orig_cols = len(input_image[0]) 
        orig_rows = len(input_image)

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
    """ The main method. Fires off the other calculations
        The process in general is: 
        1. clip the image if there's a machine learning mask,
        2. find the target contour (the edge of the object)
        3. find the target contour of the reference object
        4. using the target contours, calculate the size based on the pixels per reference object
        5. send the resultant size back. 

    """

    mlPath = os.environ['ML_PATH']+"/../"

    orig_cols = len(image_full[0]) 
    orig_rows = len(image_full)
    whichTechnique = ""

    ref_object_size = float(ref_object_size)
    image_height, image_width, channels = image_full.shape
    origCellCount = image_height*image_width

    rescaled_image, scaled_rows, scaled_cols = get_scaled_image(image_full)    
    clipped_image = None

    mlFullMask = None
    clippedFullImage = None
    xFullOffset = None
    yFullOffset = None
    xOffset = 0
    yOffset = 0
    clippedImage = None

    #if there is a mask image, clip the input
    #note: may not be a mask image, depending on what the machine learning generated.
    #can fall back to this with new fisheries since there is no model
    if mask_image is not None:
        if constants.isLobster(fishery_type):
            orig_cols = len(rescaled_image[0]) 
            orig_rows = len(rescaled_image)
            mlMask = getClippingBoundsFromMask(mask_image, rescaled_image, orig_cols, orig_rows, useCircle=True)
            mlFullMask = getClippingBoundsFromMask(full_mask_image, rescaled_image, orig_cols, orig_rows, useCircle=False)
            clippedFullImage, xFullOffset, yFullOffset = getClippedImage(rescaled_image, mlFullMask,fishery_type)
        else:

            mlMask = getClippingBoundsFromMask(mask_image, rescaled_image, scaled_cols, scaled_rows, useCircle=False)
            clippedImage, xOffset, yOffset = getClippedImage(rescaled_image, mlMask, fishery_type)
            
    
    edge_contour = contour_utils.offset_contour(mlMask, -xOffset, -yOffset)
    if constants.isLobster(fishery_type) and xFullOffset is not None:
        xOffset = xFullOffset
        yOffset = yFullOffset

    #get the contour for the appropriate fishery
    ref_object_contour = None
    all_square_contours = None
    is_square_ref = (ref_object == constants.SQUARE)
    #check to see if there is an externally declared target file where the contour
    #is found for the given fishery
    target_file = targets.get_target_file(fishery_type)
   
    if target_file is not None and mask_image is not None:
        #The target file represents the actual code that does the work for the computation
        #It's written this way so that each fishery can have a separate file for getting the contour
        #Easier to add additional fisheries by specifying a xFisheryx_target.py file and loading it
        #above
        import_expr = 'import {} as targ'.format(target_file)

        exec(import_expr)
        target_method = 'targ.get_dynamic_target_contour'
        #only the lobster returns the full contour for now...
        target_contour, target_full_contour, top_offset, left_offset = eval(target_method)(rescaled_image.copy(), clippedImage, 
                                                    fishery_type, orig_cols, orig_rows, mlPath, is_square_ref, xOffset, yOffset, ml_mask=mlMask, 
                                                    clipped_full_image=clippedFullImage, edge_contour=edge_contour)

    else:
        #falling back to the default target contour finding, since so target file was found
        tmpimg =rescaled_image.copy()
        small_abalone_template_contour = templates.get_template_contour(orig_cols, orig_rows, mlPath+"images/abalone_only_2x.png")
        isWhiteOrGray = utils.is_white_or_gray(rescaled_image.copy(), False) 

        target_contour, orig_contours = contour_utils.get_target_contour(clippedImage, rescaled_image.copy(), small_abalone_template_contour, 
                                                                                is_square_ref, (constants.isAbalone(fishery_type)), True, fishery_type)
        if target_contour is None:
            target_contour, orig_contours = contour_utils.get_target_contour(rescaled_image.copy(), 
                                            rescaled_image.copy(), small_abalone_template_contour, 
                                            is_square_ref, (constants.isAbalone(fishery_type)), isWhiteOrGray, fishery_type)
        else:
            target_contour = contour_utils.offset_contour(target_contour, xOffset, yOffset)



    #find the contour for the reference object
    if ref_object == constants.QUARTER:
        ref_object_size = constants.QUARTER_SIZE_CM
       
        clippedImages = get_clipped_quarter_image(rescaled_image.copy(), full_mask_image, target_contour)
        simpleQuarterImage =  get_clipped_quarter_image(rescaled_image.copy(), None, None)

        ref_object_template_contour = templates.get_template_contour(orig_cols, orig_rows, mlPath+"images/quarter_template_1280.png")
         
        isWhiteOrGray = True
        original_size = scaled_rows*scaled_cols
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

    
    #width or length. uses whatever is specified by the user, but has a fallback for running locally
    drawWidth = False
    
    if measurementDirection is None:
        drawWidth = constants.isScallop(fishery_type)
    else:
        drawWidth = (measurementDirection == constants.WIDTH_MEASUREMENT)
    
    #get the sizes based on contour -- handle lobster differently because its the carapace not the whole target
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
                
    #show the result image (for local runs only)
    if not is_deployed and showResults:
        utils.show_img("Final Measurements for {}".format(imageName), new_drawing)
        write_new_image(imageName, new_drawing)
    else:
        if not is_deployed:
            write_new_image(imageName, new_drawing)


    return rescaled_image, targetLength, targetWidth, left_point, right_point, width_left_point, width_right_point, left_ref_object_point, right_ref_object_point, whichTechnique
    
def write_new_image(imageName, image):
    """ writing the image for local display/testing

    """
    basename = os.path.basename(imageName)
    path = os.path.dirname(imageName)
 
    out_path = os.path.join(path, "output")
    if not os.path.exists(out_path):
        os.mkdir(out_path)

    out_name = os.path.join(out_path, basename)
    cv2.imwrite(out_name, image)


def run_program():
    res = find_length(False, None)


if __name__ == "__main__":
    run_program()

