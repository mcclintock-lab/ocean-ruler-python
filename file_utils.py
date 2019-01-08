import csv
import os
import numpy as np
import argparse
import constants
import utils

DELIM = ","
QUOTECHAR = '|'
INCHES_TO_MM = 25.4;
INCHES_TO_CM = 2.54;
INCHES_VAL = "inches";
MM_VAL = "mm";
CM_VAL = "cm";

def read_args():

    ap = argparse.ArgumentParser()
    args = ap.parse_known_args()[1]
    ap.add_argument("--image", required=False,
        help="path to the input image")
    ap.add_argument("--show", required=False,
        help="show the results. if not set, the results will write to a csv file")
    ap.add_argument("--output_file", required=False,
        help="file to read/write results from")
    ap.add_argument("--fishery_type", required=False, help="fishery type, e.g. abalone, lobster, etc.")
    ap.add_argument("--ref_object", required=False, help="reference object type: quarter or square")
    ap.add_argument("--ref_object_size", required=False, help="reference object size, 0.955 for quarter (default), squares of 2 or 3")
    ap.add_argument("--ref_object_units", required=False, help="reference object units, inches or mm, defaults to inches")

    try:

        args = vars(ap.parse_args())
        if args['image'] is None:
            ap.add_argument('allimages', metavar='fp', nargs='+', help='file names')
            args = vars(ap.parse_known_args())
    except SystemExit as err:
        ap.add_argument('allimages', metavar='fp', nargs='+', help='file names')
        args = vars(ap.parse_args())  
    
    showResults = args["show"]
    showResults = showResults == "True"
    

    out_file = args['output_file']
    if not out_file:
        out_file ="data_617.csv"


    imageName = args["image"]
    
    if imageName is None or len(imageName) == 0:
        showResults = False
        out_file ="new_data.csv"
        #out_file = "data.csv"
        allImageNames = args['allimages'][0]
        imageParts = allImageNames.split()

        if(len(imageParts) > 1):
            imageName = "{} {}".format(imageParts[0], imageParts[1])
        else:
            imageName = imageParts[0]

    fishery_type = args['fishery_type']
    if fishery_type is None or len(fishery_type) == 0:
        fishery_type = constants.ABALONE

    ref_object = args['ref_object']
    ref_object_size = args['ref_object_size']
    ref_object_units = args['ref_object_units']


    if ref_object is None or len(ref_object) == 0:
        if constants.isLobster(fishery_type):
            ref_object = constants.SQUARE
            ref_object_size = constants.DEF_SQUARE_SIZE_IN
            ref_object_units = constants.INCHES
        else:
            ref_object = constants.QUARTER
            ref_object_size = constants.QUARTER_SIZE_IN
            ref_object_units = constants.INCHES

    return imageName, showResults, out_file, fishery_type, ref_object, ref_object_size, ref_object_units

def shouldIgnore(imageName):
    if imageName.startswith("617_data/FrankPhotos"):
        #Glass_Beach_Memorial_Day_ - 2_203.jpg
        filename = imageName.split("/")
        parts = filename[2].split("-")
        sizeparts = parts[1]
        size_str = sizeparts.replace(".jpg","")
        nparts = size_str.split("_")
        if len(nparts) == 1:

            return True
        else:
            return False

    elif imageName.startswith("617_data/JonPhotos"):
        #IMG_20170528_085310.jpg
        filename = imageName.split("/")
        parts = filename[2].split("_")
        if len(parts) == 3:
            return True
        else:
            return False
    else:
        return False
    return False


def getOriginalSizeFromFilename(filename, ref_object_units):
    originalSize = 0.0
    try:
        if filename.startswith("Glass_Beach_Memorial"):
            parts = filename.split("-")
            sizeparts = parts[1]
            size_str = sizeparts.replace(".jpg","")
            nparts = size_str.split("_")
            size = float(nparts[1])
            originalSize = size*INCHES_TO_MM
        else:
            print("filename: {}".format(filename))
            parts = filename.split("_")
            

        originalSize = getSize(size_str, ref_object_units)
        print("size str: {}".format(size_str))
        
    except Exception:
        try:
            parts = filename.split("_")
            lengthOfName = len(parts)
            size_str = parts[lengthOfName-1].replace(".JPG", "")

            originalSize = getSize(size_str, ref_object_units)

        except Exception as e:
            print("something went wrong on parsing: ", e)
            originalSize = 0.0

    return originalSize


def getSize(size_str, ref_object_units):
    originalSize = 0.0
    if ref_object_units == INCHES_VAL:
        originalSize = float(size_str)
    elif ref_object_units == MM_VAL:
        originalSize = float(size_str)/INCHES_TO_MM
    elif ref_object_units == CM_VAL:
        originalSize = float(size_str)/INCHES_TO_CM
    else:
        originalSize = float(size_str)

    return originalSize


def get_real_size(imageName):
    #IMG_8.93_60.jpg
    print("image name: {}".format(imageName))
    if imageName.startswith("../ocean_ruler_images/abalone/"):
        #Glass_Beach_Memorial_Day_ - 2_203.jpg
        filename = imageName.split("/")
        print("filename: {}".format(filename))
        parts = filename[4].split("-")
        print("parts: {}".format(parts))
        sizeparts = parts[1]
        
        size_str = sizeparts.replace(".jpg","")
   
        nparts = size_str.split("_")
    
        size = float(nparts[1])
        print("real size: {}".format(size))
        return size*0.0393701
    elif imageName.startswith("617_data/JonPhotos"):
        #IMG_20170528_085310.jpg
        filename = imageName.split("/")
        parts = filename[2].split("_")

        size_str = parts[3].replace(".jpg","")
        size = float(size_str)*0.0393701
        return size
    elif imageName.startswith("feb_2017") or imageName.startswith("may_2017"):
        filename = imageName.split("/")

        parts = filename[1].split("_")
        if imageName.startswith("may_2017"):
            size_str = parts[1].replace(".JPG", "")
        else:
            size_str = parts[1]
        try:
            size = float(size_str)
            return size
        except Exception:
            return read_real_sizes(imageName)
    else:
        filename = imageName.split("/")
        
        return read_real_sizes(filename[len(filename)-1])
    

def read_real_sizes(imageName):

    real_sizes = {}
    real_sizes_file = "data/real_sizes.csv"
    size = -1.0
    with open(real_sizes_file, 'rU') as csvfile:

        csvreader = csv.reader(csvfile, delimiter=DELIM, quotechar=QUOTECHAR)
        try:
            for row in csvreader:
                name = row[0]
                currSize = row[1]
                name = name.replace(":", "_")
                imageName = imageName.replace(".jpg","")
                imageName = imageName.replace(".JPG","")
                imageName = imageName.replace("white/","")
                imageName = imageName.replace("blue/","")
                if name == imageName:
                    size = float(currSize)
                    return size

        except Exception:
            utils.print_time("can't read real files: {}".format(e),0)

    return size

def read_write_simple_csv(out_file, imageName, abaloneLength, refObjectUnits):
    all_rows = {}
    all_diffs = {}
    last_total_diff = 0.0
    total_diffs = 0.0
    if out_file == None:
        out_file = "data_617.csv"

    if os.path.exists(out_file):
        with open(out_file, 'rU') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=DELIM, quotechar=QUOTECHAR)
            try:
                for i, row in enumerate(csvreader):
                    if i > 0:
                        name = row[0]
                        size = row[1]
                        real_size = row[2]
                        diff = row[3]
                        avg = row[4]
                        if name != "Total":
                            #print "for {}, best ab key: {}, best ruler key: {}".format(name, best_ab_key, best_ruler_key)
                            all_rows[name] = [size, real_size, diff, avg]
                            all_diffs[name] = float(diff)
                        else:
                            last_total_diff = float(diff)
            except Exception as e:
                utils.print_time("problem here: {}".format(e),0)


        try:
            #real_size = get_real_size(imageName)
            real_size = getOriginalSizeFromFilename(imageName, refObjectUnits)
            print("real size: {}".format(real_size))
            if real_size > 0.0:
                diff = abs(abaloneLength - real_size)
                all_rows[imageName] = [abaloneLength, real_size, diff]
                
                all_diffs[imageName] = abs(diff)

                #total_diffs = np.sum(all_diffs.values())
                total_diffs = sum((abs(d) for d in all_diffs.values()))
                total_avg = total_diffs/len(all_diffs.values())
                with open(out_file, 'w') as csvfile:
                    writer = csv.writer(csvfile, delimiter=DELIM, quotechar=QUOTECHAR, quoting=csv.QUOTE_MINIMAL)
                    writer.writerow(["Name", "Estimated", "Real", "Val Diff", "Average"])
                    for name, sizes in all_rows.items():
                        diff = all_diffs.get(name)
                        est_size = sizes[0]
                        real_size = sizes[1]
                        avg = 0
                        row = [name, est_size, real_size, diff,avg]
                        writer.writerow(row)

                    writer.writerow(["Total", 0,0,total_diffs, total_avg])

        except Exception as err:
            utils.print_time("error trying to write the real size and diff: {}".format(err),0)

def read_write_csv(out_file, imageName, bestAbaloneKey, bestRulerKey, abaloneLength, rulerLength, rulerValue, background_val_diff):

    all_rows = {}
    all_diffs = {}
    last_total_diff = 0.0
    total_diffs = 0.0
    if os.path.exists(out_file):
        with open(out_file, 'rU') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=DELIM, quotechar=QUOTECHAR)
            try:
                for i, row in enumerate(csvreader):
                    if i > 0:
                        name = row[0]
                        size = row[1]
                        real_size = row[2]
                        diff = row[3]

                        best_ab_key = row[4]
                        best_ruler_key = row[5]
                        val_diff = row[6]
                        if name != "Total":
                            #print "for {}, best ab key: {}, best ruler key: {}".format(name, best_ab_key, best_ruler_key)
                            all_rows[name] = [size, real_size, best_ab_key, best_ruler_key, val_diff]
                            all_diffs[name] = float(diff)
                        else:
                            last_total_diff = float(diff)

            except Exception:
                utils.print_time("problem here: {}".format(e),0)

    try:
        real_size = get_real_size(imageName)
        if real_size > 0.0:
            diff = abs(abaloneLength - real_size)
            all_rows[imageName] = [abaloneLength, real_size, bestAbaloneKey, bestRulerKey, background_val_diff]
            
            all_diffs[imageName] = abs(diff)
            #total_diffs = np.sum(all_diffs.values())
            total_diffs = sum((abs(d) for d in all_diffs.values()))
            with open(out_file, 'wb') as csvfile:
                writer = csv.writer(csvfile, delimiter=DELIM, quotechar=QUOTECHAR, quoting=csv.QUOTE_MINIMAL)
                writer.writerow(["Name", "Estimated", "Real", "Difference %","Best Abalone Match","Best Ruler Match","Val Diff"])
                for name, sizes in all_rows.items():
                    diff = all_diffs.get(name)
                    est_size = sizes[0]
                    real_size = sizes[1]
                    ab_key = sizes[2]
                    ruler_key = sizes[3]
                    valDiff = sizes[4]
                    writer.writerow([name, est_size, real_size, diff,ab_key,ruler_key, valDiff])

                writer.writerow(["Total", 0,0,total_diffs,"-","-","-"])

            
    except Exception as err:
        utils.print_time("error trying to write the real size and diff: {}".format(err),0)
