import os
import numpy as np
import argparse

def read_args():
    ap = argparse.ArgumentParser()
    
    ap.add_argument("--image", required=False,
        help="path to the input image")


    try:
        args = vars(ap.parse_args())
        if args['image'] is None:
            ap.add_argument('allimages', metavar='fp', nargs='+', help='file names')
            args = vars(ap.parse_known_args())
    except SystemExit:
        ap.add_argument('allimages', metavar='fp', nargs='+', help='file names')
        args = vars(ap.parse_args())  
    


    imageName = args["image"]
    
    if imageName is None or len(imageName) == 0:
        showResults = False
        out_file ="new_data.csv"
        #out_file = "data.csv"
        allImageNames = args['allimages'][0]
        print("working on {}".format(allImageNames))
        imageParts = allImageNames.split()

        if(len(imageParts) > 1):
            imageName = "{} {}".format(imageParts[0], imageParts[1])
        else:
            imageName = imageParts[0]

    targetPath, imgName = os.path.split(imageName)
    return targetPath, imgName
