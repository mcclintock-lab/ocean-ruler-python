import cv2
import utils
import boto3
import argparse

def do_s3_upload():
    bucket_name = 'ocean-ruler-test';
    ap = argparse.ArgumentParser()
    args = ap.parse_known_args()[1]
    
    imageName = args[0]
    uuid = args[1]

    img = cv2.imread(imageName)
    thumb = utils.get_thumbnail(img)
    
    image_data = cv2.imencode('.png', img)[1].tostring()
    final_thumb = cv2.imencode('.png', thumb)[1].tostring()

    s3 = boto3.resource('s3')
    #s3Client = boto3.client('s3')
    #uuid = "full_size/"+uuid+".png"
    presigned_uuid = "public/full_size/"+uuid+".png"
    s3.Bucket(bucket_name).put_object(Key="public/full_size/"+uuid+".png", Body=image_data)

    #s3.Bucket('abalone').put_object(Key="thumbs/"+uuid+".png", Body=thumb)
    #print_time("done with thumb")
    s3.Bucket(bucket_name).put_object(Key="public/thumbs/"+uuid+".png", Body=final_thumb)

do_s3_upload();