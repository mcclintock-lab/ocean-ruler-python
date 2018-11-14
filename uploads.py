import cv2
import boto3
import time
import utils
import json
import decimal
import utils
import math

class DecimalEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, decimal.Decimal):
            if o % 1 > 0:
                return float(o)
            else:
                return int(o)
        return super(DecimalEncoder, self).default(o)


def do_dynamo_put(name, email, uuid, locCode, picDate, len_in_inches, rating, notes, 
        as_x, as_y, ae_x, ae_y,qs_x, qs_y, qe_x, qe_y, fishery_type, ref_object, ref_object_size, 
        ref_object_units, original_width, original_height, dynamo_table_name, original_filename, original_size,
        targetWidth, asw_x, asw_y, aew_x, aew_y):
    dynamodb = boto3.resource('dynamodb')
    table = dynamodb.Table(dynamo_table_name)
    try:
        lenfloat = round(float(len_in_inches),2)
    except StandardError:
        lenfloat = -1.0

    try:
        widthfloat = round(float(targetWidth),2)
    except StandardError:
        widthfloat = -1.0

    now = int(time.time()*1000)
    try:
        if original_size is None or original_size == "undefined" or len(original_size) == 0 or math.isnan(float(original_size)):
            original_size = 0
    except Error:
        original_size = 0
   
    try:
        response = table.put_item(
            Item={
                'username': name,
                'email': email,
                'uuid': uuid,
                'locCode': locCode,
                'picDate': decimal.Decimal(picDate),
                'length_in_inches':decimal.Decimal('{}'.format(lenfloat)),
                'rating':decimal.Decimal('{}'.format(rating)),
                'usernotes': notes,
                'userSubmittedAt': decimal.Decimal('{}'.format(now)),
                "ab_start_x": decimal.Decimal('{}'.format(as_x)),
                "ab_start_y":decimal.Decimal('{}'.format(as_y)),
                "ab_end_x":decimal.Decimal('{}'.format(ae_x)),
                "ab_end_y":decimal.Decimal('{}'.format(ae_y)),
                "ab_new_start_x": decimal.Decimal('{}'.format(as_x)),
                "ab_new_start_y":decimal.Decimal('{}'.format(as_y)),
                "ab_new_end_x":decimal.Decimal('{}'.format(ae_x)),
                "ab_new_end_y":decimal.Decimal('{}'.format(ae_y)),
                "q_start_x":decimal.Decimal('{}'.format(qs_x)),
                "q_start_y":decimal.Decimal('{}'.format(qs_y)),
                "q_end_x":decimal.Decimal('{}'.format(qe_x)),
                "q_end_y":decimal.Decimal('{}'.format(qe_y)),
                "q_new_start_x":decimal.Decimal('{}'.format(qs_x)),
                "q_new_start_y":decimal.Decimal('{}'.format(qs_y)),
                "q_new_end_x":decimal.Decimal('{}'.format(qe_x)),
                "q_new_end_y":decimal.Decimal('{}'.format(qe_y)),
                "newsize":decimal.Decimal('{}'.format(lenfloat)),
                "ref_object":ref_object,
                "ref_object_size":decimal.Decimal('{}'.format(ref_object_size)),
                "ref_object_units":ref_object_units,
                "orig_width":decimal.Decimal('{}'.format(original_width)),
                "orig_height":decimal.Decimal('{}'.format(original_height)),
                "fishery_type":fishery_type,
                "original_filename":original_filename,
                "original_size":decimal.Decimal('{}'.format(original_size)),
                "width_in_inches": decimal.Decimal('{}'.format(widthfloat)),
                "target_width_start_x": decimal.Decimal('{}'.format(asw_x)),
                "target_width_start_y":decimal.Decimal('{}'.format(asw_y)),
                "target_width_end_x":decimal.Decimal('{}'.format(aew_x)),
                "target_width_end_y":decimal.Decimal('{}'.format(aew_y)),
                "target_width_new_start_x": decimal.Decimal('{}'.format(asw_x)),
                "target_width_new_start_y":decimal.Decimal('{}'.format(asw_y)),
                "target_width_new_end_x":decimal.Decimal('{}'.format(aew_x)),
                "target_width_new_end_y":decimal.Decimal('{}'.format(aew_y)),
                "newwidth":decimal.Decimal('{}'.format(original_width))
            }
        )

    except Exception as e:
        print("error uploading: {}".format(e))


def do_s3_upload(image_data, final_thumb, uuid, bucket_name):
    s3 = boto3.resource('s3')

    s3Client = boto3.client('s3')
    presigned_uuid = "public/full_size/"+uuid+".png"
    s3.Bucket(bucket_name).put_object(Key="public/full_size/"+uuid+".png", Body=image_data)
    presigned_url = s3Client.generate_presigned_url('get_object', Params = {'Bucket': bucket_name, 'Key': presigned_uuid}, ExpiresIn = 3600)
    
    s3.Bucket(bucket_name).put_object(Key="public/thumbs/"+uuid+".png", Body=final_thumb)
    return presigned_url

def upload_worker(rescaled_image, thumb, img_data, 
    name, email, uuid, locCode, picDate, abaloneLength, rating, notes,
    as_x, as_y, ae_x, ae_y, qs_x, qs_y, qe_x, qe_y, fishery_type, ref_object, ref_object_size, 
    ref_object_units, original_width, original_height, dynamo_table_name, bucket_name, original_filename, original_size,
    targetWidth, asw_x, asw_y, aew_x, aew_y):
    s3 = boto3.resource('s3')
    s3Client = boto3.client('s3')
    bucket_name = 'ocean-ruler-test';
    do_dynamo_put(name, email, uuid, locCode, picDate, abaloneLength, rating, notes,
                 as_x, as_y, ae_x, ae_y, qs_x, qs_y, qe_x, qe_y, fishery_type, ref_object, 
                 ref_object_size, ref_object_units, original_width, original_height, dynamo_table_name, original_filename, original_size,
                 targetWidth, asw_x, asw_y, aew_x, aew_y)
    presigned_url = s3Client.generate_presigned_url('get_object', Params = {'Bucket': bucket_name, 'Key': uuid}, ExpiresIn = 3600)
    return presigned_url

