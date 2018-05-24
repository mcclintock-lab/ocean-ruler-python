import cv2
import boto3
import time
import utils
import json
import decimal

class DecimalEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, decimal.Decimal):
            if o % 1 > 0:
                return float(o)
            else:
                return int(o)
        return super(DecimalEncoder, self).default(o)


def do_dynamo_put(name, email, uuid, locCode, picDate, len_in_inches, rating, notes, 
        as_x, as_y, ae_x, ae_y,qs_x, qs_y, qe_x, qe_y, fishery_type, ref_object, ref_object_size, ref_object_units, original_width, original_height):
    dynamodb = boto3.resource('dynamodb')
    table = dynamodb.Table('ocean_ruler')
    try:
        lenfloat = round(float(len_in_inches),2)
    except StandardError:
        lenfloat = -1.0
    now = int(time.time()*1000)
    print("fishery type: {}".format(fishery_type))
    print("ref object: {}".format(ref_object))
    print("units: {}".format(ref_object_units))
    print("size: {}".format(ref_object_size))
    print("picDate: {}".format(picDate))
    print("length in in: {}".format(lenfloat))
    print("rating: {}".format(rating))
    print("user submitted: {}".format(now))

    print("w: {}, h: {}".format(orig_width, orig_height))

    item = {
                'username': name,
                'email': email,
                'uuid': uuid,
                'locCode': locCode,
                'picDate': decimal.Decimal(picDate),
                'length_in_inches':decimal.Decimal('{}'.format(lenfloat)),
                'rating':decimal.Decimal('{}'.format(rating)),
                'usernotes': notes,
                'userSubmittedAt': decimal.Decimal(now),
                "ab_start_x": decimal.Decimal(as_x),
                "ab_start_y":decimal.Decimal(as_y),
                "ab_end_x":decimal.Decimal(ae_x),
                "ab_end_y":decimal.Decimal(ae_y),
                "ab_new_start_x": decimal.Decimal(as_x),
                "ab_new_start_y":decimal.Decimal(as_y),
                "ab_new_end_x":decimal.Decimal(ae_x),
                "ab_new_end_y":decimal.Decimal(ae_y),
                "q_start_x":decimal.Decimal(qs_x),
                "q_start_y":decimal.Decimal(qs_y),
                "q_end_x":decimal.Decimal(qe_x),
                "q_end_y":decimal.Decimal(qe_y),
                "q_new_start_x":decimal.Decimal(qs_x),
                "q_new_start_y":decimal.Decimal(qs_y),
                "q_new_end_x":decimal.Decimal(qe_x),
                "q_new_end_y":decimal.Decimal(qe_y),
                "newsize":decimal.Decimal('{}'.format(lenfloat)),
                "ref_object":ref_object,
                "ref_object_size":ref_object_size,
                "ref_object_units":ref_object_units,
                "orig_width":decimal.Decimal('{}'.format(original_width)),
                "orig_height":decimal.Decimal('{}'.format(original_height)),
                "fishery_type":fishery_type
            }

    print("items: {}".format(item))

    try:
        response = table.put_item(
            Item=item,ReturnValues='ALL_NEW'
        )
        print("--->>> response::: {}".format(response))

    except Exception as e:
        print("error uploading: {}".format(e))
    else:
        print("{} length updated to {}".format(uuid, lenfloat))

def do_s3_upload(image_data, final_thumb, uuid):
    s3 = boto3.resource('s3')

    s3.Bucket('abalone').put_object(Key="full_size/"+uuid+".png", Body=image_data)

    #s3.Bucket('abalone').put_object(Key="thumbs/"+uuid+".png", Body=thumb)
    #print_time("done with thumb")
    s3.Bucket('abalone').put_object(Key="thumbs/"+uuid+".png", Body=final_thumb)



def upload_worker(rescaled_image, thumb, img_data, 
    name, email, uuid, locCode, picDate, abaloneLength, rating, notes,
    as_x, as_y, ae_x, ae_y, qs_x, qs_y, qe_x, qe_y, fishery_type, ref_object, ref_object_size, ref_object_units, original_width, original_height):
    #print_time("uploading data now....")
    #final_image = cv2.imencode('.png', rescaled_image)[1].tostring()
    #print_time("done encoding image")
    do_dynamo_put(name, email, uuid, locCode, picDate, abaloneLength, rating, notes,
                 as_x, as_y, ae_x, ae_y, qs_x, qs_y, qe_x, qe_y, fishery_type, ref_object, ref_object_size, ref_object_units, original_width, original_height)

    original_thumb_str = cv2.imencode('.png', thumb)[1].tostring()
    #print_time("done encoding thumb")
    final_thumb = utils.get_thumbnail(rescaled_image)
    thumb_str = cv2.imencode('.png', final_thumb)[1].tostring()
    do_s3_upload(img_data, thumb_str, uuid)
    #do_s3_upload(None, thumb_str, None, uuid)