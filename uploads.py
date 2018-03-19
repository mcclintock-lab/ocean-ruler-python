import cv2
import boto3

def do_dynamo_put(name, email, uuid, locCode, picDate, len_in_inches, rating, notes, 
        as_x, as_y, ae_x, ae_y,qs_x, qs_y, qe_x, qe_y):
    dynamodb = boto3.resource('dynamodb')
    table = dynamodb.Table('ab_length')
    try:
        lenfloat = round(float(len_in_inches),2)
    except StandardError:
        lenfloat = -1.0
    now = int(time.time()*1000)
    try:
        table.put_item(
            Item={
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
                "newsize":decimal.Decimal('{}'.format(lenfloat))
            }
        )
    except ClientError as e:
        print(e.response['Error']['Message'])
    else:
        print("{} length updated to {}".format(uuid, lenfloat))

def do_s3_upload(image_data, final_thumb, uuid):
    s3 = boto3.resource('s3')

    s3.Bucket('abalone').put_object(Key="full_size/"+uuid+".png", Body=image_data)
    print_time("done putting full size")

    #s3.Bucket('abalone').put_object(Key="thumbs/"+uuid+".png", Body=thumb)
    #print_time("done with thumb")
    s3.Bucket('abalone').put_object(Key="thumbs/"+uuid+".png", Body=final_thumb)
    print_time("don with thumb")



def upload_worker(rescaled_image, thumb, img_data, 
    name, email, uuid, locCode, picDate, abaloneLength, rating, notes,
    as_x, as_y, ae_x, ae_y, qs_x, qs_y, qe_x, qe_y):
    #print_time("uploading data now....")
    #final_image = cv2.imencode('.png', rescaled_image)[1].tostring()
    #print_time("done encoding image")
    do_dynamo_put(name, email, uuid, locCode, picDate, abaloneLength, rating, notes,
                 as_x, as_y, ae_x, ae_y, qs_x, qs_y, qe_x, qe_y)
    #print_time("done putting things into dynamo db")

    original_thumb_str = cv2.imencode('.png', thumb)[1].tostring()
    #print_time("done encoding thumb")
    final_thumb = get_thumbnail(rescaled_image)
    thumb_str = cv2.imencode('.png', final_thumb)[1].tostring()
    do_s3_upload(img_data, thumb_str, uuid)
    #do_s3_upload(None, thumb_str, None, uuid)
    print_time("done uploading data...")