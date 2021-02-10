import cv2
import utils
import boto3
import argparse

def do_s3_upload():
    bucket_name = 'oceanruler-tnc-images-prodops';
    ap = argparse.ArgumentParser()
    args = ap.parse_known_args()[1]
    
    imageName = args[0]
    uuid = args[1]

    img = cv2.imread(imageName)
    thumb = utils.get_thumbnail(img)
    
    image_data = cv2.imencode('.png', img)[1].tostring()
    final_thumb = cv2.imencode('.png', thumb)[1].tostring()
    

    '''
    #Dont need this anymore with the ec2 instance profile
    try:
        sts_client = boto3.client('sts')
        identity = sts_client.get_caller_identity()
        
        if not identity.get('Arn').startswith("arn:aws:sts::719729260530:assumed-role/poseidon"):
            # Call the assume_role method of the STSConnection object and pass the role
            # ARN and a role session name.
            assumed_role_object=sts_client.assume_role(
                RoleArn="arn:aws:iam::719729260530:role/poseidon",
                RoleSessionName="AssumeRoleSession1"
            )

            # From the response that contains the assumed role, get the temporary 
            # credentials that can be used to make subsequent API calls
            credentials=assumed_role_object['Credentials']

            # Use the temporary credentials that AssumeRole returns to make a 
            # connection to Amazon S3  
            s3=boto3.resource(
                's3',
                aws_access_key_id=credentials['AccessKeyId'],
                aws_secret_access_key=credentials['SecretAccessKey'],
                aws_session_token=credentials['SessionToken'],
            )
            print("identity assumed!")
        else:
            print("already using the assumed role...")
            s3 = boto3.resource('s3')
    except Exception as e:
        print("error assuming identity: ", e)
        s3 = boto3.resource('s3')
    '''
    try:
        s3 = boto3.resource('s3', region_name='us-west-2')
        #s3Client = boto3.client('s3')
        #uuid = "full_size/"+uuid+".png"
        
        presigned_uuid = "public/full_size/"+uuid+".png"
        s3.Bucket(bucket_name).put_object(Key="public/full_size/"+uuid+".png", Body=image_data, ACL='public-read')

        #s3.Bucket('abalone').put_object(Key="thumbs/"+uuid+".png", Body=thumb)
        #print_time("done with thumb")
        s3.Bucket(bucket_name).put_object(Key="public/thumbs/"+uuid+".png", Body=final_thumb, ACL='public-read')
    except Error as e:
        print("error trying to upload: ", e)
do_s3_upload();