
import boto3
import json
import csv

dynamodb = boto3.resource('dynamodb',region_name='us-west-2')
s3 = boto3.resource('s3', region_name='us-west-2')
s3Client = boto3.client('s3', region_name='us-west-2')
bucket_name = 'ocean-ruler-test'

resp = s3Client.list_objects_v2(Bucket=bucket_name)
contents = resp.get('Contents')

uuids = []
for item in contents:
    filename = item.get('Key')


    uuid = filename.replace("public/thumbs/", "")
    uuid = uuid.replace("thumbs/","")
    uuid = uuid.replace(".png", "")
    uuids.append(uuid)
    print("uuid: {}".format(uuid))
    if False:

        table = dynamodb.Table('ocean-ruler-test')

        try:
            response = table.get_item(
                Key={
                    'uuid': uuid,
                }
            )
        except Exception as e:
            continue

        else:
            item = response.get('Item')
            if item is not None:
                #print("GetItem succeeded: {}".format(item))
                fishery_type = item.get("fishery_type")
                #print("fishery type: {}".format(fishery_type))
                #print("fishery_type::: {}".format(fishery_type))
                
                if fishery_type == "lobster" or fishery_type == "scallop" or fishery_type == "Abalone":
                    print("found a {}, will delete".format(fishery_type))
                    print("deleting object from s3:")
                    s3Obj = s3.Object(bucket_name, filename)
                    res = s3Obj.delete()
                    print("deleting item from dynamo db::: ")
                    table.delete_item(
                        Key={
                            'uuid': uuid
                        }
                    )
            else:
                print("item is missing for {}".format(uuid))
                s3Obj = s3.Object(bucket_name, filename)
                res = s3Obj.delete()
                print('results of deletion: {}'.format(res))

'''
with open('all_uuids.csv', 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=' ',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for uuid in uuids:
        spamwriter.writerow([uuid])
'''
