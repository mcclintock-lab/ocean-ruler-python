import constants 
import boto3

def get_multiplier(fishery_type, dB):
    #old method of getting them based on pixel size
    
    multiplier = 1.1
    if(constants.isScallop(fishery_type)):
        multiplier = 1.03
    elif(fishery_type == constants.LETHRINUS_OLIVACEUS or fishery_type == constants.EPINEPHELUS_POLYPHEKADION):
        multiplier = 1.01
    else:
        
        #heuristic (aka fudge factor) for closeness to object
        #the bigger the ref object, the more zoomed in its assumed to be
        if dB < 60:
            multiplier = 1.05
        elif 60 <= dB <= 65:
            multiplier = 1.08
        elif 80 <= dB <= 150:
            multiplier = 1.12
        elif 150 <= dB <= 170:
            multiplier = 1.02
        elif 170 < dB <= 300:
            multiplier = 1.04
        elif dB > 300 and constants.isFinfish(fishery_type):

            #finfish closeups with 10 cm square...
            multiplier = 1.14
            
    return multiplier

def get_multiplier_from_db(fishery_type, dB):
    """ Actual code that loads the adjusters from the dynamodb 

    """
    dynamodb = boto3.resource('dynamodb')
    table = dynamodb.Table('ocean-ruler-depth-adjustment')
    try:
        response = table.get_item(Key={'fishery_type': fishery_type})
        depth_adjustment = response['Item']['depth_adjustment']
        return float(depth_adjustment)
    except Exception as e:
        print(e)
    
    return 1.1

