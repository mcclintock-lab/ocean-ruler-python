import cv2
import utils


def sort_by_matching_shape(target_contour, template_shape, use_hull,input_image, is_quarter=False):
    templateHull = cv2.convexHull(template_shape)
    templateArea = cv2.contourArea(template_shape)

    targetHull = cv2.convexHull(target_contour)
    targetArea = cv2.contourArea(target_contour)
    hullArea = cv2.contourArea(targetHull)

    hausdorffDistanceExtractor = cv2.createHausdorffDistanceExtractor()
    val = cv2.matchShapes(target_contour, template_shape, 2, 0.0)
    

    #doesn't work yet in python opencv - always 0
    #sd = cv2.createShapeContextDistanceExtractor()
    #shape_dist = sd.computeDistance(target_contour,template_shape)


    #hull_val = cv2.matchShapes(targetHull,template_shape,2,0.0)
    
    if is_quarter:
        area = (hullArea/templateArea)
        haus_dist = hausdorffDistanceExtractor.computeDistance(target_contour, template_shape)
    else:
        area = (targetArea /templateArea)
        haus_dist = hausdorffDistanceExtractor.computeDistance(target_contour, template_shape)
    
    #get centroid of template_shape
    template_X, template_Y = utils.get_centroid(template_shape)
    
    #if we use the hull, it loses its offset in x and y
    #need to return x,y
    #or, don't use hull -- use a closed shape
    
    if is_quarter:
        #get centroid of target contour hull
        target_X, target_Y = utils.get_centroid(targetHull)
        diffX = abs(template_X - target_X)
        diffY = abs(template_Y - target_Y)
        centroid_diff = diffX+diffY
    else:
        target_X, target_Y = utils.get_centroid(target_contour)
        diffX = abs(template_X - target_X)
        diffY = abs(template_Y - target_Y)
        centroid_diff = diffX+diffY
    
    #, diffX+diffY
    return target_contour, val, area, haus_dist, centroid_diff



