    '''
    denoised = cv2.fastNlMeansDenoisingColored(clipped_image,None,3,3,5,10)
    gray = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)
    edged_img = get_canny(denoised,0.33)
    '''
    clahe = cv2.createCLAHE(clipLimit=40.0, tileGridSize=(30,30))

    denoised = cv2.fastNlMeansDenoising(clipped_image,None,7,21,11)
    gray = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)
    #claheGray = clahe.apply(gray)
    thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,17,11)
    
    '''
    #equalized = cv2.equalizeHist(gray)
    img_yuv = cv2.cvtColor(clipped_image, cv2.COLOR_BGR2YUV)

    # equalize the histogram of the Y channel
    img_yuv[:,:,0] = clahe.apply(img_yuv[:,:,0])
    #img_yuv[:,:,0] = clahe.apply(img_yuv[:,:,0])
    # convert the YUV image back to RGB format
    img_bgr = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    '''
        if True:
        cv2.imshow("denoised", denoised)
        
        cv2.imshow("thresh", thresh)
        cv2.waitKey(0)
    if False:
        hsv_image = cv2.cvtColor(clipped_image, cv2.COLOR_BGR2HSV)
        color_edged_img = cv2.Canny(hsv_image, 0, 150,11) 
        auto_edged_img = get_canny(clipped_image)
        cv2.imshow("Original", clipped_image)
        cv2.imshow("Color, b&W, auto", np.hstack([color_edged_img, edged_img, auto_edged_img]))
        cv2.waitKey(0)