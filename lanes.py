import cv2
import numpy as np
import matplotlib.pyplot as plt


def canny(image):
    # step 1
    # Use canny to detect edges by finding steep gradients

    # covert to gray
    grey_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Apply gaussian blurr to reduce noise
    # gets called by canny
    blur = cv2.GaussianBlur(grey_image,(5,5),0)
    canny = cv2.Canny(blur, 50, 150)
    return canny

def region_of_interest(image):
    # return closed region of field of view
    height = image.shape[0]
    triangle = np.array([(200,height), (1100, height), (550, 250)])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, np.int64([triangle]), 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def displayLines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1,y1,x2,y2 = line.reshape(4)
            cv2.line(line_image, (x1,y1), (x2,y2),(255, 0, 0), 10)

    return line_image

def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1 * (3/5))
    x1 = int((y1 - intercept )/ slope)
    x2 = int((y2 - intercept)/slope)
    return np.array([x1,y1,x2,y2])

def averageSlopIntercept(image, lines):
    left_fit = []
    right_fit = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1,x2),(y1,y2),1)
        slope = parameters[0]
        intercept = parameters[1]

        if (slope<0):
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))
    left_fit_average = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)

    left_line = make_coordinates(image, left_fit_average)
    right_line = make_coordinates(image, right_fit_average)
    return np.array([left_line, right_line])


# image = cv2.imread('test_image.jpg')
# canny = canny(np.copy(image))
# croppedImage = region_of_interest(canny)
# lines=cv2.HoughLinesP(croppedImage, 2, np.pi/180, 100,np.array([]),minLineLength=40, maxLineGap=5)
# averaged_lines = averageSlopIntercept(np.copy(image), lines)
# line_image = displayLines(np.copy(image), averaged_lines)
# # add lines to original image
# combo_image = cv2.addWeighted(np.copy(image), 0.8, line_image, 1, 1)
# cv2.imshow('result', combo_image)
# cv2.waitKey(0)

cap = cv2.VideoCapture('test2.mp4')
while (cap.isOpened()):
    k, frame = cap.read()
    if (frame==None):
        break
    print(frame)
    canny = canny(np.copy(frame))
    croppedImage = region_of_interest(canny)
    lines=cv2.HoughLinesP(croppedImage, 2, np.pi/180, 100,np.array([]),minLineLength=40, maxLineGap=5)
    averaged_lines = averageSlopIntercept(np.copy(frame), lines)
    line_image = displayLines(np.copy(frame), averaged_lines)
    # add lines to original image
    combo_image = cv2.addWeighted(np.copy(frame), 0.8, line_image, 1, 1)
    cv2.imshow('result', combo_image)
    if cv2.waitKey(1) == ord('q'):
        break;

cap.release()
cv2.destroyAllWindows()
