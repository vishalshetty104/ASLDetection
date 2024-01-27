import cv2
import numpy as np
from tensorflow.keras.models import load_model
from keras_preprocessing.image import load_img, img_to_array

image_x, image_y = 64, 64
lb = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
      'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
classifier = load_model('model.h5')
cap = cv2.VideoCapture(0)

def Label(result):
    if result[0][0] == 1:
        return 'A'
    elif result[0][1] == 1:
        return 'B'
    elif result[0][2] == 1:
        return 'C'
    elif result[0][3] == 1:
        return 'D'
    elif result[0][4] == 1:
        return 'E'
    elif result[0][5] == 1:
        return 'F'
    elif result[0][6] == 1:
        return 'G'
    elif result[0][7] == 1:
        return 'H'
    elif result[0][8] == 1:
        return 'I'
    elif result[0][9] == 1:
        return 'J'
    elif result[0][10] == 1:
        return 'K'
    elif result[0][11] == 1:
        return 'L'
    elif result[0][12] == 1:
        return 'M'
    elif result[0][13] == 1:
        return 'N'
    elif result[0][14] == 1:
        return 'O'
    elif result[0][15] == 1:
        return 'P'
    elif result[0][16] == 1:
        return 'Q'
    elif result[0][17] == 1:
        return 'R'
    elif result[0][18] == 1:
        return 'S'
    elif result[0][19] == 1:
        return 'T'
    elif result[0][20] == 1:
        return 'U'
    elif result[0][21] == 1:
        return 'V'
    elif result[0][22] == 1:
        return 'W'
    elif result[0][23] == 1:
        return 'X'
    elif result[0][24] == 1:
        return 'Y'
    elif result[0][25] == 1:
        return 'Z'

res = ""
while True:
    ret, frame = cap.read()
    img = cv2.rectangle(frame, (300, 300), (10, 10), (0, 255, 0), 0)
    crop_img = img[10:300, 10:300]
    gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]
    cv2.imshow('frame', frame)
    
    if cv2.waitKey(1) == ord(' '):
        img = cv2.resize(thresh, (64, 64))
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        test_image = img_to_array(img)
        test_image = np.expand_dims(test_image, axis=0)
        print(test_image.shape)
        result = classifier.predict(test_image)
        print(result)
        output_lb = Label(result)
        print(output_lb)
        frame1 = frame.copy()
        res = res +" "+output_lb
        cv2.putText(frame1, res, (300, 100),  cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (255, 0, 0), 2)
        cv2.imshow('result', frame1)
    cv2.imshow('thresh', thresh)
    cv2.waitKey(1)

cv2.destroyAllWindows()
