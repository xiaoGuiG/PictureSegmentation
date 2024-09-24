#level1、level2使用
from keras.models import load_model
from model_code.loss_function import dice_coefficient, dice_coefficient_loss
import numpy as np
import cv2

#将模型预测的标签转化为图像
def tensorToimg(img): #0,85,170,255
    row, column, channels = img.shape
    for i in range(row):
        for j in range(column):
            if img[i, j, 0] >= 0.5:
                img[i, j, 0] = 255
            elif img[i, j, 1] >= 0.5:
                img[i, j, 0] = 170
            elif img[i, j, 2] >= 0.5:
                img[i, j, 0] = 85
            else:
                img[i, j, 0] = 0
    return img[:, :, 0]

def dice_coff(label, predict):
    return np.sum(2*label*predict)/(np.sum(label)+np.sum(predict))

def predict(test_img,test_label,test_name_list,model_path):
    model = load_model(model_path, custom_objects = {'dice_coefficient': dice_coefficient, 'dice_coefficient_loss': dice_coefficient_loss})
    result = model.predict(test_img)
    print("the dice coefficient is: " + str(dice_coff(test_label, result)))
    length, row, column, channels = result.shape
    for i in range(length):
        final_img = tensorToimg(result[i])
        cv2.imwrite('dataset/test/predict/'+test_name_list[i], final_img)