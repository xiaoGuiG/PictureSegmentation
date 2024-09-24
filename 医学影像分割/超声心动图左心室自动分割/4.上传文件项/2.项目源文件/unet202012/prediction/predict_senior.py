#level3使用
from keras.models import load_model
from model_code.loss_function import dice_coefficient, dice_coefficient_loss
import cv2
import time
import numpy as np

#将模型预测的标签转化为图像
def tensorToimg(img): #0,85,170,255
    row, column, channels = img.shape
    for i in range(row):
        for j in range(column):
            #if img[i, j, 0] > 0.5:
                #print(img[i, j, :])
            if img[i, j, 0] >= 0.5:
                img[i, j, 0] = 255
            elif img[i, j, 1] >= 0.5:
                img[i, j, 0] = 170
            elif img[i, j, 2] >= 0.5:
                img[i, j, 0] = 85
            else:
                img[i, j, 0] = 0
    return img[:, :, 0]

def pixel_accuracy(label, predict):
    start_time = time.time()
    length, row, column, channels = label.shape
    true_pixel = 0
    all_pixels = length*row*column
    for i in range(length):
        for j in range(row):
            for m in range(column):
                predict_cate = category(predict[i, j, m, :])
                label_cate = category(label[i, j, m, :])
                if predict_cate == label_cate:
                        true_pixel = true_pixel + 1

    end_time = time.time()
    print("the pixel_accuracy is: " + str(true_pixel/all_pixels))
    print("compute pixel_accuracy use time: "+str(end_time-start_time))
    return true_pixel/all_pixels

def mean_pixel_accuracy(label, predict, class_num = 4):
    start_time = time.time()
    length, row, column, channels = label.shape
    class_list = np.zeros(class_num)
    insaction_list = np.zeros(class_num)
    for i in range(length):
        for j in range(row):
            for m in range(column):
                predict_cate = category(predict[i, j, m, :])
                label_cate = category(label[i, j, m, :])
                for n in range(class_num):
                    if label_cate == n:
                        class_list[n] = class_list[n] + 1
                        if predict_cate == n:
                            insaction_list[n] = insaction_list[n] + 1
                        break
    end_time = time.time()
    mean_pixel_accuracy = 0
    for i in range(class_num):
        mean_pixel_accuracy += insaction_list[i]/class_list[i]
    mean_pixel_accuracy = mean_pixel_accuracy/class_num
    print("the mean pixel accuracy is: " + str(mean_pixel_accuracy))
    print("compute mean pixel accuracy use time: "+str(end_time-start_time))
    return mean_pixel_accuracy

def compute_mIoU(label, predict, class_num = 4):
    start_time = time.time()
    length, row, column, channels = label.shape
    class_list = np.zeros(class_num)
    insaction_list = np.zeros(class_num)
    for i in range(length):
        for j in range(row):
            for m in range(column):
                predict_cate = category(predict[i, j, m, :])
                label_cate = category(label[i, j, m, :])
                for n in range(class_num):
                    if label_cate == n | predict_cate == n:
                        class_list[n] = class_list[n] + 1
                        if predict_cate == label_cate:
                            insaction_list[n] = insaction_list[n] + 1
    mIoU = 0
    for i in range(class_num):
        mIoU += insaction_list[i] / class_list[i]
    mIoU = mIoU / class_num
    end_time = time.time()
    print("the mIoU is: " + str(mIoU))
    print("compute_mIoU use time: "+str(end_time-start_time))
    return mIoU

def dice_coff(label, predict):
    return np.sum(2*label*predict)/(np.sum(label)+np.sum(predict))

def category(img):
    if img[0] >= 0.5:
        return 1
    elif img[1] >= 0.5:
        return 2
    elif img[2] >= 0.5:
        return 3
    else:
        return 0

def predict(test_img, test_label, test_name_list,model_path):
    model = load_model(model_path, custom_objects = {'dice_coefficient': dice_coefficient, 'dice_coefficient_loss': dice_coefficient_loss})
    result = model.predict(test_img)
    print("the dice coefficient is: " + str(dice_coff(test_label, result)))
    pixel_accuracy(test_label, result)
    mean_pixel_accuracy(test_label, result)
    compute_mIoU(test_label, result)
    length, row, column, channels = result.shape
    for i in range(length):
        final_img = tensorToimg(result[i])
        cv2.imwrite('dataset/test/predict/'+test_name_list[i], final_img)