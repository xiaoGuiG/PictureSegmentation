#level2、level3使用
import os
import numpy as np
import cv2

def image_process_enhanced(img):
    img = cv2.equalizeHist(img)#像素直方图均衡
    return img


def label_to_code(label_img):
    row, column, channels = label_img.shape
    for i in range(row):
        for j in range(column):
            if label_img[i, j, 0] >= 0.75:
                label_img[i, j, :] = [1, 0, 0]
            elif (label_img[i, j, 0] < 0.75) & (label_img[i, j, 0] >= 0.5):
                label_img[i, j, :] = [0, 1, 0]
            elif (label_img[i, j, 0] < 0.5) & (label_img[i, j, 0] >= 0.25):
                label_img[i, j, :] = [0, 0, 1]
    return label_img

def load_image(root, data_type, size = None, need_name_list = False, need_enhanced = False):
    image_path = os.path.join(root, data_type, "image")
    label_path = os.path.join(root, data_type, "label")
    print(image_path)

    image_list = []
    label_list = []
    image_name_list = []

    # ----------使用全部数据训练，容易超出内存，慎用-------------
#     for file in os.listdir(image_path):
#         image_file = os.path.join(image_path, file)
#         label_file_name = file.split(".")[0]+"_gt.png"
#         label_file = os.path.join(label_path, label_file_name)
#         if need_name_list is True:
#             image_name_list.append(file)
#         img = cv2.imread(image_file)
#         label = cv2.imread(label_file)
#         if size is not None:
#             row, column, channel = size
#             img = cv2.resize(img, (column, row, channel))
#             label = cv2.resize(label, (column, row, channel))
#         #对图像进行增强
#         if need_enhanced is True:
#             img = image_process_enhanced(img)
    
#         img = img / 255
#         label = label / 255
#         image_list.append(img)
#         label = label_to_code(label)#对标签进行编码
#         label_list.append(label)

    # ----------使用部分数据训练，建议使用-------------
    i = 0
    for file in os.listdir(image_path):
        if i < 20:
            image_file = os.path.join(image_path, file)
            label_file_name = file.split(".")[0] + "_gt.png"
            label_file = os.path.join(label_path, label_file_name)
            if need_name_list is True:
                image_name_list.append(file)
            img = cv2.imread(image_file)
            label = cv2.imread(label_file)
            if size is not None:
                row, column, channel = size
                img = cv2.resize(img, (column, row, channel))
                label = cv2.resize(label, (column, row, channel))
            #对图像进行增强
            if need_enhanced is True:
                img = image_process_enhanced(img)

            img = img / 255
            label = label / 255
            image_list.append(img)
            label = label_to_code(label)#对标签进行编码
            label_list.append(label)

            i += 1


    if need_name_list is True:
        return np.array(image_list), np.array(label_list), image_name_list
    else:
        return np.array(image_list), np.array(label_list)