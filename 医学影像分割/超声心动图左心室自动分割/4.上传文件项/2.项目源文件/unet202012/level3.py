import util as util
from model_code.transfer_unet import VGG16_unet_model
from tensorflow.python.keras.callbacks import ModelCheckpoint
from prediction.predict_senior import *
from image_loader.loader_senior import *
import os
import sys
import cv2
import numpy as np

def level3():
    img_enhanced = False
    local_path = os.path.join(sys.path[0], "dataset")
    model_path = os.path.join(sys.path[0], "unet_segmentation_best.hdf5")  # 改为保存最优模型
    result_dir = os.path.join(sys.path[0], "result")
    
    # 加载训练集和验证集
    train, train_label = load_image(local_path, "train", need_enhanced=img_enhanced)
    val, val_label = load_image(local_path, 'val', need_enhanced=img_enhanced)
    
    # 训练模型
    model = VGG16_unet_model(input_size=(224, 224, 3), if_transfer=True, if_local=True)
    model_checkpoint = ModelCheckpoint(model_path, monitor='loss', verbose=1, save_best_only=True)
    history = model.fit(train, train_label, batch_size=4, epochs=3, callbacks=[model_checkpoint],
                        validation_data=(val, val_label))
    
    # 绘制训练历史
    util.plot_history(history, result_dir)
    
    # 预测阶段
    test_img, test_label, test_name_list = load_image(local_path, "test", need_name_list=True)
    
    # 加载训练好的最佳模型进行预测
    predict(test_img, test_label, test_name_list, model_path)
    
    # 输出对比图
    output_comparison_images(test_img, test_label, test_name_list, model_path, result_dir)

def output_comparison_images(test_img, test_label, test_name_list, model_path, result_dir):
    """
    生成并保存真实标签与预测结果的对比图
    """
    model = load_model(model_path, custom_objects={'dice_coefficient': dice_coefficient, 'dice_coefficient_loss': dice_coefficient_loss})
    predictions = model.predict(test_img)
    
    for i in range(len(test_img)):
        # 将预测结果和真实标签转化为可视化的图像格式
        predicted_img = tensorToimg(predictions[i])
        true_label_img = tensorToimg(test_label[i])
        original_img = (test_img[i] * 255).astype(np.uint8)  # 原图恢复为可视化
        
        # 将 2D 标签图像扩展为 3D，使其兼容拼接
        true_label_img_3d = np.stack([true_label_img] * 3, axis=-1)  # 复制三次形成 RGB 通道
        predicted_img_3d = np.stack([predicted_img] * 3, axis=-1)    # 复制三次形成 RGB 通道
        
        # 将三张图像拼接到一起
        comparison_img = np.hstack((original_img, true_label_img_3d, predicted_img_3d))
        
        # 保存对比图
        save_path = os.path.join(result_dir, f"comparison_{test_name_list[i]}")
        cv2.imwrite(save_path, comparison_img)
        print(f"Saved comparison image to {save_path}")


if __name__ == "__main__":
    level3()
