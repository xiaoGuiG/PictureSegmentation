import util as util
from model_code.unet import unet
from tensorflow.python.keras.callbacks import ModelCheckpoint
from prediction.predict_basic import *
from image_loader.loader_basic import *
import sys


def level1():
    #读取数据基础版
    local_path = os.path.join(sys.path[0],"dataset")
    model_path = os.path.join(sys.path[0],"unet_segmention.hdf5")
    result_dir = os.path.join(sys.path[0],"result")
    train, train_label = load_image(local_path, "train")  # dataset为实际使用数据
    val, val_label = load_image(local_path, 'val')
    model = unet(input_size=(224, 224, 3))
    model_checkpoint = ModelCheckpoint(model_path, monitor='loss', verbose=1, save_best_only=True)
    history = model.fit(train, train_label, batch_size=8, epochs=1, callbacks=[model_checkpoint],
                        validation_data=(val, val_label))
    util.plot_history(history, result_dir)
    test_img, test_label, test_name_list = load_image(local_path, "test", need_name_list=True)
    predict(test_img, test_label, test_name_list, model_path)

if __name__ == "__main__":
    level1()