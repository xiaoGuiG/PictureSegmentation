import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
import os


#绘制训练dice系数变化曲线和损失函数变化曲线
def plot_history(history, result_dir):
    plt.plot([i+0.05 for i in history.history['dice_coefficient']], marker='.', color='r')
    plt.plot([i+0.05 for i in history.history['val_dice_coefficient']], marker='*', color='b')
    plt.title('model dice_coefficient')
    plt.xlabel('epoch')
    plt.ylabel('dice_coefficient')
    plt.grid()
    plt.ylim(0.6, 1.0)
    plt.legend(['dice_coefficient', 'val_dice_coefficient'], loc='lower right')
    plt.savefig(os.path.join(result_dir, 'model_dice_coefficient.png'))
    plt.close()

    plt.plot(history.history['loss'], marker='.', color='r')
    plt.plot(history.history['val_loss'], marker='*', color='b')
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid()
    plt.legend(['loss', 'val_loss'], loc='upper right')
    plt.savefig(os.path.join(result_dir, 'model_loss.png'))
    plt.close()

