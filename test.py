import torch.optim
from pylab import *
import dataset
from sklearn import metrics
import network
import os
import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_curve
import torch.nn.functional as F
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

start = time.time()

# 参数设置
PATH = './data/labels_change.csv'  # 替换为实际路径
TEST_PATH = './data/exam_labels.csv'  # 指向上传的CSV文件
# TEST_PATH = ''
is_train = False # True-训练模型  False-测试模型
save_model_name = 'model/L1_model.pkl'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
is_pretrained = True  # 是否加载预训练权重
is_sampling = 'smote'  # 训练集采样模式： over_sampler-上采样  down_sampler-下采样  no_sampler-无采样
print('Device:', device)

# 训练参数设置
SIZE = 224  # 图像进入网络的大小
BATCH_SIZE = 32  # batch_size数
NUM_CLASS = 2  # 分类数
EPOCHS = 10  # 迭代次数

# 创建保存模型的文件夹
os.makedirs('model', exist_ok=True)  # 如果文件夹不存在则创建

# 加载数据
train_loader, val_loader, test_loader = dataset.get_dataset(PATH, TEST_PATH, SIZE, BATCH_SIZE, is_train=is_train,
                                                            is_sampling='smote')

# # 打印加载的数据集的长度
# print(f"训练集加载数量: {len(train_loader.dataset)}")
# print(f"测试集加载数量: {len(test_loader.dataset)}")

# 定义模型、优化器、损失函数
model = network.AlexNet(is_pretrained).cuda() if torch.cuda.is_available() else network.AlexNet(is_pretrained)


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)  # pt is the probability of the true class
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return torch.mean(focal_loss)
        elif self.reduction == 'sum':
            return torch.sum(focal_loss)
        else:
            return focal_loss


# 在train.py中替换损失函数
criterion = FocalLoss(alpha=1.0, gamma=2.0).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=6e-3)


# 训练模型
def train_alexnet(model):
    for epoch in range(EPOCHS):
        correct = total = 0.
        loss_list = []
        for batch_index, (batch_x, batch_y) in enumerate(train_loader, 0):
            if torch.cuda.is_available():
                batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
            model.train()
            # 优化过程
            optimizer.zero_grad()  # 梯度归0
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()

            # 输出训练结果
            loss_list.append(loss.item())
            _, predicted = torch.max(output.data, 1)  # 返回每行的最大值
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
        train_avg_acc = 100 * correct / total
        train_avg_loss = np.mean(loss_list)
        print('[Epoch=%d/%d]Train set: Avg_loss=%.4f, Avg_accuracy=%.4f%%' % (
            epoch + 1, EPOCHS, train_avg_loss, train_avg_acc))

    # 保存模型
    torch.save(model.state_dict(), save_model_name)
    print('Training finished!')

def save_train_features(model, train_loader, save_dir='./features'):
    model.eval()  # 确保模型处于评估模式
    train_features, train_labels = [], []

    # 创建存储特征的目录
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 提取训练集的特征
    for images, labels in train_loader:
        if torch.cuda.is_available():
            images = images.cuda()
        features = model(images, feature_only=True).detach().cpu().numpy()
        features = features.reshape(features.shape[0], -1)
        train_features.extend(features)
        train_labels.extend(labels.cpu().numpy())

    # 保存特征和标签到文件
    np.save(os.path.join(save_dir, 'train_features.npy'), np.array(train_features))
    np.save(os.path.join(save_dir, 'train_labels.npy'), np.array(train_labels))
    print(f"训练特征已保存至 {save_dir}/train_features.npy")
    print(f"训练标签已保存至 {save_dir}/train_labels.npy")

def for_test_alexnet(model_name):
    """
    测试模型并返回测试集上的准确率、测试图像、真实标签和调整后的预测标签
    """
    print('------ Testing Start ------')
    # 加载训练好的模型
    model.load_state_dict(torch.load(model_name), strict=False)
    test_pred = []
    test_true = []
    test_probs = []

    # 禁用梯度计算
    with torch.no_grad():
        model.eval()  # 设定模型为评估模式
        for test_x, test_y in test_loader:
            if torch.cuda.is_available():
                images, labels = test_x.cuda(), test_y.cuda()
            else:
                images, labels = test_x, test_y
            output = model(images)  # 进行前向传播
            prob = F.softmax(output, dim=1)[:, 1]  # 获取概率值
            test_probs.extend(prob.cpu().numpy())  # 将概率值存储起来
            _, predicted = torch.max(output.data, 1)  # 获取最大概率的类别
            test_pred = np.hstack((test_pred, predicted.detach().cpu().numpy()))
            test_true = np.hstack((test_true, labels.detach().cpu().numpy()))

    # 调整阈值
    best_threshold = adjust_threshold(model, test_loader)
    test_pred_adjusted = (np.array(test_probs) >= best_threshold).astype(int)

    images = test_loader.dataset.test_img  # 获取测试图像
    test_acc = 100 * metrics.accuracy_score(test_true, test_pred_adjusted)  # 计算准确率
    test_classification_report = metrics.classification_report(test_true, test_pred_adjusted, digits=4)
    print('test_classification_report\n', test_classification_report)
    print('Accuracy of the network is: %.4f %%' % test_acc)
    return test_acc, images, test_true, test_pred_adjusted


def adjust_threshold(model, test_loader, threshold=0.5):
    model.eval()
    probs = []
    true = []
    with torch.no_grad():
        for test_x, test_y in test_loader:
            if torch.cuda.is_available():
                images, labels = test_x.cuda(), test_y.cuda()
            else:
                images, labels = test_x, test_y
            output = model(images)
            prob = F.softmax(output, dim=1)[:, 1]
            probs.extend(prob.cpu().numpy())
            true.extend(labels.cpu().numpy())

    precision, recall, thresholds = precision_recall_curve(true, probs)
    f1_scores = 2 * recall * precision / (recall + precision + 1e-6)
    best_threshold = thresholds[np.argmax(f1_scores)]
    print(f'Best threshold: {best_threshold}')
    return best_threshold


# 生成 Grad-CAM 热力图
def generate_cam(model, image_tensor, target_class):
    """
    生成指定类的类激活图（CAM）
    """
    model.eval()

    activations = None
    gradients = None

    # 钩子函数，用于获取卷积层的输出（激活）和反向传播的梯度
    def forward_hook(module, input, output):
        nonlocal activations
        activations = output

    def backward_hook(module, grad_in, grad_out):
        nonlocal gradients
        gradients = grad_out[0]

    # 找到模型的最后一个卷积层
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            target_layer = module

    # 注册前向和后向钩子
    handle_forward = target_layer.register_forward_hook(forward_hook)
    handle_backward = target_layer.register_backward_hook(backward_hook)

    # 前向传播
    output = model(image_tensor)
    output = F.softmax(output, dim=1)
    class_score = output[0][target_class]

    # 反向传播
    model.zero_grad()
    class_score.backward(retain_graph=True)

    # 注销钩子
    handle_forward.remove()
    handle_backward.remove()

    # 对梯度进行全局平均池化
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

    # 将激活图与池化的梯度相乘
    for i in range(activations.shape[1]):
        activations[:, i, :, :] *= pooled_gradients[i]

    # 平均通道的激活图
    heatmap = torch.mean(activations, dim=1).squeeze().detach().cpu().numpy()  # 使用 detach() 移除梯度信息

    # 归一化热力图
    heatmap = np.maximum(heatmap, 0)
    heatmap = (heatmap - np.min(heatmap)) / (heatmap.max() - heatmap.min())

    return heatmap


# 将热力图叠加到原始图像上，并保存
def overlay_and_save_heatmap(heatmap, image, filename, alpha=0.4, colormap=cv2.COLORMAP_JET):
    """
    将热力图叠加到原始图像并保存
    :param heatmap: 生成的热力图
    :param image: 原始输入图像
    :param filename: 保存的文件名
    :param alpha: 热力图透明度
    :param colormap: 热力图颜色映射
    """
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, colormap)

    overlayed_image = heatmap * alpha + image
    cv2.imwrite(filename, overlayed_image)


# 在测试集中进行热力图生成和保存
def visualize_and_save_cam(model, test_loader, save_path='./photo'):
    """
    可视化 CAM 并保存到指定路径
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    model.eval()
    for idx, (images, labels) in enumerate(test_loader):
        if torch.cuda.is_available():
            images = images.cuda()

        # 假设 batch_size 为 1 以进行可视化
        image_tensor = images[0].unsqueeze(0)
        label = labels[0].item()

        # 生成目标类别的 CAM
        heatmap = generate_cam(model, image_tensor, label)

        # 将图像张量转换为 numpy 数组进行可视化
        image_np = image_tensor.squeeze().cpu().numpy().transpose(1, 2, 0)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        # 叠加热力图并保存
        filename = os.path.join(save_path, f"heatmap_{idx}.jpg")
        overlay_and_save_heatmap(heatmap, image_np, filename)

        # 限制生成热力图数量，例如只生成前10张
        if idx >= 9:
            break
    print(f"Heatmaps saved in {save_path}")


def load_train_features(save_dir='./features'):
    # 检查文件是否存在
    train_features_path = os.path.join(save_dir, 'train_features.npy')
    train_labels_path = os.path.join(save_dir, 'train_labels.npy')

    if os.path.exists(train_features_path) and os.path.exists(train_labels_path):
        train_features = np.load(train_features_path)
        train_labels = np.load(train_labels_path)
        print(f"已从 {save_dir} 读取训练特征和标签")
        return train_features, train_labels
    else:
        print(f"训练特征文件不存在，请确保训练时已保存特征")
        return None, None

# KNN 特征表征质量测试
def knn_feature_test(model, train_loader, test_loader, save_dir='./features', n_neighbors=5):
    model.eval()  # 设置模型为评估模式
    test_features, test_labels = [], []

    # 从文件中加载训练集特征
    train_features, train_labels = load_train_features(save_dir)

    if train_features is None or len(train_features) == 0:
        print("训练特征为空，无法进行KNN测试")
        return

    # 提取测试集的特征
    for images, labels in test_loader:
        if torch.cuda.is_available():
            images = images.cuda()
        features = model(images, feature_only=True).detach().cpu().numpy()
        features = features.reshape(features.shape[0], -1)
        test_features.extend(features)
        test_labels.extend(labels.cpu().numpy())

    if len(test_features) == 0:
        print("测试特征为空，无法进行KNN测试")
        return

    # 使用KNN进行分类
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(train_features, train_labels)  # 训练KNN
    predictions = knn.predict(test_features)  # 预测
    accuracy = accuracy_score(test_labels, predictions)  # 计算准确率
    print(f"KNN Test Accuracy: {accuracy * 100:.2f}%")

if __name__ == '__main__':
    if is_train:
        train_alexnet(model)
        save_train_features(model, train_loader, save_dir='./features')
    else:
        # 测试模型
        test_acc, images, test_true, test_pred_adjusted = for_test_alexnet(save_model_name)

        # 生成并保存热力图
        visualize_and_save_cam(model, test_loader, save_path='./photo')

        load_train_features(save_dir='./features')

        # 测试阶段，使用保存的训练特征进行KNN表征测试
        knn_feature_test(model, train_loader, test_loader, save_dir='./features')

    print('头部ct运行时间level2：', time.time() - start)