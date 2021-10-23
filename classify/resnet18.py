# -*- coding: utf-8 -*-
"""
Created on Thu May 14 20:01:53 2020
调用resnet预训练模型进行图片分类
数据集采用hymenoptera_data
数据集下载地址：https://download.pytorch.org/tutorial/hymenoptera_data.zip
@author:
"""
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torchvision import models
import copy
import sys
from classify.util import save_valid_result, save_train_process
from classify.preprocess import load_image, load_predict_images

model_name = "resnet"
num_classes = 2
num_epochs = 150
lr = 1e-4
momentum = 0.9
is_fixed = False
use_pretrained = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def set_parameters_require_grad(model, is_fixed):
    # 默认parameter.requires_grad = True
    # 当采用固定预训练模型参数的方法进行训练时，将预训练模型的参数设置成不需要计算梯度
    if is_fixed:
        for parameter in model.parameters():
            parameter.requires_grad = False


def init_model(model_name, num_classes, is_fixed, use_pretrained):
    if model_name == 'resnet':
        # 调用resnet模型，resnet18表示18层的resnet模型，
        # pretrained=True表示需要加载预训练好的模型参数，pretrained=False表示不加载预训练好的模型参数
        # 调用预训练的resnet50模型
        model = models.resnet50(pretrained=use_pretrained)
        # 设置参数是否需要计算梯度
        # is_fixed=True表示模型参数不需要跟新（不需要计算梯度）
        # is_fixed=False表示模型参数需要finetune（需要计算梯度）
        set_parameters_require_grad(model, is_fixed)
        # 取出全连接层的输入特征维度
        in_features = model.fc.in_features

        # 重新定义resnet50模型的全连接层,使其满足新的分类任务
        # 此时模型的全连接层默认需要计算梯度
        model.fc = nn.Linear(in_features, num_classes)

    return model


# 获取需要更新的模型参数
def get_require_updated_params(model, is_fixed):
    if is_fixed:
        require_update_params = []
        for param in model.parameters():
            if param.requires_grad:
                require_update_params.append(param)
        return require_update_params
    else:
        return model.parameters()


# 测试、验证根据data_type区分，只是加载的数据集不一样
def test(model, data_type):
    test_loader = load_image(data_type)
    model.eval()
    loss_func = nn.CrossEntropyLoss()
    loss_val = 0.0
    corrects = 0.0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            outputs = model(images)
            loss = loss_func(outputs, labels)

        _, predicts = torch.max(outputs, 1)
        loss_val += loss.item() * images.size(0)
        corrects += torch.sum(predicts.view(-1) == labels.view(-1)).item()
    test_loss = loss_val / len(test_loader.dataset)
    test_acc = corrects / len(test_loader.dataset)
    print("corrects: {}".format(corrects))
    print("Test Loss: {}, Test Acc: {}".format(test_loss, test_acc))
    return test_loss, test_acc


# 训练
def train():
    train_loader = load_image("train")
    model = init_model(model_name, num_classes, is_fixed, use_pretrained)
    model = model.to(device)
    require_update_params = get_require_updated_params(model, is_fixed)
    # 将需要跟新的参数放入优化器中进行优化
    optimizer = torch.optim.SGD(require_update_params, lr=lr, momentum=momentum)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=10, verbose=False,
                                               threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
    # 交叉熵损失函数
    loss_func = nn.CrossEntropyLoss()
    # 初始化最好的验证准确率
    best_val_acc = 0.0
    # 初始化最好的模型参数，采用deepcopy为防止优化过程中修改到best_model_params
    best_model_params = copy.deepcopy(model.state_dict())
    valid_results = []
    train_accuracies = []
    train_losses = []
    valid_accuracies = []
    valid_losses = []
    for epoch in range(num_epochs):
        model.train()
        loss_val = 0.0
        corrects = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = loss_func(outputs, labels)

            # 找出输出的最大概率所在的为
            # 二分类中：如果第一个样本输出的最大值出现在第0为，则其预测值为0
            _, predicts = torch.max(outputs, 1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # loss.item()为一个batch的平均loss的值
            # images.size(0)为当前batch中有多少样本量
            # loss.item() * images.size(0)表示一个batch的总loss值
            loss_val += loss.item() * images.size(0)

            # view(-1)表示将tensor resize成一个维度为[batch_size]的tensor
            # 计算预测值与标签值相同的数量
            corrects += torch.sum(predicts.view(-1) == labels.view(-1)).item()
            print("loss_val:", loss_val)
            print("corrects:", corrects)
        # 计算每个epoch的平均loss
        train_loss = loss_val / len(train_loader.dataset)
        # 预测准确的数量除以总的样本量即为准确率
        train_acc = corrects / len(train_loader.dataset)
        print("epoch:", epoch, "\n")
        print("Train Loss: {}, Train Acc: {}".format(train_loss, train_acc))

        # 调用测试
        test_loss, test_acc = test(model, "val")
        # 根据测试准确率跟新最佳模型的参数
        if best_val_acc < test_acc:
            best_val_acc = test_acc
            best_model_params = copy.deepcopy(model.state_dict())
        valid_result = {'epoch': epoch, "train_loss": train_loss, "train_acc": train_acc,
                        'valid_loss': test_loss, 'valid_acc': test_acc}
        valid_results.append(valid_result)
        train_accuracies.append(train_acc)
        train_losses.append(train_loss)
        valid_accuracies.append(test_acc)
        valid_losses.append(test_loss)
        # 按需更新学习率
        scheduler.step(test_acc)
    # 保存训练中间过程
    save_valid_result(valid_results)
    save_train_process(train_accuracies, train_losses, valid_accuracies, valid_losses)
    # 将模型的最优参数载入模型
    model.load_state_dict(best_model_params)
    torch.save(model.state_dict(), "resnet.pt")


# 预测
def predict():
    image_file_list, images = load_predict_images()
    model.eval()
    images = images.to(device)
    with torch.no_grad():
        outputs = model(images)
    _, predicts = torch.max(outputs, 1)
    for index, name in enumerate(image_file_list):
        print("image name: %s, predict result: %d \n", name, predicts[index])


if __name__ == '__main__':
    if sys.argv[1] == "train":
        torch.cuda.empty_cache()
        train()
    if sys.argv[1] == "test":
        model = init_model(model_name, num_classes, is_fixed, use_pretrained)
        model = model.to(device)
        model.load_state_dict(torch.load("resnet.pt"))
        acc = test(model, "test")
        print("Best Test Acc: {}".format(acc))



