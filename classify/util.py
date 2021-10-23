import os
import shutil
import sys
import csv
import matplotlib.pyplot as plt


# 将validate results保存到csv文件中
def save_valid_result(valid_results):
    file_header = ["epoch", "train_loss", "train_acc", "valid_loss", "valid_acc"]
    with open("valid_results.csv", "w+", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=file_header)
        writer.writeheader()
        for valid_result in valid_results:
            writer.writerow(valid_result)


# 将训练结果和验证结果通过图像的形式存储
def save_train_process(train_acc, train_loss, valid_acc, valid_loss):
    x = range(0, len(train_loss))
    plt.subplot(2, 2, 1)
    plt.plot(x, train_acc, '--.r')
    plt.xlabel("train accuracy vs. epoch")
    plt.ylabel("train accuracy")
    plt.subplot(2, 2, 2)
    plt.plot(x, train_loss, '--.r')
    plt.xlabel("train loss vs. epoch")
    plt.ylabel("train loss")
    plt.subplot(2, 2, 3)
    plt.plot(x, valid_acc, '--.r')
    plt.xlabel("valid loss vs. epoch")
    plt.ylabel("valid loss")
    plt.subplot(2, 2, 4)
    plt.plot(x, valid_loss, '--.r')
    plt.xlabel("valid loss vs. epoch")
    plt.ylabel("valid loss")
    plt.savefig("train_process.jpg")
    plt.show()


# 将dir目录下的图像按照6:2:2的比例分割训练集、验证集、测试集
def split_image_data(dir_path, save_dir_path):
    if os.path.exists(dir_path):
        for f1 in os.listdir(dir_path):
            second_dir_path = os.path.join(dir_path, f1)
            file_list = os.listdir(second_dir_path)
            train_save_dir = os.path.join(save_dir_path, "train", f1)
            if not os.path.exists(train_save_dir):
                os.makedirs(train_save_dir)
            val_save_dir = os.path.join(save_dir_path, "val", f1)
            if not os.path.exists(val_save_dir):
                os.makedirs(val_save_dir)
            test_save_dir = os.path.join(save_dir_path, "test", f1)
            if not os.path.exists(test_save_dir):
                os.makedirs(test_save_dir)
            for index, file in enumerate(file_list):
                file_path = os.path.join(second_dir_path, file)
                if index <= len(file_list) * 0.6:
                    save_path = os.path.join(train_save_dir, file)
                elif len(file_list) * 0.6 < index <= len(file_list) * 0.8:
                    save_path = os.path.join(val_save_dir, file)
                else:
                    save_path = os.path.join(test_save_dir, file)
                shutil.move(file_path, save_path)


if __name__ == '__main__':
    split_image_data(sys.argv[1], sys.argv[2])
