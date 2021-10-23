from torchvision import datasets, transforms
import os
import torch
from PIL import Image
import matplotlib.pyplot as plt

data_dir = os.path.join(os.getcwd(), 'segmentation_results' + os.sep)
batch_size = 32
input_size = 384


# 获取数据，并对数据做预处理
# 该数据集已经被预处理成了可用ImageFolder处理的形式
def load_image(data_type, directory=None):
    if directory is None:
        directory = data_dir
    images = datasets.ImageFolder(os.path.join(directory, data_type),
                                  transforms.Compose([
                                      transforms.RandomResizedCrop(input_size),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor()
                                  ]))
    if data_type == "train":
        return torch.utils.data.DataLoader(images, batch_size=batch_size, shuffle=True)
    else:
        return torch.utils.data.DataLoader(images, batch_size=batch_size)


def load_predict_images(directory=None):
    if directory is None:
        directory = data_dir
    transform = transforms.Compose([
                                      transforms.RandomResizedCrop(input_size),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor()
                                  ])
    image_file_list = []
    images = []
    for file in os.listdir(directory):
        file_path = os.path.join(directory, file)
        raw_img = Image.open(file_path)
        image_file_list.append(file)
        images.append(transform(raw_img))
    return image_file_list, images


if __name__ == '__main__':
    img = Image.open("D:\\华中科技大学-研究生\\甲状腺结节诊断辅助系统\\ultrasound_data\\train\\0\\1475_2.bmp")
    print("原图大小：", img.size)
    data1 = transforms.RandomResizedCrop(input_size)(img)
    print("随机裁剪后的大小:", data1.size)
    data2 = transforms.RandomResizedCrop(input_size)(img)
    data3 = transforms.RandomResizedCrop(input_size)(img)
    print("随机resize的大小:", data3.size)
    plt.subplot(2, 2, 1), plt.imshow(img), plt.title("原图")
    plt.subplot(2, 2, 2), plt.imshow(data1), plt.title("转换后的图1")
    plt.subplot(2, 2, 3), plt.imshow(data2), plt.title("转换后的图2")
    plt.subplot(2, 2, 4), plt.imshow(data3), plt.title("转换后的图3")
    plt.show()
