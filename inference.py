
import os
import glob
from segmentation.data_loader import RescaleT
from segmentation.data_loader import ToTensorLab
from segmentation.data_loader import SalObjDataset
from skimage import io, transform
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from model import U2NET
from model import U2NETP
import cv2
import numpy as np
from PIL import Image
from segmentation.u2net_test import normPRED, save_output, save_front_image
import classify.resnet18 as resnet


def segmentation():
    model_name = 'u2net'  # u2netp
    image_dir = os.path.join(os.getcwd(), 'segmentation', 'test_data', 'test_images')
    prediction_dir = os.path.join(os.getcwd(), 'segmentation_results', + os.sep)
    model_dir = os.path.join(os.getcwd(), 'segmentation', 'saved_models', model_name, model_name + '.pth')
    img_name_list = glob.glob(image_dir + os.sep + '*')
    print(img_name_list)

    # --------- 2. dataloader ---------
    # 1. dataloader
    test_salobj_dataset = SalObjDataset(img_name_list=img_name_list,
                                        lbl_name_list=[],
                                        transform=transforms.Compose([RescaleT(320),
                                                                      ToTensorLab(flag=0)])
                                        )
    test_salobj_dataloader = DataLoader(test_salobj_dataset,
                                        batch_size=1,
                                        shuffle=False,
                                        num_workers=1)

    # --------- 3. model define ---------
    if (model_name == 'u2net'):
        print("...load U2NET---173.6 MB")
        net = U2NET(3, 1)
    elif (model_name == 'u2netp'):
        print("...load U2NEP---4.7 MB")
        net = U2NETP(3, 1)

    if torch.cuda.is_available():
        net.load_state_dict(torch.load(model_dir))
        net.cuda()
    else:
        net.load_state_dict(torch.load(model_dir, map_location='cpu'))
    net.eval()

    # --------- 4. inference for each image ---------
    for i_test, data_test in enumerate(test_salobj_dataloader):

        print("inferencing:", img_name_list[i_test].split(os.sep)[-1])

        inputs_test = data_test['image']
        inputs_test = inputs_test.type(torch.FloatTensor)

        if torch.cuda.is_available():
            inputs_test = Variable(inputs_test.cuda())
        else:
            inputs_test = Variable(inputs_test)

        d1, d2, d3, d4, d5, d6, d7 = net(inputs_test)

        # normalization
        pred = d1[:, 0, :, :]
        pred = normPRED(pred)

        # save results to test_results folder
        if not os.path.exists(prediction_dir):
            os.makedirs(prediction_dir, exist_ok=True)
        save_output(img_name_list[i_test], pred, prediction_dir)
        save_front_image(img_name_list[i_test], prediction_dir)

        del d1, d2, d3, d4, d5, d6, d7


def classify():
    model = resnet.init_model(resnet.model_name, resnet.num_classes,
                              resnet.is_fixed, resnet.use_pretrained)
    model = model.to(resnet.device)
    model.load_state_dict(torch.load("resnet.pt"))
    acc = resnet.test(model, "test")


if __name__ == '__main__':
    segmentation()
    classify()