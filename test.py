import os
import random

import torch
import torchvision.models as models
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import cv2
from tqdm import tqdm
import colorama

import dataset as dset


class FeatureExtractor():
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        def gradient_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        def activation_hook(module, input, output):
            self.activations = output

        target_layer.register_forward_hook(activation_hook)
        target_layer.register_backward_hook(gradient_hook)

    def forward(self, input_tensor):
        return self.model(input_tensor)

    def get_gradients(self):
        return self.gradients

    def get_activations(self):
        return self.activations


def predict_image(model, target_layer, image_path):
    image = Image.open(image_path).convert("RGB")
    input_tensor = dset.DatasetParams.data_transforms["valid"](image)
    input_tensor = input_tensor.unsqueeze(0)

    feature_extractor = FeatureExtractor(model, target_layer)
    output = feature_extractor.forward(input_tensor)
    predicted_class = torch.argmax(output[0]).item()

    model.zero_grad()
    output[0, predicted_class].backward()

    gradients = feature_extractor.get_gradients()
    activations = feature_extractor.get_activations()

    weights = torch.mean(gradients,
                         dim=(2, 3),
                         keepdim=True)
    heatmap = torch.sum(weights * activations,
                        dim=1,
                        keepdim=True)
    heatmap = F.relu(heatmap)
    heatmap = F.interpolate(heatmap,
                            size=(input_tensor.shape[2], input_tensor.shape[3]),
                            mode="bilinear",
                            align_corners=False)
    heatmap = heatmap.squeeze().cpu().detach().numpy()
    heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))

    img = Image.open(image_path)
    img = img.resize((input_tensor.shape[2], input_tensor.shape[3]))
    img = np.asarray(img)

    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    
    overlayed_img = heatmap / 255 * 0.5 + img / 255 * 0.5
    overlayed_img = np.clip(overlayed_img, 0, 1)
    
    return overlayed_img, dset.DatasetParams.classes[predicted_class]


def get_images(path, n, images):
    for class_dir in os.listdir(path):
        pathes = os.listdir(f"{path}/{class_dir}")
        for image in random.sample(pathes, n) if len(pathes) > n else pathes:
            images.append((f"{path}/{class_dir}/{image}", image, class_dir))


if __name__ == "__main__":
    weights_path = "work/20250311_141016/resnet50_best.pth"

    model = models.resnet50()
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(dset.DatasetParams.classes))
    model.load_state_dict(torch.load(weights_path))
    model.eval()

    target_layer = model.layer4[-1].conv2
    
    N = 3
    images = []
    get_images(dset.DatasetParams.train_path, N, images)
    get_images(dset.DatasetParams.valid_path, N, images)

    correct = 0
    with tqdm(images, desc="Testing progress") as pbar:
        for image_path, image_name, image_class in pbar:
            res, pred = predict_image(model, target_layer, image_path)
            if pred == image_class:
                print(colorama.Fore.RESET + f"Image {image_name}:" + image_class + "|" + colorama.Fore.GREEN + pred + colorama.Fore.RESET)
                correct += 1
                plt.imshow(res)
                plt.axis("off")
                plt.imsave(f"test/1/correct/{image_name}", res)
            else:
                print(colorama.Fore.RESET + f"Image {image_name}:" + image_class + "|" + colorama.Fore.RED + pred + colorama.Fore.RESET)
                plt.imshow(res)
                plt.axis("off")
                plt.imsave(f"test/1/wrong/{image_name}", res)

    print(f"Total acc: {correct}/{len(images)} {correct / len(images)}")
