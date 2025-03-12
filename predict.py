import argparse

from PIL import Image
import torch
import torchvision.models as models
import torch.nn as nn

import dataset as dset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-w", "--weights", help="Path to weights for ResNet50 model")
    parser.add_argument("-i", "--image", help="Path to target image")

    args = parser.parse_args()
    weights_path = args.weights
    image_path = args.image

    model = models.resnet50()
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(dset.DatasetParams.classes))
    model.load_state_dict(torch.load(weights_path))
    model.eval()

    img = Image.open(image_path).convert("RGB")
    img_tensor = dset.DatasetParams.data_transforms["valid"](img)
    img_tensor = img_tensor.unsqueeze(0)

    with torch.no_grad():
        output = model(img_tensor)
        _, predicted = torch.max(output, 1)

    print("Predicted class:", dset.DatasetParams.classes[predicted[0]])
