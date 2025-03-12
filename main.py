import os
import time
import datetime
import json

import torch
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import colorama

import dataset as dset


def train_model(model, criterion, optimizer, num_epochs, save_period=-1):
    # starting time detection
    since = time.time()

    # saveing the best weights
    best_model_wts = model.state_dict()
    best_acc = 0.0
    
    save_directory = f"work/{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.mkdir(save_directory)
    
    # dict for matrics
    result = {
        "train": {
            "accuracy": [],
            "loss": []
        },
        "valid": {
            "accuracy": [],
            "loss": []
        }
    }

    # main training loop {{{
    for epoch in range(num_epochs):
        # phases loop {{{
        for phase in ["train", "valid"]:
            # choosing model phase
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            # data loop {{{
            if phase == "valid":
                print("\033[38;5;240m", end="")
            with tqdm(dataloaders[phase], desc=f"{phase} Epoch {epoch + 1}/{num_epochs}") as pbar:
                for inputs, labels in pbar:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # on/off gradient based on phase value for this code chunk
                    with torch.set_grad_enabled(phase == "train"):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # updating weights if phase value is "train"
                        if phase == "train":
                            loss.backward()
                            optimizer.step()

                    # collecting  loss
                    running_loss += loss.item() * inputs.size(0)
                    # collecting count of correct predictions
                    running_corrects += torch.sum(preds == labels.data)
                    
                    # updating tqdm progress bar with mean loss and accuracy
                    epoch_loss = running_loss / dataset_sizes[phase]
                    epoch_acc = running_corrects.double() / dataset_sizes[phase]
                    pbar.set_postfix({"loss": epoch_loss, "acc": epoch_acc.item()})
            print(colorama.Fore.RESET, end="")
            # }}} finish of data loop

            # printing and collecting epoch statistics {{{
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            
            result[phase]["accuracy"].append(epoch_acc.item())
            result[phase]["loss"].append(epoch_loss)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            # }}}

            # updating best model weights and accuracy if needed
            if phase == "valid" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()
        # }}} finish of phase loop

        # if save period > 0 and current epoch is valid saving model weights in work directory
        if save_period > 0:
            if (epoch + 1) % save_period == 0:
                torch.save(model.state_dict(), f"{save_directory}/resnet50_{epoch + 1}.pth")
    # }}} training loop finish
            

    # printing final training information
    time_elapsed = time.time() - since
    print("Training complete in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))
    print("Best val Acc: {:4f}".format(best_acc))

    # saving best and latest model weights
    torch.save(best_model_wts, f"{save_directory}/resnet50_best.pth")
    torch.save(model.state_dict(), f"{save_directory}/resnet50_last.pth")

    # saving training data in json file
    with open(f"{save_directory}/results.json", "w") as json_file:
        json.dump(result, json_file, indent=4)
    return result
        

if __name__ == "__main__":
    # Loading data {{{
    image_datasets = {
        "train": dset.CustomDataset(dset.DatasetParams.train_path,
                                    classes=dset.DatasetParams.classes,
                                    transform=dset.DatasetParams.data_transforms["train"]),
        "valid": dset.CustomDataset(dset.DatasetParams.valid_path,
                                    classes=dset.DatasetParams.classes,
                                    transform=dset.DatasetParams.data_transforms["valid"])
    }
    dataloaders = {
        "train": DataLoader(image_datasets["train"],
                            batch_size=dset.DatasetParams.batch_size,
                            shuffle=True,
                            num_workers=dset.DatasetParams.num_workers),
        "valid": DataLoader(image_datasets["valid"],
                            batch_size=dset.DatasetParams.batch_size,
                            shuffle=False,
                            num_workers=dset.DatasetParams.num_workers)
    }
    dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "valid"]}
    # }}}

    # Model preparing {{{
    model_ft = models.resnet50(pretrained=True)

    # Freezing pretrained layers
    for param in model_ft.parameters():
        param.requires_grad = False

    # Replacing final layer with our classes
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, len(dset.DatasetParams.classes))

    # Select device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Starting on ", device)
    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()
    print(model_ft.fc.parameters())
    optimizer_ft = optim.Adam(model_ft.fc.parameters(), lr=0.001)
    # }}}
    
    # Training model {{{
    train_results = train_model(model_ft, criterion, optimizer_ft, num_epochs=100, save_period=5)
    # }}}
