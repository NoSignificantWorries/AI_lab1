import os

import matplotlib.pyplot as plt

import dataset as dset

def save_hist(save_path, train_dataset, valid_dataset):
    count_of_train = [0] * len(dset.DatasetParams.classes)
    count_of_valid = [0] * len(dset.DatasetParams.classes)

    x = list(range(len(dset.DatasetParams.classes)))
    for label in train_dataset.labels:
        count_of_train[label] += 1
    for label in valid_dataset.labels:
        count_of_valid[label] += 1
    
    fig, axis = plt.subplots(figsize=(12, 6))
    width = 0.7
    axis.bar(x, count_of_train, width, label='Train', color='skyblue')
    axis.bar(x, count_of_valid, width, bottom=count_of_train, label='Valid', color='lightcoral')
    
    axis.set_xlabel('Classes')
    axis.set_ylabel('Number of Images')
    axis.set_title('Distribution of Classes in Train and Validation Sets')
    axis.set_xticks(x)
    axis.set_xticklabels(dset.DatasetParams.classes, rotation=45, ha="right")
    axis.legend()

    fig.tight_layout()

    plt.savefig(f"{save_path}/dataset.png", format="png", dpi=600)


if __name__ == "__main__":
    train = dset.CustomDataset(dset.DatasetParams.train_path,
                               dset.DatasetParams.classes,
                               dset.DatasetParams.data_transforms["train"])
    valid = dset.CustomDataset(dset.DatasetParams.valid_path,
                               dset.DatasetParams.classes,
                               dset.DatasetParams.data_transforms["valid"])
    
    print(f"Total classes: {len(dset.DatasetParams.classes)}")
    print(f"Total images: {len(train) + len(valid)}")
    print(f"Total train images: {len(train)}")
    print(f"Total valid images: {len(valid)}")
    save_hist("dataset/analysis", train, valid)
    # print(dset.get_mean_std())
