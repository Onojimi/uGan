from os.path import join

from dataset import DatasetFromFolder


def get_training_set(root_dir, direction):
    train_dir = join(root_dir, "train")

    return DatasetFromFolder(train_dir, direction)


def get_val_set(root_dir, direction):
    val_dir = join(root_dir, "val")

    return DatasetFromFolder(val_dir, direction)
