import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import os


class DRDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Caminho para o arquivo csv com anotações.
            root_dir (string): Diretório com todas as imagens.
            transform (callable, optional): Transformação opcional a ser aplicada
                em uma amostra.
        """
        self.frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.frame.iloc[idx, 0])
        image = Image.open(img_name)
        label = self.frame.iloc[idx, 1]

        label = 0 if label == 0 else 1

        if self.transform:
            image = self.transform(image)

        return image, label

    # def __getitem__(self, idx):
    #     img_name = os.path.join(self.root_dir, self.frame.iloc[idx, 0])
    #     image = Image.open(img_name)
    #     label = self.frame.iloc[idx, 1]
    #
    #     if self.transform:
    #         image = self.transform(image)
    #
    #     return image, label



