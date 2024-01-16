from torch.utils.data import Dataset
import torch
from PIL import Image
import pandas as pd
import os


class CustomImageDataset(Dataset):
    def __init__(self, image_list, transform=None, isColor=True):
        self.isColor = isColor
        self.image_list = image_list
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        folder, filename, class_index = self.image_list[idx]
        img_path = os.path.join(folder, filename)

        if self.isColor:
            image = Image.open(img_path)
        else: 
            image = Image.open(img_path).convert('L')

        if self.transform:
            image = self.transform(image)

        return image, class_index
    
def get_image_list(folders, index_files, isMerged):
    image_list = []
    for folder, index_file in zip(folders, index_files):
        df = pd.read_csv(index_file)
        for _, row in df.iterrows():
            if isMerged:
                filename = row['File_name'].strip("'")
                class_values = row[['class_0', 'class_1', 'class_2', 'class_3']]
                class_index = class_values.astype(int).argmax()
                if class_index in [1, 2]: 
                    class_index = 1
                elif class_index == 3:  
                    class_index = 2
                class_index = torch.tensor(class_index)
                image_list.append((folder, filename, class_index))
            else:
                filename = row['File_name'].strip("'")
                class_index = torch.tensor(row[['class_0', 'class_1', 'class_2', 'class_3']].astype(int).argmax())
                image_list.append((folder, filename, class_index))
    return image_list

