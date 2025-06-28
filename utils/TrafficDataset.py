# Reference:
# https://www.kaggle.com/code/chibani410/gtsrb-99-test-accuracy-pytorch


from torch.utils.data import Dataset
import os
from PIL import Image

class TrafficSignDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.iloc[idx]['image_path']
        img_path = os.path.join(self.img_dir, img_name)
        label = int(self.df.iloc[idx]['ClassId'])

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, label