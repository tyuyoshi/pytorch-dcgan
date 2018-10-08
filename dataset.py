import glob
from skimage import io
from torch.utils.data import Dataset

class Dataset_Converter(Dataset):

    def __init__(self, root_dir = './datasets/', transform = None):

        self.root_dir = root_dir
        self.transform = transform
        self.image_names = glob.glob(root_dir + '*.jpg')

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]

        image = io.imread(img_name)

        if self.transform:
            image = self.transform(image)

        return image
