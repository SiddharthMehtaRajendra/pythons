import os
import torch
from torch.utils.data import Dataset, DataLoader
cuda = torch.cuda.is_available()
import numpy as np
import torchvision
from PIL import Image




class Config:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

config = Config(
    num_classes = 7,
    width = 48,
    height = 48,
    num_epochs = 40,
    feat_dim = 7,
    lr_cent = 0.5,
    closs_weight = 0.5
)

train_size= 589
val_size= 196
test_size = 196
batch_size = 1


def parse_data(datadir, label_map):
    img_list = []
    file_list = []
    
    for root, directories, filenames in os.walk(datadir):      
        for filename in filenames:
            file_list.append(filename)
            if filename.endswith('.png'):
                
                filei = os.path.join(root, filename)
                file_ids = filename.split('_')
                file_id = file_ids[0] + '_' + file_ids[1]
                if file_id in label_map:
                    img_list.append(filei)
    return img_list[:train_size], img_list[train_size:train_size+val_size], img_list[train_size+val_size: train_size+val_size+test_size]



def parse_emotion_data(datadir):
    em_map = {}
    file_list = []
    for root, directories, filenames in os.walk(datadir):
        for filename in filenames:
            file_list.append(filename)
            if filename.endswith('.txt'):
                   
                f = open(root +  "/" + filename, 'r')
                lines = []
                for line in f.readlines():
                    lines.append(line)
                value = lines[0]
                f.close()
                
                keys = filename.split('_')
                key = keys[0] + '_' + keys[1]
                em_map[key] = int(float(value.strip())) - 1
                
    return em_map



label_map = parse_emotion_data("Emotion")



class ImageDataset(Dataset):
    def __init__(self, file_list, label_map, train = False):
        self.file_list = file_list
        self.label_map = label_map
        self.train = train
        self.data_len = len(self.file_list)

    def __len__(self):
        if self.train:
            return self.data_len * 5
        else:
            return self.data_len

    def __getitem__(self, index):
        img = None
        img_pil = None
        img_h = config.width
        img_w = config.height
        if index < self.data_len:            
            img = Image.open(self.file_list[index])
            img_pil = torchvision.transforms.Resize((img_h,img_w))(img)
            img = torchvision.transforms.ToTensor()(img_pil)
        
        if img.shape[0] == 3:
            img = torchvision.transforms.Grayscale(num_output_channels=1)(img_pil)
            img = torchvision.transforms.ToTensor()(img)
        img = torchvision.transforms.Normalize(mean=[0.485], std=[0.229])(img)
        keys = self.file_list[index].split('/')[-1].split('.')[0].split('_')
        label = self.label_map[keys[0] + '_' + keys[1]]
        
        return img, label



def get_test_loader():
    _, _, test_img_list = parse_data("cohn-kanade-images", label_map)
    test_dataset = ImageDataset(test_img_list, label_map)
    return torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, 
                                               shuffle=False, num_workers=8, drop_last=True)




