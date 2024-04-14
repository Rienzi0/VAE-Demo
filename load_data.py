from PIL import Image
from torch.utils.data import Dataset,DataLoader,TensorDataset,ConcatDataset
import os
import pydicom
import numpy as np
import torch
from matplotlib import pyplot as plt
import torchvision.transforms as transforms

class GetData(Dataset):

    def __init__(self, root_dir):

  
        self.root_dir = root_dir
        
        folder_dir = "CelebChild"
        self.folder_path = os.path.join(root_dir,folder_dir)
        self.file_name_list = os.listdir(self.folder_path)
        self.file_name_list.sort()
        self.transform = torch.nn.Sequential(
            transforms.Grayscale(),
            transforms.Resize(28),
            #transforms.ConvertImageDtype(torch.float),
            transforms.Normalize([0.5], [0.5])
            
        )
    
    def __getitem__(self, index) :
        
        file_name = self.file_name_list[index]
        file_path = os.path.join(self.folder_path,file_name)
        
        
        # read figure  --- t1 & t1 are dicom file, lable is png file
         
        
        
        image = Image.open(file_path)
        image = np.array(image)
        # image = image / 255
        
        
        np.set_printoptions(precision=4)
        # image = np.expand_dims(image, axis=2)
        image = torch.tensor(image,dtype=torch.float32).requires_grad_(True)        
        
       
        image = image.permute(2,0,1)
        image = self.transform(image)
        
        
        
        
        return image
    
    def __len__(self):
        return len(self.file_name_list)
        

        
if __name__ == "__main__":
    
    set = GetData(root_dir="")
    img = set[0]
    print(img.shape)
    
    
    