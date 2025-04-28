import torch
import os
import torch
from torch import nn
from torch.nn.functional import relu
from mne.io import read_raw_edf
from tqdm import tqdm
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class ResidualBlock(nn.Module):
    def __init__(self,in_feature_maps,out_feature_maps,n_features) -> None:
        super().__init__()
        self.c1 = nn.Conv1d(in_feature_maps,out_feature_maps,kernel_size=8,padding='same',bias=False)
        self.bn1 = nn.LayerNorm((out_feature_maps,n_features),elementwise_affine=False)

        self.c2 = nn.Conv1d(out_feature_maps,out_feature_maps,kernel_size=5,padding='same',bias=False)
        self.bn2 = nn.LayerNorm((out_feature_maps,n_features),elementwise_affine=False)

        self.c3 = nn.Conv1d(out_feature_maps,out_feature_maps,kernel_size=3,padding='same',bias=False)
        self.bn3 = nn.LayerNorm((out_feature_maps,n_features),elementwise_affine=False)

        self.c4 = nn.Conv1d(in_feature_maps,out_feature_maps,1,padding='same',bias=False)
        self.bn4 = nn.LayerNorm((out_feature_maps,n_features),elementwise_affine=False)

    def forward(self,x):
        identity = x
        x = self.c1(x)
        x = self.bn1(x)
        x = relu(x)

        x = self.c2(x)
        x = self.bn2(x)
        x = relu(x)

        x = self.c3(x)
        x = self.bn3(x)
        x = relu(x)

        identity = self.c4(identity)
        identity = self.bn4(identity)

        x = x+identity
        x = relu(x)
        
        return x
    
class Frodo(nn.Module):
    def __init__(self,n_features) -> None:
        super().__init__()
        self.n_features = n_features
        self.block1 = ResidualBlock(1,8,n_features)
        self.block2 = ResidualBlock(8,16,n_features)
        self.block3 = ResidualBlock(16,16,n_features)

        self.gap = nn.AvgPool1d(kernel_size=n_features)
        self.fc1 = nn.Linear(in_features=16,out_features=3)
    def forward(self,x,classification=True):
        x = x.view(-1,1,self.n_features)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.gap(x)
        if(classification):
            x = self.fc1(x.squeeze())
            return x
        else:
            return x.squeeze()
        
class Gandalf(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = Frodo(n_features=5000)
        self.lstm = nn.LSTM(16,32,bidirectional=True)
        self.fc1 = nn.Linear(64,3)
    def forward(self,x_2d,classification=True):
        x_2d = x_2d.view(-1,9,1,5000)
        x = []
        for t in range(x_2d.size(1)):
            xi = self.encoder(x_2d[:,t,:,:],classification=False)
            x.append(xi.unsqueeze(0))
        x = torch.cat(x)
        out,_ = self.lstm(x)
        if(classification):
            x = self.fc1(out[-1])
        else:
            x = out[-1]
        return x
    
class EEGDataset(torch.utils.data.Dataset):
    def __init__(self,X):
        self.len = len(X)
        self.X = torch.cat([torch.zeros(4,5000),X,torch.zeros(4,5000)])

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.X[idx:idx+9].flatten()
    
model_path = f'gandalf.pt'
eeg_ch_name = 'EEG 1'
device = 'mps'

if not os.path.exists(model_path):
    raise FileExistsError(model_path)

model = Gandalf()
model.load_state_dict(torch.load(model_path,map_location='cpu'))
criterion = torch.nn.CrossEntropyLoss()

edf_filenames = [file for file in os.listdir(f'data') if file.endswith('.edf')]
input_fname = edf_filenames[0]
raw = read_raw_edf(input_fname=f'data/{input_fname}')
data = raw.get_data(picks=eeg_ch_name)
eeg = torch.from_numpy(data[0]).float()
eeg = eeg.view(-1,5000)

sns.histplot(eeg.std(dim=1),label=eeg_ch_name)
plt.xlim([0,.0003])
plt.legend()
plt.title(input_fname)
plt.savefig('dist.jpg',bbox_inches='tight')

print(edf_filenames)
print(raw.ch_names)
print(len(raw.ch_names))
print(len(data))

model.eval()
model.to(device)

with torch.no_grad():
    dataloader = DataLoader(EEGDataset(eeg),batch_size=32)
    logits = torch.cat([model(Xi.to(device)).cpu() for Xi in tqdm(dataloader)])
    y_pred = logits.softmax(dim=1).argmax(axis=1)

pd.DataFrame(y_pred,columns=['y_pred']).to_csv(f'data/{input_fname.replace('.edf','')}.csv',index=False)