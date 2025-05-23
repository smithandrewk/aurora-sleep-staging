import pandas as pd
import os
import torch
import sqlite3
from sqlite3 import Error
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

def show_tables(conn):
    """ show all tables in the database """
    cur = conn.cursor()
    query = "SELECT name FROM sqlite_master WHERE type='table';"
    cur.execute(query)
    result = cur.fetchall()
    return [result[i][0] for i in range(len(result))]

def drop_temporary_tables(conn):
    """ drop temporary tables in the database """
    cur = conn.cursor()
    temporary_table_names = [table for table in show_tables(conn) if 'temporary' in table]
    for temporary_table_name in temporary_table_names:
        cur.execute(f"DROP TABLE IF EXISTS {temporary_table_name};")
    conn.commit()
    
def print_query(query,conn):
    cur = conn.cursor()
    cur.execute(query)
    result = cur.fetchall()
    for row in result:
        print(row)

def get_recording_start_stop(conn):
    """
    author: Andrew Smith
    date: 17 April 2025
    description: Get recording start and stop times from the zdb database 
    table internal_property. The table internal_property gets created the 
    first time someone opens a
    Ponemah experiment in NeuroScore. So, the recording start stop times are
    somewhere in the Ponemah files.

    future: get recording start stop directly from Ponemah files to try and
    avoid zdb
    """
    """ get recording start and stop times """
    cur = conn.cursor()
    query = "SELECT value FROM internal_property WHERE key='RecordingStart'"
    cur.execute(query)
    result = cur.fetchall()
    recording_start = int(result[0][0])
    query = "SELECT value FROM internal_property WHERE key='RecordingStop'"
    cur.execute(query)
    result = cur.fetchall()
    recording_stop = int(result[0][0])
    return recording_start, recording_stop

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
    
def score_recording(edf_path,model_path,eeg_ch_name,device,zdb_path):
    if not os.path.exists(model_path):
        raise FileExistsError(model_path)

    model = Gandalf()
    model.load_state_dict(torch.load(model_path,map_location='cpu',weights_only=True))

    raw = read_raw_edf(input_fname=edf_path,include=[eeg_ch_name])
    data = raw.get_data()
    eeg = torch.from_numpy(data[0]).float()
    eeg = eeg.view(-1,5000)

    model.eval()
    model.to(device)

    with torch.no_grad():
        dataloader = DataLoader(EEGDataset(eeg),batch_size=32)
        logits = torch.cat([model(Xi.to(device)).cpu() for Xi in tqdm(dataloader)])
        y_pred = logits.softmax(dim=1).argmax(axis=1)

    pd.DataFrame(y_pred,columns=['y_pred']).to_csv(f"{zdb_path.replace('.zdb','.y_pred')}",index=False)

recordings_dir = f'data'
recording_ids = sorted([file.replace('.zdb','') for file in os.listdir(recordings_dir) if file.endswith('.zdb')])
eeg_ch_name = "EEG 1"
device = 'cuda'
model_path = 'gandalf.pt'
channel_mapping = {"25-Feb-AAV-1": "EEG 2", 
                   "25-Feb-AAV-2": "EEG 1",
                   "25-Feb-AAV-3": "EEG 1",
                   "25-Feb-AAV-4": "EEG 2",
                   "25-Feb-AAV-5": "EEG 2",
                   "25-Mar-AAV-1": "EEG 2",
                   "25-Mar-AAV-2": "EEG 2",
                   "25-Mar-AAV-3": "EEG 1",
                   "25-Mar-AAV-4": "EEG 1",
                   "25-Mar-AAV-5": "EEG 2",
                   "25-Mar-AAV-6": "EEG 2",
                   "25-Mar-AAV-7": "EEG 2",
                   "25-Mar-AAV-8": "EEG 2"}
for recording_id in recording_ids:
    print(recording_id)
    eeg_ch_name = channel_mapping[recording_id.split('.')[1]]
    print(eeg_ch_name)
    zdb_path = f'{recordings_dir}/{recording_id}.zdb'
    edf_path = f'{recordings_dir}/{recording_id}.edf'
    score_recording(edf_path=edf_path,model_path=model_path,eeg_ch_name=eeg_ch_name,device=device,zdb_path=zdb_path)