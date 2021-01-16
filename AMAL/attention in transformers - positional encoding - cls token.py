import re
from pathlib import Path
import torch
import numpy as np
import torch.nn as nn
import math
from torch.utils.data import Dataset,DataLoader
from datamaestro import prepare_dataset
import logging
import torch.nn.functional as F
import torch.optim as optim

logging.basicConfig(level=logging.INFO)

class FolderText(Dataset):
    def __init__(self, classes, folder: Path, tokenizer, load=False):
        self.tokenizer = tokenizer
        self.files = []
        self.filelabels = []
        self.labels = {}
        for ix, key in enumerate(classes):
            self.labels[key] = ix
        
        for label in classes:
            for file in (folder / label).glob("*.txt"):
                self.files.append(file.read_text(encoding="utf-8") if load else file)
                self.filelabels.append(self.labels[label])

    def __len__(self):
        return len(self.filelabels)
    
    def __getitem__(self, ix):
        s = self.files[ix]
        return self.tokenizer(s if isinstance(s, str) else s.read_text(encoding="utf-8")), self.filelabels[ix]
    
def get_imdb_data(embedding_size=50):
    WORDS = re.compile(r"\S+")

    word2id, embeddings = prepare_dataset('edu.stanford.glove.6b.%d' % embedding_size).load()
    OOVID = len(word2id)
    word2id["__OOV__"] = OOVID
    embeddings = np.vstack((embeddings, np.zeros(embedding_size)))

    def tokenizer(t):
        return [word2id.get(x, OOVID) for x in re.findall(WORDS, t.lower())]

    logging.info("Loading embeddings")

    logging.info("Get the IMDB dataset")
    ds = prepare_dataset("edu.stanford.aclimdb")

    return word2id, embeddings, FolderText(ds.train.classes, ds.train.path, tokenizer, load=True), FolderText(ds.test.classes, ds.test.path, tokenizer, load=True)
    
def collate(batch):
    ''' pad the sequences of the batch '''
    X,Y,X_lengths,max_len = [],[],[],0
    for item in batch:
        Y.append(item[1])
        X.append(torch.LongTensor(item[0]))
        X_lengths.append(len(item[0]))
        max_len = max(max_len,len(item[0]))
    X = torch.nn.utils.rnn.pad_sequence(X,padding_value = word2id["__OOV__"]) # l x b
    Y = torch.LongTensor(Y) # b
    X_lengths = torch.tensor(X_lengths).to(dtype=torch.long)
    mask = torch.ones(X.shape[1],X.shape[0],X.shape[0]) # b x l x l
    for i in range(X.shape[1]):
        mask[i,X.shape[1]:,:] = 0
        mask[i,:,X.shape[1]:] = 0
    return X,Y,X_lengths,mask.bool()

class PositionalEncoding(nn.Module):
    "Position embeddings"

    def __init__(self, d_model: int, max_len: int = 5000):
        """Génère des embeddings de position

        Args:
            d_model (int): Dimension des embeddings à générer
            max_len (int, optional): Longueur maximale des textes.
                Attention, plus cette valeur est haute, moins bons seront les embeddings de position.
        """
        super().__init__()

        pe = torch.zeros(max_len, d_model, dtype=torch.float)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        pe.requires_grad = False
        self.register_buffer('pe', pe)

    def forward(self, x):
        """Ajoute les embeddings de position"""
        x = x + self.pe[:, :x.size(1)]
        return x
    
class SelfAttention(nn.Module):
    def __init__(self,in_size,k=64,device='cpu'):
        super().__init__()
        self.tokeys = nn.Linear(in_size,k)
        self.toqueries = nn.Linear(in_size,k)
        self.tovalues = nn.Linear(in_size,k)
        self.k = k
        self.device = device
        
    def forward(self,x,mask):
        # x : l x b
        keys = self.tokeys(x) # l x b x k
        queries = self.toqueries(x) # l x b x k
        values = self.tovalues(x) # l x b x k
        tmp = torch.bmm(queries.permute(1,0,2),keys.permute(1,2,0)) / math.sqrt(self.k) # b x l x l
        tmp[mask] = -float('inf')
        score = F.softmax(tmp,dim=1)  # b x l x l
        out = torch.bmm(score,values.permute(1,0,2)).permute(1,0,2)  # l x b x k
        return out
    
class MultipleAttention(nn.Module):
    def __init__(self,embeddings,emb_size,L,nb_classes,device='cpu'):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(torch.Tensor(embeddings),freeze=True)
        self.linear_cls = nn.Linear(1,emb_size)
        self.attentions = nn.ModuleList([SelfAttention(emb_size,L[0],device).to(device)] + [SelfAttention(L[i],L[i+1],device).to(device) for i in range(len(L)-1)])
        self.linear = nn.Linear(L[-1],nb_classes)
        self.PositionalEncoding = PositionalEncoding(d_model=emb_size)
        self.activation = nn.ReLU()
        self.batch_norm = nn.BatchNorm1d(emb_size)
        self.device = device
        
    def forward(self,x,x_lengths,mask):
        # x : l x b
        pretrained = self.embedding(x[1:,:]) # l-1 x b x e
        cls = torch.ones(size=(x.shape[1],1)).to(self.device) # b x 1
        not_pretrained = self.linear_cls(cls).unsqueeze(0) # 1 x b x e
        embedded = torch.cat([not_pretrained,pretrained],dim=0) # l x b x e
        entry = self.PositionalEncoding(self.batch_norm(embedded.permute(0,2,1)).permute(0,2,1)) # l x b x e
        out = self.activation(self.attentions[0](entry,mask) + embedded) # l x b x L[0]
        for i in range(1,len(self.attentions)-1):
            out = self.activation(self.attentions[i](out,mask) + out) # l x b x L[-1]
        out = out[0,:,:] # b x L[-1]
        out = self.linear(out) # b x num_classes
        return out 

def accuracy(yhat,y):
    predict = torch.argmax(yhat,dim=1)
    return torch.sum(predict==y).float()/len(y)

def train(model,iterator,optimizer,criterion,device,training=False):
    model.eval()
    if training:
        model.train()
    epoch_loss = 0
    epoch_acc = 0
    
    for x,y,x_lengths,mask in iterator:
        x,y,x_lengths,mask = x.to(device),y.to(device),x_lengths.to(device),mask.to(device)
        output = model(x,x_lengths,mask)
        loss = criterion(output,y)
       
        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        acc = accuracy(output,y)
        epoch_loss += loss.item()
        epoch_acc += acc.item()
    return epoch_loss/len(iterator),epoch_acc/len(iterator)

''' dataset '''
emb_size = 50
word2id, embeddings, train_data, test_data = get_imdb_data(emb_size)
# x : l x b 

''' iterator '''
batch_size = 8
train_iterator = DataLoader(train_data,shuffle=True, batch_size=batch_size, collate_fn=collate, drop_last=True)  
test_iterator = DataLoader(test_data,shuffle=False, batch_size=batch_size, collate_fn=collate, drop_last=True)  

''' model '''
nb_classes = 2
L = [64,64,64]
L = [emb_size,emb_size,emb_size]
device = "cuda" if torch.cuda.is_available else "cpu"
# device = 'cpu'
model = MultipleAttention(embeddings,emb_size,L,nb_classes,device=device).to(device)

''' hyperparameters '''
lr = 1e-4
optimizer = optim.Adam(model.parameters(),lr=lr)
criterion = nn.CrossEntropyLoss(ignore_index = word2id["__OOV__"])
nb_epochs = 5

print("Training ...")
for epoch in range(nb_epochs):
    train_loss,train_acc = train(model,train_iterator,optimizer,criterion,device,training=True)
    test_loss,test_acc = train(model,test_iterator,None,criterion,device,training=False)
    print(f'Epoch {epoch+1:02}')
    print(f'\tTrain Loss : {train_loss:.3f}')
    print(f'\tTrain Accuracy : {train_acc:.3f}')    
    print(f'\tTest Accuracy : {test_acc:.3f}')

''' heatmap of product scalar between positional encoding and itself '''
pe = model.PositionalEncoding.pe
matrix = torch.mm(pe.squeeze(0),pe.squeeze(0).permute(1,0)) 
import matplotlib.pyplot as plt
plt.figure()
plt.imshow(matrix.cpu().detach().numpy())
plt.show()