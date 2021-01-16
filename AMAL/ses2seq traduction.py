from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch
from torch.nn.utils.rnn import pad_sequence
from typing import List
from tqdm import tqdm
import re
import unicodedata
import string
import sentencepiece as spm
import numpy as np

def normalize(s):
    return re.sub(' +',' ', "".join(c if c in string.ascii_letters else " "
         for c in unicodedata.normalize('NFD', s.lower().strip())
         if  c in string.ascii_letters+" "+string.punctuation)).strip()

class Vocabulary:
    """Permet de gérer un vocabulaire.

    En test, il est possible qu'un mot ne soit pas dans le
    vocabulaire : dans ce cas le token "__OOV__" est utilisé.
    Attention : il faut tenir compte de cela lors de l'apprentissage !

    Utilisation:

    - en train, utiliser v.get("blah", adding=True) pour que le mot soit ajouté
      automatiquement
    - en test, utiliser v["blah"] pour récupérer l'ID du mot (ou l'ID de OOV)
    """
    PAD = 0
    EOS = 1
    SOS = 2
    OOVID = 3

    def __init__(self, oov: bool):
        self.oov = oov
        self.id2word = ["PAD", "EOS", "SOS"]
        self.word2id = {"PAD": Vocabulary.PAD, "EOS": Vocabulary.EOS, "SOS": Vocabulary.SOS}
        if oov:
            self.word2id["__OOV__"] = Vocabulary.OOVID
            self.id2word.append("__OOV__")

    def __getitem__(self, word: str):
        if self.oov:
            return self.word2id.get(word, Vocabulary.OOVID)
        return self.word2id[word]

    def get(self, word: str, adding=True):
        try:
            return self.word2id[word]
        except KeyError:
            if adding:
                wordid = len(self.id2word)
                self.word2id[word] = wordid
                self.id2word.append(word)
                return wordid
            if self.oov:
                return Vocabulary.OOVID
            raise

    def __len__(self):
        return len(self.id2word)

    def getword(self, idx: int):
        if idx < len(self):
            return self.id2word[idx]
        return None

    def getwords(self, idx: List[int]):
        return [self.getword(i) for i in idx]

class TradDataset():
    def __init__(self,data,vocOrig,vocDest,adding=True,max_len=10):
        self.sentences =[]
        for s in tqdm(data.split("\n")):
            if len(s)<1:continue
            orig,dest=map(normalize,s.split("\t")[:2])
            if len(orig)>max_len: continue
            self.sentences.append((torch.tensor([vocOrig.get(o) for o in orig.split(" ")]+[Vocabulary.EOS]),torch.tensor([vocDest.get(o) for o in dest.split(" ")]+[Vocabulary.EOS])))
    
    def __len__(self):return len(self.sentences)
    
    def __getitem__(self,i): return self.sentences[i]

def collate(batch):
    src,trg = zip(*batch)
    src_len = torch.tensor([len(s) for s in src])
    trg_len = torch.tensor([len(t) for t in trg])
    return pad_sequence(src),src_len,pad_sequence(trg),trg_len

class TradDatasetBPE_new():
    def __init__(self,data,spp,adding=True,max_len=10):
        self.sentences =[]
        for s in tqdm(data.split("\n")):
            if len(s)<1:continue
            orig,dest=map(normalize,s.split("\t")[:2])
            if len(orig)>max_len: continue
            self.sentences.append((torch.tensor(spp.encode(orig, out_type=int)+[Vocabulary.EOS]),torch.tensor(spp.encode(dest, out_type=int)+[Vocabulary.EOS])))
    
    def __len__(self):return len(self.sentences)
    
    def __getitem__(self,i): return self.sentences[i]
    
class TradDatasetBPE_old():
    def __init__(self,data,spp,adding=True,max_len=10):
        self.sentences =[]
        for s in tqdm(data.split("\n")):
            if len(s)<1:continue
            orig,dest=map(normalize,s.split("\t")[:2])
            if len(orig)>max_len: continue
            self.sentences.append((torch.tensor(spp.EncodeAsIds(orig)+[Vocabulary.EOS]),torch.tensor(spp.EncodeAsIds(dest)+[Vocabulary.EOS])))
    
    def __len__(self):return len(self.sentences)
    
    def __getitem__(self,i): return self.sentences[i]   
    
def accuracy(yhat,y):
    predict = torch.argmax(yhat,dim=1)
    return torch.sum(predict==y).float()

def train(encoder,decoder,iterator,optimizer,criterion,device,training=False,constraint_probability=0.999,decay=0.999,clip=1):
    encoder.eval()
    decoder.eval()
    if training:
        encoder.train()
        decoder.train()
    epoch_loss = 0
    epoch_acc = 0
    epoch_non_padded = 0
    
    for x,x_len,y,y_len in iterator:
        # x and y : l x b
        x,y = x.to(device),y.to(device)
        h = encoder(x, h0)
        y = torch.cat((torch.tensor([Vocabulary.SOS]*batch_size).view(1,-1).to(device), y), dim=0)
        nb_good_prediction = 0
        nb_non_padded = 0
        loss = 0
        
        constraint_probability *= decay
        
        if not training or torch.randn(1).item() > constraint_probability: 
            for i in range(y.shape[0]-1):
                yhat,h = decoder(y[i].unsqueeze(0),h)
                loss += criterion(yhat.squeeze(0), y[i+1])

                mask = y[i+1]!=Vocabulary.PAD
                nb_good_prediction += accuracy(yhat.squeeze(0)[mask],y[i+1][mask]).item()
                nb_non_padded += torch.sum(mask).item()
            loss /= nb_non_padded
        else:                
            yhat,h = decoder(y[0].unsqueeze(0), h)
            for i in range(1, y.shape[0]):
                logit = yhat.argmax(dim=2)
                yhat,h = decoder(logit,h)
                loss += criterion(yhat.squeeze(0), y[i])
                
                logit = logit.squeeze(0)
                mask = logit!=Vocabulary.PAD
                nb_good_prediction += accuracy(yhat.squeeze(0)[mask],y[i][mask]).item()
                nb_non_padded += torch.sum(mask).item()        
            loss /= nb_non_padded
                            
        epoch_loss += loss.item()
        epoch_acc += nb_good_prediction     
        epoch_non_padded += nb_non_padded
                  
        if training:    
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(decoder.parameters()), clip)
            optimizer.step()
            
    epoch_non_padded = float(epoch_non_padded)    
    return epoch_loss,epoch_acc/epoch_non_padded

''' model class'''
class Encoder(nn.Module):
    def __init__(self,in_size,emb_size,latent_size):
        super().__init__()
        self.embedding = nn.Embedding(in_size,emb_size)
        self.rnn = nn.GRU(emb_size,latent_size)
        
    def forward(self,src,h):
        # src : l x b
        # h : 1 x b x latent size
        embedded = self.embedding(src)
        latent = self.rnn(embedded,h)[0]
        return latent
    
class Decoder(nn.Module):
    def __init__(self,emb_size,latent_size,out_size):
        super().__init__()
        self.embedding = nn.Embedding(out_size,emb_size)
        self.act = nn.ReLU() 
        self.rnn = nn.GRU(emb_size,latent_size)
        self.linear = nn.Linear(latent_size,out_size)
        
    def forward(self,latent,h):
        # latent : b
        # h : 1 x b x latent size
        embedded = self.act(self.embedding(latent))
        output,h = self.rnn(embedded)
        output = self.linear(output)
        return output,h
    
''' data '''
file = "fra-eng.txt"
with open(file) as f:
    lines = f.readlines()
lines = [lines[x] for x in torch.randperm(len(lines))]

''' dataset '''
idxTrain = int(0.8*len(lines))
max_len = 20
vocEng = Vocabulary(True)
vocFra = Vocabulary(True)
train_data = TradDataset("".join(lines[:idxTrain]),vocEng,vocFra,max_len=max_len)
test_data = TradDataset("".join(lines[idxTrain:]),vocEng,vocFra,max_len=max_len)

''' dataset with byte pair encoding '''
#newer version 
#spp = spm.SentencePieceProcessor(model_file='model.model') 
#train_data = TradDatasetBPE_new("".join(lines[:idxTrain]),spp,max_len=max_len)
#test_data = TradDatasetBPE_new("".join(lines[idxTrain:]),spp,max_len=max_len)

#older version
spp = spm.SentencePieceProcessor()
spp.Load("model.model")
train_data = TradDatasetBPE_old("".join(lines[:idxTrain]),spp,max_len=max_len)
test_data = TradDatasetBPE_old("".join(lines[idxTrain:]),spp,max_len=max_len)

''' iterator '''
batch_size = 128
train_iterator = DataLoader(train_data,shuffle=True, batch_size=batch_size, collate_fn=collate, drop_last=True)  
test_iterator = DataLoader(test_data,shuffle=False, batch_size=batch_size, collate_fn=collate, drop_last=True)  

''' hyperparameters '''
in_size = len(vocEng)
out_size = len(vocFra)
emb_size = 50
latent_size = 100
lr = 1e-2

''' model '''
device = 'cuda' if torch.cuda.is_available else 'cpu'
encoder = Encoder(in_size,emb_size,latent_size).to(device)
decoder = Decoder(emb_size,latent_size,out_size).to(device)
h0 = torch.zeros(1, batch_size, latent_size).to(device)

''' objective '''
optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()),lr=lr)
criterion = nn.CrossEntropyLoss(ignore_index = Vocabulary.PAD)
nb_epochs = 500

print("Training ...")
for epoch in range(nb_epochs):
    train_loss,train_acc = train(encoder,decoder,train_iterator,optimizer,criterion,device, training=True)
    test_loss,test_acc = train(encoder,decoder,test_iterator,_,criterion,device, training=False)
    print(f'Epoch {epoch+1:02}')
    print(f'\tTrain Loss : {train_loss:.3f}')
    print(f'\tTrain Accuracy : {train_acc:.3f}')    
    print(f'\tTest Accuracy : {test_acc:.3f}')
    
    
encoder.to("cpu")
decoder.to("cpu")    
with torch.no_grad():
    index = np.random.choice(len(test_iterator.dataset))
    eng,fra = test_iterator.dataset[index]
    h0 = torch.zeros(1,1,latent_size)
    ht = encoder(eng.view(-1,1), h0) 
    tmp = torch.cat((torch.tensor([Vocabulary.SOS]).view(1,1), fra.view(-1,1)))
    res = []
    
    for i in range(len(tmp)-1):
        yhat,ht = decoder(tmp[i].unsqueeze(0), ht)
        res.append(yhat.argmax().item())
    
    print(f"original sequence : {vocEng.getwords(eng)}")
    print(f"target : {vocFra.getwords(fra)}")
    print(f"prediction : {vocFra.getwords(res)}")   