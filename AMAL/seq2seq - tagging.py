import logging
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch
from datamaestro import prepare_dataset
from torch.nn.utils.rnn import pad_sequence
from typing import List

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
    OOVID = 1
    PAD = 0

    def __init__(self, oov: bool):
        self.oov =  oov
        self.id2word = [ "PAD"]
        self.word2id = { "PAD" : Vocabulary.PAD}
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

    def getword(self,idx: int):
        if idx < len(self):
            return self.id2word[idx]
        return None

    def getwords(self,idx: List[int]):
        return [self.getword(i) for i in idx]

class TaggingDataset():
    def __init__(self, data, words: Vocabulary, tags: Vocabulary, adding=True):
        self.sentences = []
        for s in data:
            self.sentences.append(([words.get(token["form"], adding) for token in s], [tags.get(token["upos"], adding) for token in s]))
    
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, ix):
        return self.sentences[ix]

def collate(batch):
    """ pad the sequences of the batch """
    X = pad_sequence([torch.LongTensor(b[0]) for b in batch])
    Y = pad_sequence([torch.LongTensor(b[1]) for b in batch])
    return X,Y

''' dataset '''
logging.basicConfig(level=logging.INFO)
ds = prepare_dataset('org.universaldependencies.french.gsd')
logging.info("Loading datasets...")
words = Vocabulary(True)
tags = Vocabulary(False)
train_data = TaggingDataset(ds.train, words, tags, adding=True)
val_data = TaggingDataset(ds.validation, words, tags, adding=True)
test_data = TaggingDataset(ds.test, words, tags, adding=False)
logging.info("Vocabulary size: %d", len(words))

''' training functions '''
def accuracy(yhat,y):
    predict = torch.argmax(yhat,dim=1)
    return torch.sum(predict==y).float()

def train(model,iterator,optimizer,criterion,device,training=False):
    model.eval()
    if training:
        model.train()
    epoch_loss = 0
    epoch_acc = 0
    epoch_non_padded = 0
    
    for x,y in iterator:
        # x and y : l x b
        x,y = x.to(device),y.to(device)
        
        mask = (y!=Vocabulary.PAD).detach()
        nb_non_padded = torch.sum(mask).item()
        epoch_non_padded += nb_non_padded
        latent = model.forward(x) #l x b x latent size 
        
        loss = 0
        nb_good_tagging = 0
        for i in range(x.shape[0]):
            yhat = model.decode(latent[i]) #b x tag size
            nb_good_tagging += accuracy(yhat[mask[i]],y[i][mask[i]]).item()
            loss += criterion(yhat,y[i])
            
        epoch_loss += loss.item()
        epoch_acc += nb_good_tagging 
               
        if training:
            loss /= nb_non_padded            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    epoch_non_padded = float(epoch_non_padded)    
    return epoch_loss/epoch_non_padded,epoch_acc/epoch_non_padded

''' model class'''
class Labelling(nn.Module):
    def __init__(self, words_size, tag_size, latent_size, emb_size):
        super(Labelling, self).__init__()
        self.embedding = nn.Embedding(words_size, emb_size)
        self.rnn = nn.LSTM(emb_size, latent_size)
        self.linear = nn.Linear(latent_size, tag_size)
        
    def forward(self, x):
        # x : l x b
        embedded = self.embedding(x) # l x b x e
        encoded = self.rnn(embedded)[0] # b x latent size
        return encoded
    
    def decode(self,encoded):
        return self.linear(encoded) # b x tag size

''' iterator '''
batch_size = 128
train_iterator = DataLoader(train_data, collate_fn=collate, batch_size=batch_size, shuffle=True, drop_last=True)
val_iterator = DataLoader(val_data, collate_fn=collate, batch_size=batch_size, shuffle=False, drop_last=True)
test_iterator = DataLoader(test_data, collate_fn=collate, batch_size=batch_size, shuffle=False, drop_last=True)

''' model hyperparameters '''
in_size = len(words)
out_size = len(tags)
latent_size = 64
emb_size = 1024

''' model '''
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = Labelling( in_size, out_size, latent_size, emb_size).to(device)

''' parameters '''
nb_epochs = 10
lr = 1e-3
criterion = nn.CrossEntropyLoss(reduction='sum', ignore_index=Vocabulary.PAD)
optimizer = optim.Adam(model.parameters(),lr=lr)

print("Training ...")
for epoch in range(nb_epochs):
    train_loss,train_acc = train(model,train_iterator,optimizer,criterion,device, training=True)
    test_loss,test_acc = train(model,test_iterator,_,criterion,device, training=False)
    print(f'Epoch {epoch+1:02}')
    print(f'\tTrain Loss : {train_loss:.3f}')
    print(f'\tTrain Accuracy : {train_acc:.3f}')    
    print(f'\tTest Accuracy : {test_acc:.3f}')
    
    
    