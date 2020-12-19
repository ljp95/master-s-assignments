import argparse
import os
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.utils.data
import time
import copy
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from PIL import Image
import numpy as np
from sklearn.svm import LinearSVC
from sklearn import svm

models.vgg.model_urls["vgg16"] = "http://webia.lip6.fr/~robert/cours/rdfia/vgg16-397923af.pth"
os.environ["TORCH_HOME"] = "/tmp/torch"
PRINT_INTERVAL = 50
CUDA = False

def get_dataset(batch_size, path):
    # Cette fonction permet de recopier 3 fois une image qui
    # ne serait que sur 1 channel (donc image niveau de gris)
    # pour la "transformer" en image RGB. Utilisez la avec
    # transform.Lambda
    def duplicateChannel(img):
        img = img.convert('L')
        np_img = np.array(img, dtype=np.uint8)
        np_img = np.dstack([np_img, np_img, np_img])
        img = Image.fromarray(np_img, 'RGB')
        return img

    train_dataset = datasets.ImageFolder(path+'/train',
        transform = transforms.Compose([ # TODO Pré-traitement à faire
            # Add duplicateChannel, check image values : 255 ?
            transforms.Resize((224,224), interpolation=2),
            transforms.ToTensor(),
            transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
        ]))
    val_dataset = datasets.ImageFolder(path+'/test',
        transform = transforms.Compose([ # TODO Pré-traitement à faire
            transforms.Resize((224,224), interpolation=2),
            transforms.ToTensor(),
            transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
        ]))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                        batch_size=batch_size, shuffle=False, pin_memory=CUDA, num_workers=2)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                        batch_size=batch_size, shuffle=False, pin_memory=CUDA, num_workers=2)

    return train_loader, val_loader

def extract_features(data, model):
    # TODO init features matrices
    for i, (input, target) in enumerate(data):
        if i % PRINT_INTERVAL == 0:
            print('Batch {0:03d}/{1:03d}'.format(i, len(data)))
        if CUDA:
            input = input.cuda()
        # TODO Feature extraction à faire
        if i == 0:
            X = model.forward(input).data # data to not keep track of history outputs
            y = target.data
        else:            
            X = torch.cat((X,model.forward(input).data),dim=0) 
            y = torch.cat((y,target.data),dim=0)
    X = (X/X.norm(p=2,dim=1).view(-1,1)) # L2 normalization
    return X,y

class VGG16relu7(nn.Module):
    def __init__(self,vgg16):
        super(VGG16relu7, self).__init__()
        # recopier toute la partie convolutionnelle
        self.features = nn.Sequential( *list(vgg16.features.children()))
        # garder une partie du classifieur, -2 pour s'arrêter à relu7
        self.classifier = nn.Sequential(*list(vgg16.classifier.children())[:-2])
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
'''manual
batch_size = 10
CUDA = True
path = '15SceneData'
'''
def main(params):
    print('Instanciation du modèle')
    vgg16 = models.vgg16(pretrained=True)
    # TODO À remplacer par un reseau tronché pour faire de la feature extraction
    model = VGG16relu7(vgg16) 
#    model = models.resnet18(pretrained=True)
#    model = models.resnext101_32x8d(pretrained=True)

    model.eval()
    if CUDA: # si on fait du GPU, passage en CUDA
        model = model.cuda()

    # On récupère les données
    print('Récupération des données')
    train, test = get_dataset(batch_size, path)
#    train, test = get_dataset(params.batch_size, params.path)

    
    # Extraction des features
    print('Feature extraction')
    debut = time.time()
    X_train, y_train = extract_features(train, model)
    X_test, y_test = extract_features(test, model)
    print("Temps extraction features : {}".format(time.time()-debut))

    # TODO Apprentissage et évaluation des SVM à faire
    if CUDA: #if GPU, get back to cpu as svm use numpy data
        X_train = X_train.to('cpu')
        X_test = X_test.to('cpu')
    
    for n in [100,250,500]:
        X_train_pca = copy.deepcopy(X_train)
        y_train_pca = copy.deepcopy(y_train)
        X_test_pca = copy.deepcopy(X_test)
        y_test_pca = copy.deepcopy(y_test)
        
        # PCA
        debut = time.time()
        pca = PCA(n_components=n)
        pca.fit(X_train_pca)
        X_train_pca = pca.transform(X_train_pca)
        X_test_pca = pca.transform(X_test_pca)
        print("PCA time : {}".format(time.time()-debut))
        
        # SVM
        print('Apprentissage des SVM')
        debut = time.time()
        clf = svm.SVC(C=1,gamma='scale')
#        clf = LinearSVC(C=1)
        clf.fit(X_train_pca,y_train_pca)
        print("Training time : {}".format(time.time()-debut))
        debut = time.time()
        accuracy = clf.score(X_test_pca,y_test_pca)
        accuracy_train = clf.score(X_train_pca,y_train_pca)
        print("Inference time : {}".format(time.time()-debut))
        print("Accuracy test : {}".format(accuracy))
        print("Accuracy train : {}\n".format(accuracy_train))
    
    # possible C
    C = [1,5,10,15]
    for c in C:
        print('Apprentissage des SVM')
        debut = time.time()
#        clf = svm.SVC(C=c,gamma='scale')
        clf = LinearSVC(C=c)
        clf.fit(X_train,y_train)
        print("Training time : {}".format(time.time()-debut))
        debut = time.time()
        accuracy = clf.score(X_test,y_test)
        accuracy_train = clf.score(X_train,y_train)
        print("Inference time : {}".format(time.time()-debut))
        print("Accuracy test : {}".format(accuracy))
        print("Accuracy train : {}\n".format(accuracy_train))



if __name__ == '__main__':

    # Paramètres en ligne de commande
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default='15SceneData', type=str, metavar='DIR', help='path to dataset')
    parser.add_argument('--batch-size', default=8, type=int, metavar='N', help='mini-batch size (default: 8)')
    parser.add_argument('--cuda', dest='cuda', action='store_true', help='activate GPU acceleration')

    args = parser.parse_args()
    if args.cuda:
        CUDA = True
        cudnn.benchmark = True

    main(args)

    input("done")
