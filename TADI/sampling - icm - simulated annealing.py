import math
import random
import numpy as np

def echan(im_bin,beta_reg):

    i=0;
    j=0;
    
    for i in range(im_bin.shape[0]):
        for j in range(im_bin.shape[1]):
            
            Ureg0 = 0
            Ureg1 = 0

            i1=(i-1)%im_bin.shape[0]
            i2=(i+1)%im_bin.shape[0]
            j1=(j-1)%im_bin.shape[1]
            j2=(j+1)%im_bin.shape[1]
            
            Ureg0 = sum([im_bin[i1,j],im_bin[i2,j],im_bin[i,j1],im_bin[i,j2]])*beta_reg

            p0=math.exp(-Ureg0);
            
            Ureg1 = 4*beta_reg - Ureg0

            p1=math.exp(-Ureg1);
            
            if (p0+p1!=0.):
                if(random.uniform(0,1)<p0/(p0+p1)):
                    im_bin[i,j] = 0 
                else : 
                    im_bin[i,j] = 1
            
    return im_bin

def iter_icm(im_bin,im_toclass,beta_reg,m0,m1):

    i=0;
    j=0;
    
    for i in range(im_bin.shape[0]):
        for j in range(im_bin.shape[1]):

            i1=(i-1)%im_bin.shape[0]
            i2=(i+1)%im_bin.shape[0]
            j1=(j-1)%im_bin.shape[1]
            j2=(j+1)%im_bin.shape[1]
            
            Ureg0 = sum([im_bin[i1,j],im_bin[i2,j],im_bin[i,j1],im_bin[i,j2]])*beta_reg;
            Uattdo0 = (im_toclass[i,j]-m0)**2;
            U0 = Ureg0 + Uattdo0;
            #p0=math.exp();

            Ureg1 = 4*beta_reg - Ureg0
            Uattdo1 = (im_toclass[i,j]-m1)**2;
            U1 = Ureg1 + Uattdo1;
            #p1=math.exp();
            
            if (U0<U1):
                im_bin[i,j] = 0
            else : 
                im_bin[i,j] = 1
                

            
    return im_bin

def iter_SA(im_bin,im_toclass,beta_reg,m0,m1,T0):

    i=0;
    j=0;
    
    T = T0
    it = 0
    while(T>1.5):
        for i in range(im_bin.shape[0]):
            for j in range(im_bin.shape[1]):
                
                Ureg0 = 0
                Ureg1 = 0
    
                i1=(i-1)%im_bin.shape[0]
                i2=(i+1)%im_bin.shape[0]
                j1=(j-1)%im_bin.shape[1]
                j2=(j+1)%im_bin.shape[1]
                
                Ureg0 = sum([im_bin[i1,j],im_bin[i2,j],im_bin[i,j1],im_bin[i,j2]])*beta_reg
                Uattdo0 = (im_toclass[i,j]-m0)**2;
                U0 = Ureg0 + Uattdo0;
                p0=np.exp(-U0/T);
                
                Ureg1 = 4*beta_reg - Ureg0
                Uattdo1 = (im_toclass[i,j]-m1)**2;
                U1 = Ureg1 + Uattdo1;
                p1=np.exp(-U1/T);
                
                if (p0+p1!=0.):
                    if(random.uniform(0,1)<p0/(p0+p1)):
                        im_bin[i,j] = 0 
                    else : 
                        im_bin[i,j] = 1
        T -= 1
        it += 1
        if(it%10==0):
            print(it)

    return im_bin
            
