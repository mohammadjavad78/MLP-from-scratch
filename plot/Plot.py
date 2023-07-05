import matplotlib.pyplot as plt
import pickle
import numpy as np
import random

#read all data
def unpack(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

data1=unpack('../datasets/data_batch_1')[b'data']
label1=unpack('../datasets/data_batch_1')[b'labels']

data2=unpack('../datasets/data_batch_2')[b'data']
label2=unpack('../datasets/data_batch_2')[b'labels']

data3=unpack('../datasets/data_batch_3')[b'data']
label3=unpack('../datasets/data_batch_3')[b'labels']

data4=unpack('../datasets/data_batch_4')[b'data']
label4=unpack('../datasets/data_batch_4')[b'labels']

data5=unpack('../datasets/data_batch_5')[b'data']
label5=unpack('../datasets/data_batch_5')[b'labels']

data6=unpack('../datasets/test_batch')[b'data']
label6=unpack('../datasets/test_batch')[b'labels']

#add all data in data1
data1  = np.append(data1,data2,axis=0)
data1  = np.append(data1,data3,axis=0)
data1  = np.append(data1,data4,axis=0)
data1  = np.append(data1,data5,axis=0)
data1  = np.append(data1,data6,axis=0)
label1 = np.append(label1,label2,axis=0)
label1 = np.append(label1,label3,axis=0)
label1 = np.append(label1,label4,axis=0)
label1 = np.append(label1,label5,axis=0)
label1 = np.append(label1,label6,axis=0)


fig,axes= plt.subplots(10,10,figsize=(10,10))

labels=['Airplane','Automobile','Bird','Cat','Deer','Dog','Frog','Horse','Ship','Truck']

#make random numbers untill readall needed random number
for k in range(10):
    for j in range(10):
        i=random.randint(0,60000)
        while(label1[i]!=j):
            i=random.randint(0,60000)
        plt.subplot(10, 10, k*10+label1[i]+1)
        dd=np.array(data1[i],dtype=int).reshape((3,32,32)).T
        #rotate images to show correctly
        dd=np.rot90(dd)
        dd=np.rot90(dd)
        dd=np.rot90(dd)
        dd=np.fliplr(dd)
        axes[9][j].set_xlabel(f'{labels[int(label1[i])]}')
        axes[k][j].set_xticks([])
        axes[k][j].set_yticks([])
        axes[k][0].set_ylabel(f'{k}')
        axes[k][j].imshow(dd,aspect='auto')


fig.supxlabel('Acctual Labels')
fig.supylabel('Examples')

plt.show()
fig.savefig('fig.png')





