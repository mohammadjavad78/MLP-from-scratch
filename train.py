import numpy as np
import pickle
import os
import sys


from utlis import Layer
from utlis import Read_yaml
from plot import Plotinp

 



#Reading dataset
def unpack(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
#Reading yml
yaml=Read_yaml.Getyaml()
entries = os.listdir(yaml['dataset_path'])
data=[]
label=[]

for i in entries:
    data.append(unpack(f'{yaml["dataset_path"]}{i}')[b'data'])
    label.append(unpack(f'{yaml["dataset_path"]}{i}')[b'labels'])
#add all data in a list
for i in range(len(data)-1):
    data[0]  = np.append(data[0],data[i+1],axis=0)
    label[0] = np.append(label[0],label[i+1],axis=0)



#change format of y
def onehot(nn,k,classify=True):
    oo=[]
    for n in nn:
        onehotn=[0 for i in range(k)]
        if(classify):
            onehotn[int(n)]=1
        else:
            onehotn[0]=n
        oo.append(onehotn)
    return np.array(oo)


#split inputs in inp and xoutp
rangeinp=len(data[0])
inp=[[np.array(data[0][i].flatten()),label[0][i]] for i in range(rangeinp)]
inp.sort(key=lambda x: x[1])
inp1=[inp[0:6000],inp[6000:12000],inp[12000:18000],inp[18000:24000],inp[24000:30000],inp[30000:36000],inp[36000:42000],inp[42000:48000],inp[48000:54000],inp[54000:60000]]
inp=[inp1[i][j] for j in range(6000) for i in range(len(inp1))]
xinp=np.array([inp[i][0] for i in range(len(inp))])
xoutp1=[inp[i][1] for i in range(len(inp))]
xoutp=onehot(xoutp1,yaml['Layer'][-1][1])


def one_one():
    #add Layers
    Layers=[]
    Layers.append(Layer.Layer(xinp=xinp,numoutp=yaml['Layer'][0][1],activation_func=yaml['Layer'][0][0],normalize=yaml['Normalize'],Loss_func=yaml['Loss_function'],mu=yaml['Layer'][0][2],sigma=yaml['Layer'][0][3],bias=yaml['Layer'][0][4],seed=42))

    for i in range(len(yaml['Layer'])-1):
        Layers.append(Layer.Layer(xinp=Layers[i],numoutp=yaml['Layer'][i+1][1],activation_func=yaml['Layer'][i+1][0],Loss_func=yaml['Loss_function'],mu=yaml['Layer'][i+1][2],sigma=yaml['Layer'][i+1][3],bias=yaml['Layer'][i+1][4],seed=42))

    Layers[0].forward(Layers[0].scaleinputs(xinp[0:2]))
    Layers[0].backward(0.001,Layers[0].scaleinputs(xinp[0:2]),xoutp[0:2])
    Layers[0].forward(Layers[0].scaleinputs(xinp[0:2]))


# one_one()

def one_two():
    Layers=[]
    Layers.append(Layer.Layer(xinp=xinp,numoutp=yaml['Layer'][0][1],activation_func=yaml['Layer'][0][0],normalize=yaml['Normalize'],Loss_func=yaml['Loss_function'],mu=yaml['Layer'][0][2],sigma=yaml['Layer'][0][3],bias=yaml['Layer'][0][4],seed=42))

    for i in range(len(yaml['Layer'])-1):
        Layers.append(Layer.Layer(xinp=Layers[i],numoutp=yaml['Layer'][i+1][1],activation_func=yaml['Layer'][i+1][0],Loss_func=yaml['Loss_function'],mu=yaml['Layer'][i+1][2],sigma=yaml['Layer'][i+1][3],bias=yaml['Layer'][i+1][4],seed=42))

    params=Layers[0].minibatch(xinp,xoutp,yaml['learning_rate'],bachsize=yaml['batch_size'],epochs=yaml['num_epochs'],momentom=yaml['momentom'],train=yaml['train_split'],val=yaml['val_split'],ll=Layers[1])
    Plotinp.plots([params[0],params[3],params[6]],None,"accuracy",["train_acc","test_acc","val_acc"],"epochs","percent")
    Plotinp.plots([params[1],params[4],params[7]],None,"error",["train_error","test_error","val_error"],"epochs","percent")
    Plotinp.plots([params[2],params[5],params[8]],None,"Loss",["train_loss","test_loss","val_loss"],"epochs"," ")


# one_two()


def one_three():
    Layers=[]
    Layers.append(Layer.Layer(xinp=xinp,numoutp=yaml['Layer'][0][1],activation_func='sigmoid',normalize=yaml['Normalize'],Loss_func='MSE',mu=yaml['Layer'][0][2],sigma=yaml['Layer'][0][3],bias=yaml['Layer'][0][4],seed=42))

    for i in range(len(yaml['Layer'])-1):
        Layers.append(Layer.Layer(xinp=Layers[i],numoutp=yaml['Layer'][i+1][1],activation_func='sigmoid',Loss_func='MSE',mu=yaml['Layer'][i+1][2],sigma=yaml['Layer'][i+1][3],bias=yaml['Layer'][i+1][4],seed=42))

    params=Layers[0].minibatch(xinp,xoutp,yaml['learning_rate'],bachsize=yaml['batch_size'],epochs=yaml['num_epochs'],momentom=yaml['momentom'],train=yaml['train_split'],val=yaml['val_split'])
    Plotinp.plots([params[0],params[3],params[6]],None,"accuracy",["train_acc","test_acc","val_acc"],"epochs","percent")
    Plotinp.plots([params[1],params[4],params[7]],None,"error",["train_error","test_error","val_error"],"epochs","percent")
    Plotinp.plots([params[2],params[5],params[8]],None,"Loss",["train_loss","test_loss","val_loss"],"epochs"," ")

# one_three()

def one_four():
    Layers=[]
    Layers.append(Layer.Layer(xinp=xinp,numoutp=yaml['Layer'][0][1],activation_func=yaml['Layer'][0][0],normalize=yaml['Normalize'],Loss_func=yaml['Loss_function'],mu=0,sigma=0,bias=0,seed=42))

    for i in range(len(yaml['Layer'])-1):
        Layers.append(Layer.Layer(xinp=Layers[i],numoutp=yaml['Layer'][i+1][1],activation_func=yaml['Layer'][i+1][0],Loss_func=yaml['Loss_function'],mu=0,sigma=0,bias=0,seed=42))

    params=Layers[0].minibatch(xinp,xoutp,yaml['learning_rate'],bachsize=yaml['batch_size'],epochs=yaml['num_epochs'],momentom=yaml['momentom'],train=yaml['train_split'],val=yaml['val_split'])
    Plotinp.plots([params[0],params[3],params[6]],None,"accuracy",["train_acc","test_acc","val_acc"],"epochs","percent")
    Plotinp.plots([params[1],params[4],params[7]],None,"error",["train_error","test_error","val_error"],"epochs","percent")
    Plotinp.plots([params[2],params[5],params[8]],None,"Loss",["train_loss","test_loss","val_loss"],"epochs"," ")


# one_four()

def one_five():
    Layers=[]
    Layers.append(Layer.Layer(xinp=xinp,numoutp=yaml['Layer'][0][1],activation_func=yaml['Layer'][0][0],normalize='Standardized',Loss_func=yaml['Loss_function'],mu=yaml['Layer'][0][2],sigma=yaml['Layer'][0][3],bias=yaml['Layer'][0][4],seed=42))
    # Layers.append(Layer.Layer(xinp=xinp,numoutp=yaml['Layer'][0][1],activation_func=yaml['Layer'][0][0],normalize='do_nothing',Loss_func=yaml['Loss_function'],mu=yaml['Layer'][0][2],sigma=yaml['Layer'][0][3],bias=yaml['Layer'][0][4],seed=42))

    for i in range(len(yaml['Layer'])-1):
        Layers.append(Layer.Layer(xinp=Layers[i],numoutp=yaml['Layer'][i+1][1],activation_func=yaml['Layer'][i+1][0],Loss_func=yaml['Loss_function'],mu=yaml['Layer'][i+1][2],sigma=yaml['Layer'][i+1][3],bias=yaml['Layer'][i+1][4],seed=42))

    params=Layers[0].minibatch(xinp,xoutp,yaml['learning_rate'],bachsize=yaml['batch_size'],epochs=yaml['num_epochs'],momentom=yaml['momentom'],train=yaml['train_split'],val=yaml['val_split'],ll=Layers[1])
    Plotinp.plots([params[0],params[3],params[6]],None,"accuracy",["train_acc","test_acc","val_acc"],"epochs","percent")
    Plotinp.plots([params[1],params[4],params[7]],None,"error",["train_error","test_error","val_error"],"epochs","percent")
    Plotinp.plots([params[2],params[5],params[8]],None,"Loss",["train_loss","test_loss","val_loss"],"epochs"," ")

# one_five()



def one_six():
    Layers=[]
    Layers.append(Layer.Layer(xinp=xinp,numoutp=yaml['Layer'][0][1],activation_func=yaml['Layer'][0][0],normalize=yaml['Normalize'],Loss_func=yaml['Loss_function'],mu=yaml['Layer'][0][2],sigma=yaml['Layer'][0][3],bias=yaml['Layer'][0][4],seed=42))

    for i in range(len(yaml['Layer'])-1):
        Layers.append(Layer.Layer(xinp=Layers[i],numoutp=yaml['Layer'][i+1][1],activation_func=yaml['Layer'][i+1][0],Loss_func=yaml['Loss_function'],mu=yaml['Layer'][i+1][2],sigma=yaml['Layer'][i+1][3],bias=yaml['Layer'][i+1][4],seed=42))

    params=Layers[0].minibatch(xinp,xoutp,yaml['learning_rate'],bachsize=-1,epochs=yaml['num_epochs'],momentom=yaml['momentom'],train=yaml['train_split'],val=yaml['val_split'])
    # params=Layers[0].minibatch(xinp,xoutp,yaml['learning_rate'],bachsize=1,epochs=yaml['num_epochs'],momentom=yaml['momentom'],train=yaml['train_split'],val=yaml['val_split'])
    # params=Layers[0].minibatch(xinp,xoutp,yaml['learning_rate'],bachsize=64,epochs=yaml['num_epochs'],momentom=yaml['momentom'],train=yaml['train_split'],val=yaml['val_split'])
    # params=Layers[0].minibatch(xinp,xoutp,yaml['learning_rate'],bachsize=128,epochs=yaml['num_epochs'],momentom=yaml['momentom'],train=yaml['train_split'],val=yaml['val_split'])
    # params=Layers[0].minibatch(xinp,xoutp,yaml['learning_rate'],bachsize=256,epochs=yaml['num_epochs'],momentom=yaml['momentom'],train=yaml['train_split'],val=yaml['val_split'])
    Plotinp.plots([params[0],params[3],params[6]],None,"accuracy",["train_acc","test_acc","val_acc"],"epochs","percent")
    Plotinp.plots([params[1],params[4],params[7]],None,"error",["train_error","test_error","val_error"],"epochs","percent")
    Plotinp.plots([params[2],params[5],params[8]],None,"Loss",["train_loss","test_loss","val_loss"],"epochs"," ")



# one_six()


def one_seven():
    Layers=[]
    Layers.append(Layer.Layer(xinp=xinp,numoutp=yaml['Layer'][0][1],activation_func=yaml['Layer'][0][0],normalize=yaml['Normalize'],Loss_func=yaml['Loss_function'],mu=yaml['Layer'][0][2],sigma=yaml['Layer'][0][3],bias=yaml['Layer'][0][4],seed=42))

    for i in range(len(yaml['Layer'])-1):
        Layers.append(Layer.Layer(xinp=Layers[i],numoutp=yaml['Layer'][i+1][1],activation_func=yaml['Layer'][i+1][0],Loss_func=yaml['Loss_function'],mu=yaml['Layer'][i+1][2],sigma=yaml['Layer'][i+1][3],bias=yaml['Layer'][i+1][4],seed=42))

    params=Layers[0].minibatch(xinp,xoutp,0.01,bachsize=yaml['batch_size'],epochs=yaml['num_epochs'],momentom=yaml['momentom'],train=yaml['train_split'],val=yaml['val_split'])
    # params=Layers[0].minibatch(xinp,xoutp,0.1,bachsize=yaml['batch_size'],epochs=yaml['num_epochs'],momentom=yaml['momentom'],train=yaml['train_split'],val=yaml['val_split'])
    # params=Layers[0].minibatch(xinp,xoutp,10,bachsize=yaml['batch_size'],epochs=yaml['num_epochs'],momentom=yaml['momentom'],train=yaml['train_split'],val=yaml['val_split'])
    # params=Layers[0].minibatch(xinp,xoutp,100,bachsize=yaml['batch_size'],epochs=yaml['num_epochs'],momentom=yaml['momentom'],train=yaml['train_split'],val=yaml['val_split'])
    Plotinp.plots([params[0],params[3],params[6]],None,"accuracy",["train_acc","test_acc","val_acc"],"epochs","percent")
    Plotinp.plots([params[1],params[4],params[7]],None,"error",["train_error","test_error","val_error"],"epochs","percent")
    Plotinp.plots([params[2],params[5],params[8]],None,"Loss",["train_loss","test_loss","val_loss"],"epochs"," ")

# one_seven()



def one_eight():
    Layers=[]
    # Layers.append(Layer.Layer(xinp=xinp,numoutp=yaml['Layer'][0][1],activation_func='tanh',normalize=yaml['Normalize'],Loss_func=yaml['Loss_function'],mu=yaml['Layer'][0][2],sigma=yaml['Layer'][0][3],bias=yaml['Layer'][0][4],seed=42))
    # Layers.append(Layer.Layer(xinp=xinp,numoutp=yaml['Layer'][0][1],activation_func='sigmoid',normalize=yaml['Normalize'],Loss_func=yaml['Loss_function'],mu=yaml['Layer'][0][2],sigma=yaml['Layer'][0][3],bias=yaml['Layer'][0][4],seed=42))
    # Layers.append(Layer.Layer(xinp=xinp,numoutp=yaml['Layer'][0][1],activation_func='ReLu',normalize=yaml['Normalize'],Loss_func=yaml['Loss_function'],mu=yaml['Layer'][0][2],sigma=yaml['Layer'][0][3],bias=yaml['Layer'][0][4],seed=42))
    Layers.append(Layer.Layer(xinp=xinp,numoutp=yaml['Layer'][0][1],activation_func='leakyReLu',normalize=yaml['Normalize'],Loss_func=yaml['Loss_function'],mu=yaml['Layer'][0][2],sigma=yaml['Layer'][0][3],bias=yaml['Layer'][0][4],seed=42))

    for i in range(len(yaml['Layer'])-1):
        Layers.append(Layer.Layer(xinp=Layers[i],numoutp=yaml['Layer'][i+1][1],activation_func='linear',Loss_func=yaml['Loss_function'],mu=yaml['Layer'][i+1][2],sigma=yaml['Layer'][i+1][3],bias=yaml['Layer'][i+1][4],seed=42))

    params=Layers[0].minibatch(xinp,xoutp,yaml['learning_rate'],bachsize=yaml['batch_size'],epochs=yaml['num_epochs'],momentom=yaml['momentom'],train=yaml['train_split'],val=yaml['val_split'])
    Plotinp.plots([params[0],params[3],params[6]],None,"accuracy",["train_acc","test_acc","val_acc"],"epochs","percent")
    Plotinp.plots([params[1],params[4],params[7]],None,"error",["train_error","test_error","val_error"],"epochs","percent")
    Plotinp.plots([params[2],params[5],params[8]],None,"Loss",["train_loss","test_loss","val_loss"],"epochs"," ")


# one_eight()



def one_nine():
    Layers=[]
    Layers.append(Layer.Layer(xinp=xinp,numoutp=yaml['Layer'][0][1],activation_func=yaml['Layer'][0][0],normalize=yaml['Normalize'],Loss_func=yaml['Loss_function'],mu=yaml['Layer'][0][2],sigma=yaml['Layer'][0][3],bias=yaml['Layer'][0][4],seed=42))

    for i in range(len(yaml['Layer'])-1):
        Layers.append(Layer.Layer(xinp=Layers[i],numoutp=yaml['Layer'][i+1][1],activation_func=yaml['Layer'][i+1][0],Loss_func=yaml['Loss_function'],mu=yaml['Layer'][i+1][2],sigma=yaml['Layer'][i+1][3],bias=yaml['Layer'][i+1][4],seed=42))

    params=Layers[0].minibatch(xinp,xoutp,yaml['learning_rate'],bachsize=yaml['batch_size'],epochs=yaml['num_epochs'],momentom=yaml['momentom'],train=yaml['train_split'],val=yaml['val_split'])
    Plotinp.plots([params[0],params[3],params[6]],None,"accuracy",["train_acc","test_acc","val_acc"],"epochs","percent")
    Plotinp.plots([params[1],params[4],params[7]],None,"error",["train_error","test_error","val_error"],"epochs","percent")
    Plotinp.plots([params[2],params[5],params[8]],None,"Loss",["train_loss","test_loss","val_loss"],"epochs"," ")


# one_nine()

def one_ten():
    Layers=[]
    Layers.append(Layer.Layer(xinp=xinp,numoutp=yaml['Layer'][0][1],activation_func=yaml['Layer'][0][0],normalize=yaml['Normalize'],Loss_func=yaml['Loss_function'],mu=yaml['Layer'][0][2],sigma=yaml['Layer'][0][3],bias=yaml['Layer'][0][4],seed=42))

    for i in range(len(yaml['Layer'])-1):
        Layers.append(Layer.Layer(xinp=Layers[i],numoutp=yaml['Layer'][i+1][1],activation_func=yaml['Layer'][i+1][0],Loss_func=yaml['Loss_function'],mu=yaml['Layer'][i+1][2],sigma=yaml['Layer'][i+1][3],bias=yaml['Layer'][i+1][4],seed=42))

    params=Layers[0].minibatch(xinp,xoutp,yaml['learning_rate'],bachsize=yaml['batch_size'],epochs=yaml['num_epochs'],momentom=0.1,train=yaml['train_split'],val=yaml['val_split'],ll=Layers[1])
    Plotinp.plots([params[0],params[3],params[6]],None,"accuracy",["train_acc","test_acc","val_acc"],"epochs","percent")
    Plotinp.plots([params[1],params[4],params[7]],None,"error",["train_error","test_error","val_error"],"epochs","percent")
    Plotinp.plots([params[2],params[5],params[8]],None,"Loss",["train_loss","test_loss","val_loss"],"epochs"," ")

# one_ten()

def one_eleven():
    Layers=[]
    Layers.append(Layer.Layer(xinp=xinp[:8000],numoutp=yaml['Layer'][0][1],activation_func=yaml['Layer'][0][0],normalize=yaml['Normalize'],Loss_func=yaml['Loss_function'],mu=yaml['Layer'][0][2],sigma=yaml['Layer'][0][3],bias=yaml['Layer'][0][4],seed=42))

    for i in range(len(yaml['Layer'])-1):
        Layers.append(Layer.Layer(xinp=Layers[i],numoutp=yaml['Layer'][i+1][1],activation_func=yaml['Layer'][i+1][0],Loss_func=yaml['Loss_function'],mu=yaml['Layer'][i+1][2],sigma=yaml['Layer'][i+1][3],bias=yaml['Layer'][i+1][4],seed=42))

    params=Layers[0].minibatch(xinp[:8000],xoutp[:8000],yaml['learning_rate'],bachsize=yaml['batch_size'],epochs=yaml['num_epochs'],momentom=yaml['momentom'],train=yaml['train_split'],val=yaml['val_split'],ll=Layers[1],earlystopping=True)
    Plotinp.plots([params[0],params[3],params[6]],None,"accuracy",["train_acc","test_acc","val_acc"],"epochs","percent")
    Plotinp.plots([params[1],params[4],params[7]],None,"error",["train_error","test_error","val_error"],"epochs","percent")
    Plotinp.plots([params[2],params[5],params[8]],None,"Loss",["train_loss","test_loss","val_loss"],"epochs"," ")

# one_eleven()

def one_twelve():
    Layers=[]
    Layers.append(Layer.Layer(xinp=xinp,numoutp=yaml['Layer'][0][1],activation_func=yaml['Layer'][0][0],normalize=yaml['Normalize'],Loss_func=yaml['Loss_function'],mu=yaml['Layer'][0][2],sigma=yaml['Layer'][0][3],bias=yaml['Layer'][0][4],seed=42))

    for i in range(len(yaml['Layer'])-1):
        Layers.append(Layer.Layer(xinp=Layers[i],numoutp=yaml['Layer'][i+1][1],activation_func=yaml['Layer'][i+1][0],Loss_func=yaml['Loss_function'],mu=yaml['Layer'][i+1][2],sigma=yaml['Layer'][i+1][3],bias=yaml['Layer'][i+1][4],seed=42))

    params=Layers[0].KFold(xinp,xoutp,yaml['learning_rate'],epochs=10,momentom=yaml['momentom'],train=yaml['train_split'],K=5)
    Plotinp.plots([params[0],params[3],params[6]],None,"accuracy",["train_acc","test_acc","val_acc"],"epochs","percent")
    Plotinp.plots([params[1],params[4],params[7]],None,"error",["train_error","test_error","val_error"],"epochs","percent")
    Plotinp.plots([params[2],params[5],params[8]],None,"Loss",["train_loss","test_loss","val_loss"],"epochs"," ")

# one_twelve()

def one_thirteen():
    Layers=[]
    Layers.append(Layer.Layer(xinp=xinp,numoutp=16,activation_func='LeakyReLu',normalize='Normalized',Loss_func=yaml['Loss_function'],mu=0,sigma=0.1,bias=0.1,seed=42))

    Layers.append(Layer.Layer(xinp=Layers[0],numoutp=32,activation_func='LeakyReLu',Loss_func=yaml['Loss_function'],mu=0,sigma=0.1,bias=0.1,seed=42))
    Layers.append(Layer.Layer(xinp=Layers[1],numoutp=10,activation_func='linear',Loss_func=yaml['Loss_function'],mu=0,sigma=0.05,bias=0,seed=42))

    params=Layers[0].minibatch(xinp,xoutp,0.01,bachsize=yaml['batch_size'],epochs=yaml['num_epochs'],momentom=yaml['momentom'],train=yaml['train_split'],val=yaml['val_split'],ll=Layers[1])
    Plotinp.plots([params[0],params[3],params[6]],None,"accuracy",["train_acc","test_acc","val_acc"],"epochs","percent")
    Plotinp.plots([params[1],params[4],params[7]],None,"error",["train_error","test_error","val_error"],"epochs","percent")
    Plotinp.plots([params[2],params[5],params[8]],None,"Loss",["train_loss","test_loss","val_loss"],"epochs"," ")


# one_thirteen()








def two_three():
    yaml=Read_yaml.Getyaml('config2.yml')
    entries = os.listdir(yaml['dataset_path'])

    #read csv file and input and output
    arr = np.loadtxt(f'{yaml["dataset_path"]}/{entries[0]}',delimiter=",", dtype=str)
    arr=arr[1:].T
    x=np.array(arr[:arr.shape[0]-1],dtype=float).T
    y=np.array(arr[-1],dtype=float)
    rangeinp=len(x)
    inp=[[np.array(x[i]),y[i]] for i in range(rangeinp)]
    xinp=np.array([inp[i][0] for i in range(len(inp))])
    xoutp1=[inp[i][1] for i in range(len(inp))]
    xoutp=[inp[i][1] for i in range(len(inp))]
    xoutp=onehot(xoutp1,1,False)
    ma=np.max(xoutp)
    mi=np.min(xoutp)
    xoutp=(xoutp-np.min(xoutp))/(np.max(xoutp)-np.min(xoutp))

    Layers=[]
    Layers.append(Layer.Layer(xinp=xinp,numoutp=yaml['Layer'][0][1],activation_func=yaml['Layer'][0][0],normalize=yaml['Normalize'],Loss_func=yaml['Loss_function'],mu=yaml['Layer'][0][2],sigma=yaml['Layer'][0][3],bias=yaml['Layer'][0][4],seed=42,allwith=False))

    for i in range(len(yaml['Layer'])-1):
        Layers.append(Layer.Layer(xinp=Layers[i],numoutp=yaml['Layer'][i+1][1],activation_func=yaml['Layer'][i+1][0],Loss_func=yaml['Loss_function'],mu=yaml['Layer'][i+1][2],sigma=yaml['Layer'][i+1][3],bias=yaml['Layer'][i+1][4],seed=42))

    params=Layers[0].minibatch(xinp,xoutp,yaml['learning_rate'],bachsize=yaml['batch_size'],epochs=yaml['num_epochs'],momentom=yaml['momentom'],train=yaml['train_split'],val=yaml['val_split'],ll=Layers[1],classify=False)#done

    # Plotinp.plots([params[0],params[3],params[6]],None,"accuracy",["train_acc","test_acc","val_acc"],"epochs","percent")
    # Plotinp.plots([params[1],params[4],params[7]],None,"error",["train_error","test_error","val_error"],"epochs","percent")
    Plotinp.plots([params[2],params[5],params[8]],None,"Loss",["train_loss","test_loss","val_loss"],"epochs"," ")


# two_three()

#add all functions in a func list to run in from argv
funcs=[[one_one,one_two,one_three,one_four,one_five,one_six,one_seven,one_eight,one_nine,one_ten,one_eleven,one_twelve,one_thirteen],[0,0,two_three]]


funcs[int(sys.argv[1])-1][int(sys.argv[2])-1]()
