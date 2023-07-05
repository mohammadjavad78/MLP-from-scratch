import numpy as np


from losses import Loss_function
from utlis import activation_functions

#create class of Layer
class Layer:
    def __init__(self,xinp,numoutp=5,mu=0,sigma=1,bias=0,activation_func='tanh',normalize="Do Nothing",Loss_func="Cross_entropy",allwith=True,seed=None):
        self.last=True
        self.allwith=allwith
        self.previous=xinp
        self.next=None
        self.normalize=normalize
        self.num=1
        self.seed=seed


        if(type(xinp)==type(np.array([])) or type(xinp)==type([])):
            self.a=np.array(xinp)
        else:
            xinp.last=False
            xinp.next=self
            self.a=np.array(xinp.A)
            self.num=xinp.num+1

        if(Loss_func=="Cross_entropy"):
            self.Loss=Loss_function.Cross_entropy
            self.grad_Loss=Loss_function.grad_Cross_entropy
        else:
            self.Loss=Loss_function.MSE
            self.grad_Loss=Loss_function.grad_MSE

        if(activation_func=='sigmoid'):
            self.activation_func=activation_functions.sigmoid
            self.grad_activation_func=activation_functions.grad_sigmoid
            
        elif(activation_func=='softmax'):
            self.activation_func=activation_functions.softmax
            self.grad_activation_func=activation_functions.grad_softmax

        elif(activation_func=='tanh'):
            self.activation_func=activation_functions.tanh
            self.grad_activation_func=activation_functions.grad_tanh
        elif(activation_func=='ReLu'):
            self.activation_func=activation_functions.relu
            self.grad_activation_func=activation_functions.grad_relu
        elif(activation_func=='linear'):
            self.activation_func=activation_functions.linear
            self.grad_activation_func=activation_functions.grad_linear
        else:
            self.activation_func=activation_functions.leakyrelu
            self.grad_activation_func=activation_functions.grad_leakyrelu


        self.setminmax(self.a)

        np.random.seed(self.seed)  
        self.w=np.random.normal(mu, sigma,size=(self.a.shape[1],numoutp))
        self.b=np.zeros((1,numoutp))+bias
        self.z=np.dot(self.a,self.w)+self.b
        self.A=self.activation_func(self.z)
        self.lastdeltaw=np.zeros_like(self.w)
        self.lastdeltab=np.zeros_like(self.b)



#A=f(aw+b)
    def forward(self,x=None):
        if(type(x)!=type(None)):
            self.a=x
        self.z=np.dot(self.a,self.w)+self.b
        self.A=self.activation_func(self.z)
        output=self.A
        if(self.next!=None):
            output=self.next.forward(self.A)
        return output


#first need forward
    def backward(self,learning_rate,x,y,momento=0):
        self.forward(np.array(x))
        self.totalbackward(learning_rate,np.array(x),y,momento)


    def totalbackward(self,learning_rate,x,y,momento=0):
        if(self.last):
            self.dLtotal_dA=self.grad_Loss(self.A,y)
        else:
            self.dLtotal_dA=self.next.totalbackward(learning_rate,x,y,momento)


        self.dA_dz=self.grad_activation_func(self.z)
        self.dz_dw=self.a.T
        self.dz_da=self.w.T


        self.dLtotal_da=np.dot(np.multiply(self.dLtotal_dA,self.dA_dz),self.dz_da)

        self.dLtotal_dw=np.dot(self.dz_dw,np.multiply(self.dLtotal_dA,self.dA_dz))
        self.dLtotal_db=np.sum(np.multiply(self.dLtotal_dA,self.dA_dz),axis=0)
        

        self.w=self.w-learning_rate*self.dLtotal_dw-momento*self.lastdeltaw
        self.b=self.b-learning_rate*self.dLtotal_db-momento*self.lastdeltab


        self.lastdeltaw=learning_rate*self.dLtotal_dw+momento*self.lastdeltaw
        self.lastdeltab=learning_rate*self.dLtotal_db+momento*self.lastdeltab
        return self.dLtotal_da


    def minibatch(self,x,y,learning_rate=0.1,momentom=0,bachsize=1,epochs=50,train=0.8334,val=1,ll=None,earlystopping=False,classify=True):
        x_test=x[:int(len(x)*(1-train))]
        y_test=y[:int(len(x)*(1-train))]
        etcx=x[int(len(x)*(1-train)):]
        etcy=y[int(len(x)*(1-train)):]
        x_train=etcx[:int(len(etcx)*val)]
        y_train=etcy[:int(len(etcx)*val)]
        x_val=etcx[int(len(etcx)*val):]
        y_val=etcy[int(len(etcx)*val):]

        #shuffle x
        np.random.seed(self.seed)  
        rng_state = np.random.get_state()
        np.random.shuffle(x_test)
        np.random.set_state(rng_state)
        np.random.shuffle(y_test)
        np.random.seed(self.seed)  
        rng_state = np.random.get_state()
        np.random.shuffle(x_train)
        np.random.set_state(rng_state)
        np.random.shuffle(y_train)
        np.random.seed(self.seed)  
        rng_state = np.random.get_state()
        np.random.shuffle(x_val)
        np.random.set_state(rng_state)
        np.random.shuffle(y_val)



        if(bachsize==-1):
            bachsize=len(y_train)

        self.setminmax(x_train)
        x_train=self.scaleinputs(np.array(x_train))
        x_test=self.scaleinputs(np.array(x_test))
        x_val=self.scaleinputs(np.array(x_val))


        train_acc=[]
        train_error=[]
        train_loss=[]

        test_acc=[]
        test_error=[]
        test_loss=[]

        val_acc=[]
        val_error=[]
        val_loss=[]

        #do epochs
        for epoch in range(epochs):
            for j in range(x_train.shape[0]//(bachsize)):
                
                self.backward(learning_rate,x_train[j*bachsize:min((j+1)*bachsize,x_train.shape[0])],y_train[j*bachsize:min((j+1)*bachsize,x_train.shape[0])],momentom)
                o=self.forward(x_train[j*bachsize:min((j+1)*bachsize,x_train.shape[0])])


            o=self.forward(x_train)
            if(classify):
                out=o.argmax(axis=1)
                yout=y_train.argmax(axis=1)
                k=np.sum([out[i]!=yout[i] for i in range(len(yout))])/len(yout)
            else:
                out=o
                yout=y_train
                k=[abs(out[i]-yout[i]) for i in range(len(yout))]
            train_error.append(round(np.mean(k)*100,3))
            train_acc.append(round(100-train_error[-1],3))
            train_loss.append(round(np.mean(self.Loss(o,y_train)),3))

            print(f"accuracy:{train_acc[-1]}%,Loss:{train_loss[-1]},epochs:{round(epoch/epochs,3)}=>1")

            o=self.forward(x_val)
            if(classify):
                out=o.argmax(axis=1)
                yout=y_val.argmax(axis=1)
                k=np.sum([out[i]!=yout[i] for i in range(len(yout))])/len(yout)
            else:
                out=o
                yout=y_val
                k=[abs(out[i]-yout[i]) for i in range(len(yout))]
            if(len(y_val)>0):
                val_error.append(round(np.mean(k)*100,3))
            else:
                val_error.append(100)
            val_acc.append(round(100-val_error[-1],3))
            val_loss.append(round(np.mean(self.Loss(o,y_val)),3))
            

            o=self.forward(x_test)
            if(classify):
                out=o.argmax(axis=1)
                yout=y_test.argmax(axis=1)
                k=np.sum([out[i]!=yout[i] for i in range(len(yout))])/len(yout)
            else:
                out=o
                yout=y_test
                k=[abs(out[i]-yout[i]) for i in range(len(yout))]
            test_error.append(round(np.mean(k)*100,3))
            test_acc.append(round(100-test_error[-1],3))
            test_loss.append(round(np.mean(self.Loss(o,y_test)),3))
            

            if(len(val_acc)>4 and earlystopping):
                if(val_acc[-1]<val_acc[-2] and val_acc[-2]<val_acc[-3] and val_acc[-3]<val_acc[-4]):
                    return train_acc,train_error,train_loss,test_acc,test_error,test_loss,val_acc,val_error,val_loss

        return train_acc,train_error,train_loss,test_acc,test_error,test_loss,val_acc,val_error,val_loss
    

    def KFold(self,x,y,learning_rate=0.1,momentom=0,epochs=50,train=0.8334,K=5):

        train_acc=[]
        train_error=[]
        train_loss=[]

        val_acc=[]
        val_error=[]
        val_loss=[]

        test_acc=[]
        test_error=[]
        test_loss=[]

        train_acc1=[]
        train_error1=[]
        train_loss1=[]

        val_acc1=[]
        val_error1=[]
        val_loss1=[]

        test_acc1=[]
        test_error1=[]
        test_loss1=[]

        x_test=x[:int(len(x)*(1-train))]
        y_test=y[:int(len(x)*(1-train))]

        for epoch in range(epochs):
            etcx=x[int(len(x)*(1-train)):]
            etcy=y[int(len(x)*(1-train)):]

            np.random.seed(self.seed+epoch)  
            rng_state = np.random.get_state()
            np.random.shuffle(etcx)
            np.random.set_state(rng_state)
            np.random.shuffle(etcy)
            


        



            x_train1=etcx[:int(len(etcx)*0.2)]
            y_train1=etcy[:int(len(etcx)*0.2)]
            x_train2=etcx[int(len(etcx)*0.2):int(len(etcx)*0.4)]
            y_train2=etcy[int(len(etcx)*0.2):int(len(etcx)*0.4)]
            x_train3=etcx[int(len(etcx)*0.4):int(len(etcx)*0.6)]
            y_train3=etcy[int(len(etcx)*0.4):int(len(etcx)*0.6)]
            x_train4=etcx[int(len(etcx)*0.6):int(len(etcx)*0.8)]
            y_train4=etcy[int(len(etcx)*0.6):int(len(etcx)*0.8)]
            x_train5=etcx[int(len(etcx)*0.8):]
            y_train5=etcy[int(len(etcx)*0.8):]

            
            np.random.seed(self.seed)  
            rng_state = np.random.get_state()
            np.random.shuffle(x_train1)
            np.random.set_state(rng_state)
            np.random.shuffle(y_train1)
            np.random.seed(self.seed)  
            rng_state = np.random.get_state()
            np.random.shuffle(x_train2)
            np.random.set_state(rng_state)
            np.random.shuffle(y_train2)
            np.random.seed(self.seed)  
            rng_state = np.random.get_state()
            np.random.shuffle(x_train3)
            np.random.set_state(rng_state)
            np.random.shuffle(y_train3)
            np.random.seed(self.seed)  
            rng_state = np.random.get_state()
            np.random.shuffle(x_train4)
            np.random.set_state(rng_state)
            np.random.shuffle(y_train4)
            np.random.seed(self.seed)  
            rng_state = np.random.get_state()
            np.random.shuffle(x_train5)
            np.random.set_state(rng_state)
            np.random.shuffle(y_train5)

            x_trains=[x_train2+x_train3+x_train4+x_train5,x_train1+x_train3+x_train4+x_train5,x_train1+x_train2+x_train4+x_train5,x_train1+x_train2+x_train3+x_train5,x_train1+x_train2+x_train3+x_train4]
            y_trains=[y_train2+y_train3+y_train4+y_train5,y_train1+y_train3+y_train4+y_train5,y_train1+y_train2+y_train4+y_train5,y_train1+y_train2+y_train3+y_train5,y_train1+y_train2+y_train3+y_train4]
            x_vals=[x_train1,x_train2,x_train3,x_train4,x_train5]
            y_vals=[y_train1,y_train2,y_train3,y_train4,y_train5]

            #K split
            for i in range(K):
            
            
                x_train=x_trains[i]
                y_train=y_trains[i]
                x_val=x_vals[i]
                y_val=y_vals[i]

                self.setminmax(x_train)
                x_train=self.scaleinputs(np.array(x_train))
                x_test=self.scaleinputs(np.array(x_test))
                x_val=self.scaleinputs(np.array(x_val))

                o=self.backward(learning_rate,x_train,y_train,momentom)
                o=self.forward(x_train)
                out=o.argmax(axis=1)
                yout=y_train.argmax(axis=1)
                train_acc1.append(round(sum([out[i]==yout[i] for i in range(len(y_train))])/len(y_train)*100,3))
                train_error1.append(100-train_acc1[-1])
                train_loss1.append(round(np.mean(self.Loss(o,y_train)),3))
                print(f"accuracy:{round(sum([out[i]==yout[i] for i in range(len(y_train))])/len(y_train)*100,3)}%,Loss:{round(np.sum(self.Loss(o,y_train)),3)},epochs:{epoch/epochs}=>0")

                o=self.forward(x_val)
                out=o.argmax(axis=1)
                yout=y_val.argmax(axis=1)
                val_acc1.append(round(sum([out[i]==yout[i] for i in range(len(y_val))])/len(y_val)*100,3))
                val_error1.append(100-val_acc1[-1])
                val_loss1.append(round(np.mean(self.Loss(o,y_val)),3))

                
                o=self.forward(x_test)
                out=o.argmax(axis=1)
                yout=y_test.argmax(axis=1)
                test_acc1.append(round(sum([out[i]==yout[i] for i in range(len(y_test))])/len(y_test)*100,3))
                test_error1.append(100-test_acc1[-1])
                test_loss1.append(round(np.mean(self.Loss(o,y_test)),3))


                
                if(i==K-1):
                    train_acc.append(np.mean(train_acc1))
                    train_error.append(np.mean(train_error1))
                    train_loss.append(np.mean(train_loss1))
                    val_acc.append(np.mean(val_acc1))
                    val_error.append(np.mean(val_error1))
                    val_loss.append(np.mean(val_loss1))
                    test_acc.append(np.mean(test_acc1))
                    test_error.append(np.mean(test_error1))
                    test_loss.append(np.mean(test_loss1))




                    train_acc1=[]
                    train_error1=[]
                    train_loss1=[]
                    val_acc1=[]
                    val_error1=[]
                    val_loss1=[]
                    test_acc1=[]
                    test_error1=[]
                    test_loss1=[]

        return train_acc,train_error,train_loss,test_acc,test_error,test_loss,val_acc,val_error,val_loss
    




    #split inputs
    def scaleinputs(self,x=None):
        if(self.allwith):
            if(self.normalize=="Normalized"):
                x=(x-self.min)/(self.max-self.min)
            elif(self.normalize=="Standardized"):
                x=(x-self.mean)/(self.std)
            return x
        else:
            if(self.normalize=="Normalized"):
                for i in range(x.shape[1]):
                    x.T[i]=(x.T[i]-self.min[i])/(self.max[i]-self.min[i])
            elif(self.normalize=="Standardized"):
                for i in range(x.shape[1]):
                    x.T[i]=(x.T[i]-self.mean[i])/self.std[i]
            return x
    
    #get min or max or mean or std of input
    def setminmax(self,x):
        if(self.allwith):
            self.min=np.min(x)
            self.max=np.max(x)
            self.mean=np.mean(x)
            self.std=np.std(x)
        else:
            self.min=np.min(x,axis=0)
            self.max=np.max(x,axis=0)
            self.mean=np.mean(x,axis=0)
            self.std=np.std(x,axis=0)


            
    #print it
    def __str__(self):
        string=f"Layer{self.num}:\ninp:{str(self.a)} \n\nw:{str(self.w)}\nb:{str(self.b)}\n\n out:{self.A}\n"
        return string

    #get Layer[0]
    def __getitem__(self,item):
        return self.w.T[item]    

    #set Layer[0]
    def __setitem__(self,item,value):
        self.w.T[item]=value  