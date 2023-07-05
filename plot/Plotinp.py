import matplotlib.pyplot as plt
import numpy as np

xpoints = [[1, 8,11]]
ypoints = np.array([3, 10,11])
# a function to plot all xpoints in a array
def plots(xpoints=[[1,2,3]],ypoints=None,title="No title",leg=["leg"],xlabel="xlabel",ylabel="ylabel"):
    plt.figure(figsize=(15,6))
    if(type(ypoints)==type(None)):
        ypoints=np.arange(0,len(xpoints[0]))
    for i in range(len(xpoints)):
        plt.plot(ypoints,xpoints[i],label=leg[i])
    plt.xticks(ypoints)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.show()

if __name__=="__main__":
    plots(xpoints)