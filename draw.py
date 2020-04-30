import os
import numpy as np
import matplotlib.pyplot as plt
def draw(data_list,title,ylabel,name,root_dir):
    data_list = np.array(data_list)
    train = data_list[:,0]
    dev = data_list[:,1]
    test = data_list[:,2]
    epoches = np.linspace(0,len(data_list)-1,len(data_list))
    plt.plot(epoches,train,label='train' )
    plt.plot(epoches, dev, label='dev')
    plt.plot(epoches, test, label='test')
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel('epoch')
    plt.legend()
    plt.savefig(os.path.join(root_dir,name+ylabel)+".png")
    plt.close()
