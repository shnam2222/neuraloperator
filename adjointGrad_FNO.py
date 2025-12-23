"""
@author: Zongyi Li
This file is the Fourier Neural Operator for 1D problem Training a FNO on RCWA Adjoint Gradient 
[paper](https://arxiv.org/pdf/2010.08895.pdf)
=============================

"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import matplotlib.pyplot as plt
import scipy
from scipy import io
import random

# Import global variables
from config import instance

import operator
from functools import reduce
from functools import partial
from timeit import default_timer
from FNO_Adj import FNO1D

#import sys
#sys.path.append('C:\\Users\\user\\neuraloperator\\master')
from utilities3 import *

torch.manual_seed(0)
np.random.seed(0)


#modes = 16
width = 128
modes = 16

################################################################
# read data
################################################################

# Data is of the shape (number of samples, grid size)

def train_FNO():

    #  configurations all in config
    n_total = instance.n_total
    n_train = instance.n_train
    n_test = instance.n_test
    epochs = instance.epochs
    batch_size = instance.batch_size
    learning_rate = instance.learning_rate
    step_size = instance.step_size
    gamma = instance.gamma
    size  = instance.size
    ###################################


    dataset =  io.loadmat('C:\\Users\\user\\archive\\neuraloperator\\Adjoint\\dataset_adj_cutoff_norm.mat')

    sample = random.sample(range(0, n_total), (n_train+n_test))
    train_sample = sample[0:n_train]
    test_sample = sample[-n_test:]


    #pattern_train = np.transpose(dataset['img'][:,0:n_train])
    pattern_train = np.transpose(dataset['img'][:,train_sample])
    pattern_train = torch.tensor(np.float32(pattern_train))
    #gradient_train = dataset['Gradient'][0:n_train,:]
    gradient_train = dataset['Gradient'][train_sample,:]
    gradient_train = torch.tensor(np.float32(gradient_train))


    #pattern_test = np.transpose(dataset['img'][:,-n_test:])
    pattern_test = np.transpose(dataset['img'][:,test_sample])
    #gradient_test = dataset['Gradient'][-n_test:,:]
    gradient_test = dataset['Gradient'][test_sample,:]
    pattern_test = torch.tensor(np.float32(pattern_test))
    gradient_test = torch.tensor(np.float32(gradient_test))

    grid = np.linspace(0, 1, size).reshape(1, size, 1)
    grid = torch.tensor(grid, dtype=torch.float)

    #pattern_train = torch.cat([pattern_train.reshape(n_train,size,1), grid.repeat(n_train,1,1)], dim=2)
    #pattern_test = torch.cat([pattern_test.reshape(n_test,size,1), grid.repeat(n_test,1,1)], dim=2)

    pattern_train = pattern_train.reshape(n_train,size,1)
    pattern_test = pattern_test.reshape(n_test,size,1)

    print(np.shape(pattern_train))
    print(np.shape(gradient_train))
    print(np.shape(pattern_test))
    print(np.shape(gradient_test))

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(pattern_train, gradient_train), batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(pattern_test, gradient_test), batch_size=batch_size, shuffle=True)

    # model
    model = FNO1D(modes, width).cuda()
    print(model.count_params())

    ################################################################
    # training and evaluation
    ################################################################


    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    total_time = 0

    myloss = LpLoss(size_average=False)
    for ep in range(epochs):
        model.train()
        t1 = default_timer()
        train_mse = 0
        train_l2 = 0
        for x, y in train_loader:
            x, y = x.cuda(), y.cuda()

            optimizer.zero_grad()
            #print(np.shape(x))
            out = model(x)

            mse = F.mse_loss(out, y, reduction='mean')
            # mse.backward()
            l2 = myloss(out.view(batch_size, -1), y.view(batch_size, -1))
            l2.backward() # use the l2 relative loss

            optimizer.step()
            train_mse += mse.item()
            train_l2 += l2.item()

        scheduler.step()
        model.eval()
        test_l2 = 0.0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.cuda(), y.cuda()

                out = model(x)
                test_l2 += myloss(out.view(batch_size, -1), y.view(batch_size, -1)).item()

        train_mse /= len(train_loader)
        train_l2 /= n_train
        test_l2 /= n_test

        t2 = default_timer()
        print(ep, t2-t1, train_mse, train_l2, test_l2)
        total_time = total_time + t2 -t1


    print("Total Time: ", total_time)
    randn = random.randrange(n_total)
    sample_pattern = np.transpose(dataset['img'][:,randn])
    sample_pattern = torch.tensor(np.float32(sample_pattern))
    sample_gradient = dataset['Gradient'][randn,:]
    #sample = torch.cat([sample_pattern.reshape(1,size,1), grid.repeat(1,1,1)], dim=2)
    sample = sample_pattern.reshape(1,size,1)

    sample_out = model(sample.cuda())
    sample_out = sample_out.cpu()
    sample_out = sample_out.detach().numpy()
    grid = grid.numpy()
    '''
    plt.plot(np.reshape(grid, 256), sample_gradient, 'r-')
    plt.plot(np.reshape(grid, 256), sample_out, 'b-')
    plt.ylabel("normalized gradient")
    plt.xlabel("grid")
    plt.show()
    '''

    print("RMSE: ", np.linalg.norm(sample_gradient-sample_out))

    test_RMSE = np.array([])

    errors = 0


    for i in range(n_test):
        randn = random.randrange(n_total)
        sample_pattern = np.transpose(dataset['img'][:,randn])
        sample_pattern = np.expand_dims(sample_pattern,axis=-1)
        sample_pattern = np.expand_dims(sample_pattern,axis= 0)
        sample = torch.tensor(np.float32(sample_pattern))
        sample_gradient = dataset['Gradient'][randn,:]
        
        sample_out = model(sample.cuda())
        sample_out = sample_out.cpu()
        sample_out = sample_out.detach().numpy()
        sample_out = sample_out.reshape(size)
        sample_gradient = sample_gradient.reshape(size)
        #print("RMSE: ", np.linalg.norm(sample_gradient-sample_out))
        test_RMSE = np.append(test_RMSE, np.linalg.norm(sample_gradient-sample_out))

        if np.linalg.norm(sample_gradient-sample_out)>0.2:
            '''
            plt.plot(np.reshape(grid, 256), sample_gradient, 'r-')
            plt.plot(np.reshape(grid, 256), sample_out, 'b-')
            plt.ylabel("normalized gradient")
            plt.xlabel("grid")
            plt.show()
            '''
            errors = errors+1

    print(test_RMSE)
    print("Total Errors: ", errors)

    print("Average RMSE: ", np.average(test_RMSE),"STDEV", np.std(test_RMSE))
    x = np.linspace(1,n_test, n_test)
    x = x.reshape(n_test)
    test_RMSE = test_RMSE.reshape(n_test)
    '''
    plt.scatter(x, test_RMSE)
    plt.show()
    '''
    #results_dict = {"Training Time": total_time, "Test Results": test_RMSE}
    #filename = "FNO_"+"n_train_"+str(n_train)+"_epochs_"+str(epochs)+"_" + str(trial_no) + ".mat"
    #io.savemat(file_name=filename, mdict = results_dict, format = '5')

    return total_time, test_RMSE

#torch.save(model, 'C:\\Users\\user\\neuraloperator\\Adjoint\\normalized_FNO.pt')

'''
# torch.save(model, 'model/ns_fourier_burgers_8192')
pred = torch.zeros(gradient_test.shape)
index = 0
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(pattern_test, gradient_test), batch_size=1, shuffle=False)
with torch.no_grad():
    for x, y in test_loader:
        test_l2 = 0
        x, y = x.cuda(), y.cuda()

        out = model(x)
        pred[index] = out

        test_l2 += myloss(out.view(1, -1), y.view(1, -1)).item()
        print(index, test_l2)
        index = index + 1

# scipy.io.savemat('pred/burger_test.mat', mdict={'pred': pred.cpu().numpy()})
'''