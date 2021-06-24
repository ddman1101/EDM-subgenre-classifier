#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 17 12:10:10 2021

@author: maclab
"""

import torch.nn as nn
import math
import torch
from torch.autograd import Variable

def fit_model_3input(model,
                     loss_func,
                     optimizer,
                     input_shape_1,
                     input_shape_2,
                     input_shape_3,
                     num_epochs,
                     train_loader,
                     test_loader,
                     patience = 10,
                     save     = False,
                     name     = "joint-model-small"):
    # Training the Model
    last_loss = 100
    temp_times = 0
    training_loss = []
    training_accuracy = []
    validation_loss = []
    validation_accuracy = []
    m = nn.DataParallel(model)
    m.train()
    for epoch in range(num_epochs):
        #training model & store loss & acc / epoch
        correct_train = 0
        total_train = 0
        for i, ((images_1, labels_1), (images_2, labels2), (images_3, labels3)) in enumerate(train_loader):
            # 1.Define variables
#            temp_ = []
            images_1 = images_1.cuda()
            labels_1 = labels_1.cuda()
            images_2 = images_2.cuda()
            images_3 = images_3.cuda()
            train_1  = Variable(images_1.view(input_shape_1))
            train_2  = Variable(images_2.view(input_shape_2))
            train_3  = Variable(images_3.view(input_shape_3))
            labels_1 = Variable(labels_1)
            # 2.Clear gradients
            optimizer.zero_grad()
            # 3.Forward propagation
            outputs = model(train_1, train_2, train_3).cuda()
#            print(outputs.shape, labels_1)
            if len(outputs.shape) == 1:
                outputs = torch.reshape(outputs, (1, outputs.shape[0]))
            # 4.Calculate softmax and cross entropy loss
            train_loss = loss_func(outputs, labels_1)
            # 5.Calculate gradients
            train_loss.backward()
            # 6.Update parameters
            optimizer.step()
            # 7.Get predictions from the maximum value
            predicted = torch.max(outputs.data, 1)[1]
            # 8.Total number of labels
            total_train += len(labels_1)
            # 9.Total correct predictions
            correct_train += (predicted == labels_1).float().sum()
        # 10.store val_acc / epoch
        train_accuracy = 100 * correct_train / float(total_train)
        training_accuracy.append(train_accuracy)
        # 11.store loss / epoch
        training_loss.append(train_loss.data)
        #evaluate model & store loss & acc / epoch
        correct_test = 0
        total_test = 0
        for ((images_1, labels_1), (images_2, labels2), (images_3, labels_3)) in test_loader:
            # 1.Define variables
            images_1      = images_1.cuda()
            labels_1      = labels_1.cuda()
            images_2      = images_2.cuda()
            images_3      = images_3.cuda()
#            labels_1 = labels_1.cuda()
            test_1        = Variable(images_1.view(input_shape_1))
            test_2        = Variable(images_2.view(input_shape_2))
            test_3        = Variable(images_3.view(input_shape_3))
            labels_1      = Variable(labels_1)
            # 2.Forward propagation
            outputs       = model(test_1, test_2, test_3).cuda()
            if len(outputs.shape) == 1:
                outputs   = torch.reshape(outputs, (1, outputs.shape[0]))
            # 3.Calculate softmax and cross entropy loss
            val_loss      = loss_func(outputs, labels_1)
            # 4.Get predictions from the maximum value
            predicted     = torch.max(outputs.data, 1)[1]
            # 5.Total number of labels
            total_test   += len(labels_1)
            # 6.Total correct predictions
            correct_test += (predicted == labels_1).float().sum()
        # 6.store val_acc / epoch
        val_accuracy = 100 * correct_test / float(total_test)
        validation_accuracy.append(val_accuracy)
        # 11.store val_loss / epoch
        validation_loss.append(val_loss.data)
        if save == True:
            torch.save(model, './tempogram_model/joint_model/{}_{}.pkl'.format(name, epoch+1))
#            torch.save({
#                    'epoch': epoch+1,
#                    'model_state_dict'     : model.state_dict(),
#                    'optimizer_state_dict' : optimizer.state_dict(),
#                    'training_loss'        : train_loss.data,
#                    'validation_loss'      : val_loss.data,
#            }, './tempogram_model/{}-{}.pkl'.format(name, epoch+1))
        print('Train Epoch: {}/{} Traing_Loss: {} Traing_acc: {:.6f}% Val_Loss: {} Val_accuracy: {:.6f}%'.format(epoch+1, num_epochs, train_loss.data, train_accuracy, val_loss.data, val_accuracy))
        # 12."Early Stopping" 
        if epoch == 0 :
            if validation_loss[epoch] >= last_loss:
                temp_times = temp_times + 1
                print('trigger times : {}'.format(temp_times))
                if temp_times > patience :
                    print('Early stopping!\nStart to test process.')
                    return training_loss, training_accuracy, validation_loss, validation_accuracy
            else : 
                temp_times = 0
                print('trigger times: 0')
        else :
            if validation_loss[epoch] >= last_loss:
                temp_times = temp_times + 1
                print('trigger times : {}'.format(temp_times))
                if temp_times > patience :
                    print('Early stopping!\nStart to test process.')
                    return training_loss, training_accuracy, validation_loss, validation_accuracy
            else : 
                temp_times = 0
                print('trigger times: 0')
        last_loss = validation_loss[epoch]
    return training_loss, training_accuracy, validation_loss, validation_accuracy

def fit_model_1input(model,
                     loss_func,
                     optimizer,
                     input_shape_1,
                     input_shape_2,
                     input_shape_3,
                     num_epochs,
                     train_loader,
                     test_loader,
                     patience = 10,
                     save     = False,
                     name     = "SCcnn-model"):
    # Training the Model
    last_loss = 100
    temp_times = 0
    training_loss = []
    training_accuracy = []
    validation_loss = []
    validation_accuracy = []
    m = nn.DataParallel(model)
    m.train()
#    output_ = []
#    model_ = []
    for epoch in range(num_epochs):
        #training model & store loss & acc / epoch
        correct_train = 0
        total_train = 0
        for i, ((images_1, labels_1), (images_2, labels2), (images_3, labels3)) in enumerate(train_loader):
            # 1.Define variables
            labels_1 = labels_1.to(device)
            images_3 = images_3.to(device)
            train_3  = Variable(images_3.view(input_shape_3))
            labels_1 = Variable(labels_1)
            # 2.Clear gradients
            optimizer.zero_grad()
            # 3.Forward propagation
            outputs = model(train_3).to(device)
            if len(outputs.shape) == 1:
                outputs = torch.reshape(outputs, (1, outputs.shape[0]))
            # 4.Calculate softmax and cross entropy loss
            train_loss = loss_func(outputs, labels_1)
            # 5.Calculate gradients
            train_loss.backward()
            # 6.Update parameters
            optimizer.step()
            # 7.Get predictions from the maximum value
            predicted = torch.max(outputs.data, 1)[1]
            # 8.Total number of labels
            total_train += len(labels_1)
            # 9.Total correct predictions
            correct_train += (predicted == labels_1).float().sum()
        # 10.store val_acc / epoch
        train_accuracy = 100 * correct_train / float(total_train)
        training_accuracy.append(train_accuracy)
        # 11.store loss / epoch
        training_loss.append(train_loss.data)
        #evaluate model & store loss & acc / epoch
        correct_test = 0
        total_test = 0
        for ((images_1, labels_1), (images_2, labels2), (images_3, labels_3)) in test_loader:
            # 1.Define variables
            images_1 = images_1.to(device)
            labels_1 = labels_1.to(device)
            images_2 = images_2.to(device)
            images_3 = images_3.to(device)
#            labels_1 = labels_1.cuda()
            test_1 = Variable(images_1.view(input_shape_1))
            test_2 = Variable(images_2.view(input_shape_2))
            test_3 = Variable(images_3.view(input_shape_3))
            labels_1 = Variable(labels_1)
            # 2.Forward propagation
            outputs = model(test_1, test_2, test_3).to(device)
            if len(outputs.shape) == 1:
                outputs = torch.reshape(outputs, (1, outputs.shape[0]))
            # 3.Calculate softmax and cross entropy loss
            val_loss = loss_func(outputs, labels_1)
            # 4.Get predictions from the maximum value
            predicted = torch.max(outputs.data, 1)[1]
            # 5.Total number of labels
            total_test += len(labels_1)
            # 6.Total correct predictions
            correct_test += (predicted == labels_1).float().sum()
        # 6.store val_acc / epoch
        val_accuracy = 100 * correct_test / float(total_test)
        validation_accuracy.append(val_accuracy)
        # 11.store val_loss / epoch
        validation_loss.append(val_loss.data)
        if save == True:
            torch.save(model, './tempogram_model/joint_model/{}_{}.pkl'.format(name, epoch+1))
#            torch.save({
#                    'epoch': epoch+1,
#                    'model_state_dict'     : model.state_dict(),
#                    'optimizer_state_dict' : optimizer.state_dict(),
#                    'training_loss'        : train_loss.data,
#                    'validation_loss'      : val_loss.data,
#            }, './tempogram_model/{}-{}.pkl'.format(name, epoch+1))
        print('Train Epoch: {}/{} Traing_Loss: {} Traing_acc: {:.6f}% Val_Loss: {} Val_accuracy: {:.6f}%'.format(epoch+1, num_epochs, train_loss.data, train_accuracy, val_loss.data, val_accuracy))
        # 12."Early Stopping" 
        if epoch == 0 :
            if validation_loss[epoch] >= last_loss:
                temp_times = temp_times + 1
                print('trigger times : {}'.format(temp_times))
                if temp_times > patience :
                    print('Early stopping!\nStart to test process.')
                    return training_loss, training_accuracy, validation_loss, validation_accuracy
            else : 
                temp_times = 0
                print('trigger times: 0')
        else :
            if validation_loss[epoch] >= last_loss:
                temp_times = temp_times + 1
                print('trigger times : {}'.format(temp_times))
                if temp_times > patience :
                    print('Early stopping!\nStart to test process.')
                    return training_loss, training_accuracy, validation_loss, validation_accuracy
            else : 
                temp_times = 0
                print('trigger times: 0')
        last_loss = validation_loss[epoch]
    return training_loss, training_accuracy, validation_loss, validation_accuracy

