#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 17:43:33 2021

@author: maclab
"""

import torch
from tqdm import tqdm
from torch.autograd import Variable
import pandas as pd

def late_fusion_chunk(testing_loader, num, input_shape_1, input_shape_2, input_shape_3, loss_func):
    model = torch.load("./model_pkl/late-30sec-batch256.pkl")
    total_testing = 0
    correct_testing = 0
    answers = []
    output = []
    model.eval()
    with torch.no_grad():
        for ((matrix1, labels1), (matrix2, labels2), (matrix3, labels3)) in tqdm(testing_loader):
            matrix1   = matrix1.cuda()
            matrix2   = matrix2.cuda()
            matrix3   = matrix3.cuda()
            matrix1   = Variable(matrix1.view(input_shape_1)).cuda()
            matrix2   = Variable(matrix2.view(input_shape_2)).cuda()
            matrix3   = Variable(matrix3.view(input_shape_3)).cuda()
            labels1   = labels1.cuda()
            labels1   = Variable(labels1).cuda()
            outputs   = model(matrix1, matrix2, matrix3).cuda()
            test_loss = loss_func(outputs, labels1)
            predicted = torch.max(outputs.data, 1)[1]
            answers.append(int(predicted[0]))
            total_testing += len(labels1)
            correct_testing += (predicted == labels1).float().sum()
            output.append(outputs)
    testing_accuracy = 100 * correct_testing / float(total_testing)
    print(testing_accuracy)
    print(correct_testing)
    print(float(total_testing))
#    ConfusionMatrix = confusion_matrix(targets_test, answers)        
    return testing_accuracy, predicted, test_loss.data, output, answers

def early_fusion_chunk(testing_loader, num, input_shape_1, input_shape_2, input_shape_3, loss_func):
    model = torch.load("./model_pkl/early-30sec-batch256.pkl")
    total_testing = 0
    correct_testing = 0
    answers = []
    output = []
    model.eval()
    with torch.no_grad():
        for ((matrix1, labels1), (matrix2, labels2), (matrix3, labels3)) in tqdm(testing_loader):
            matrix1   = matrix1.cuda()
            matrix2   = matrix2.cuda()
            matrix3   = matrix3.cuda()
            matrix1   = Variable(matrix1.view(input_shape_1)).cuda()
            matrix2   = Variable(matrix2.view(input_shape_2)).cuda()
            matrix3   = Variable(matrix3.view(input_shape_3)).cuda()
            labels1   = labels1.cuda()
            labels1   = Variable(labels1).cuda()
            outputs   = model(matrix1, matrix2, matrix3).cuda()
            test_loss = loss_func(outputs, labels1)
            predicted = torch.max(outputs.data, 1)[1]
            answers.append(int(predicted[0]))
            total_testing += len(labels1)
            correct_testing += (predicted == labels1).float().sum()
            output.append(outputs)
    testing_accuracy = 100 * correct_testing / float(total_testing)
    print(testing_accuracy)
    print(correct_testing)
    print(float(total_testing))
#    ConfusionMatrix = confusion_matrix(targets_test, answers)        
    return testing_accuracy, predicted, test_loss.data, output, answers

def SCcnn_all_chunk(testing_loader, num, input_shape_3, loss_func):
    model = torch.load("./model_pkl/short-chunk-cnn-batch256-all.pkl")
    total_testing = 0
    correct_testing = 0
    answers = []
    output = []
    model.eval()
    with torch.no_grad():
        for ((matrix1, labels1), (matrix2, labels2), (matrix3, labels3)) in tqdm(testing_loader):
#            matrix1   = matrix1.cuda()
#            matrix2   = matrix2.cuda()
            matrix3   = matrix3.cuda()
#            matrix1   = Variable(matrix1.view(input_shape_1)).cuda()
#            matrix2   = Variable(matrix2.view(input_shape_2)).cuda()
            matrix3   = Variable(matrix3.view(input_shape_3)).cuda()
            labels1   = labels1.cuda()
            labels1   = Variable(labels1).cuda()
            outputs   = model(matrix3).cuda()
            test_loss = loss_func(outputs, labels1)
            predicted = torch.max(outputs.data, 1)[1]
            answers.append(int(predicted[0]))
            total_testing += len(labels1)
            correct_testing += (predicted == labels1).float().sum()
            output.append(outputs)
    testing_accuracy = 100 * correct_testing / float(total_testing)
    print(testing_accuracy)
    print(correct_testing)
    print(float(total_testing))
#    ConfusionMatrix = confusion_matrix(targets_test, answers)        
    return testing_accuracy, predicted, test_loss.data, output, answers

def SCcnn_chunk(testing_loader, num, input_shape_3, loss_func):
    model = torch.load("./model_pkl/short-chunk-cnn-batch256.pkl")
    total_testing = 0
    correct_testing = 0
    answers = []
    output = []
    model.eval()
    with torch.no_grad():
        for ((matrix1, labels1), (matrix2, labels2), (matrix3, labels3)) in tqdm(testing_loader):
#            matrix1   = matrix1.cuda()
#            matrix2   = matrix2.cuda()
            matrix3   = matrix3.cuda()
#            matrix1   = Variable(matrix1.view(input_shape_1)).cuda()
#            matrix2   = Variable(matrix2.view(input_shape_2)).cuda()
            matrix3   = Variable(matrix3.view(input_shape_3)).cuda()
            labels1   = labels1.cuda()
            labels1   = Variable(labels1).cuda()
            outputs   = model(matrix3).cuda()
            test_loss = loss_func(outputs, labels1)
            predicted = torch.max(outputs.data, 1)[1]
            answers.append(int(predicted[0]))
            total_testing += len(labels1)
            correct_testing += (predicted == labels1).float().sum()
            output.append(outputs)
    testing_accuracy = 100 * correct_testing / float(total_testing)
    print(testing_accuracy)
    print(correct_testing)
    print(float(total_testing))
#    ConfusionMatrix = confusion_matrix(targets_test, answers)        
    return testing_accuracy, predicted, test_loss.data, output, answers


def Using_joint_model(testing_loader, 
                      input_shape_1, 
                      input_shape_2, 
                      input_shape_3):
    model = torch.load("./model_pkl/joint-model-all-2500_200.pkl")
    answers = []
    model.eval()
    with torch.no_grad():
        for (matrix1, matrix2, matrix3) in tqdm(testing_loader):
            matrix1   = matrix1[0].cuda()
            matrix2   = matrix2[0].cuda()
            matrix3   = matrix3[0].cuda()
            matrix1   = Variable(matrix1.view(input_shape_1)).cuda()
            matrix2   = Variable(matrix2.view(input_shape_2)).cuda()
            matrix3   = Variable(matrix3.view(input_shape_3)).cuda()
            outputs   = model(matrix1, matrix2, matrix3).cuda()
            predicted = torch.max(outputs.data, 1)[1]
            answers.append(int(predicted[0]))
    print("Done!!!")      
    return answers

def song_level_test(answers, z_test):
    songidtest = pd.Categorical(z_test)
    songidtest = songidtest.categories
    genre_list = pd.read_csv("genre_list.csv")
    genre_list = list(genre_list["0"])
    test_ = pd.DataFrame({"answer": answers, "id" : z_test})
    predict = []
    song = []
    for i in tqdm(range(len(songidtest))):
        temp_number = test_[test_["id"] == songidtest[i]]["answer"].value_counts().idxmax()
        predict.append(genre_list[temp_number])
        song.append(songidtest[i])
    result = pd.DataFrame({"predict" : predict, "song" : song})
    result.to_csv("result.csv", sep=",")
    return result