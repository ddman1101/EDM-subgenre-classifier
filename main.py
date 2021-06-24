#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 12:05:01 2021

@author: maclab
"""

"""
Name     : main.py
Function : Execute the EDM-subgenre-classification (Just for Using)

Introduction : There are 3 model to use. (Short-chunk cnn + Resnet, Short-chunk
               cnn + late-fusion, Short-chunk cnn + early-fusion)

"""

import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "3,4,5"

import torch
import torch.nn as nn
from model import Joint_ShortChunkCNN_Res
from dataset import set_the_dataset_using, save_feature, ConcatDataset
from testing import late_fusion_chunk, early_fusion_chunk, SCcnn_all_chunk, SCcnn_chunk, Using_joint_model, song_level_test

# EDM subgenre list with the original order 
if __name__ == "__main__":
    
    device = torch.device('cuda')
    batch_size  = 16
    model       = Joint_ShortChunkCNN_Res().to(device)
    model       = nn.DataParallel(model)
    print(model)
    input_shape_1 = (-1,384,50)
    input_shape_2 = (-1,193,50)
    input_shape_3 = (-1,1,128,200)
    
    # Calculate the audio to npy and loading data into datalaoder
    print("Start to extracting feature and set up the dataset")
    save_feature(data_folder = "./data/audio/")
    feature_t, feature_f, feature_m, z_test = set_the_dataset_using()
    data_loader = torch.utils.data.DataLoader(
            
            ConcatDataset(feature_t, feature_f, feature_m),                                           
            batch_size = 16,
            shuffle = False
            
            )
    print("Done!")
    print("Start predict the tracks")
    answers = Using_joint_model(data_loader,
                                input_shape_1,
                                input_shape_2,
                                input_shape_3)
    
    result = song_level_test(answers, z_test)
    print("Done! The result is under the folder and called result.csv")