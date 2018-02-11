#!/usr/local/bin/python
# -*- coding: utf-8 -*-
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# model LinkNet realization for Keras (TensorFlow?)
#
# from: https://arxiv.org/pdf/1707.03718.pdf
# 
#   Abhishek Chaurasia, Eugenio Culurciello
#   LinkNet: Exploiting Encoder Representations for Efficient Semantic Segmentation
#
#
#
# realizate by Vladimir Sorokin
#
# For FREE use
#
#
# keys: LinkNet, Keras, Boba
#
# 2017-11-25
#
# 2017-12-11 1. The add operator in blockEncoder move after batch operator and after activation operator
#            2. Add batch operator 2th conv2 in blockEncoder   
#
# 2017-12-15 1. Fix bugs with names layers
#            2. Change last layers
#            3. Add activation parameter for last layer
#            4. Add conv2D before last layer
#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from keras.models import Input, Model
from keras.layers import Convolution2D, MaxPooling2D, UpSampling2D, Conv2D, Concatenate, Activation, Dropout,Add
from keras.layers import Conv2DTranspose, SpatialDropout2D
from keras.layers.normalization import BatchNormalization

def LinkNetBoba (img_shape, n_out=1, depth=4, acti='elu', dropout=False, batch=True, acti_last='sigmoid', printOK=False):
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Encoder block
    #
    
    def blockEncoder(i, depth, maxDepth, mm, nn) :
        
        io = i
        
        #print('e 0 depth=',depth,io.shape)
        
        io = Conv2D(nn, (3, 3), strides=2, padding='same', name='conv1d'+str(maxDepth-depth))(io)
        if batch : io = BatchNormalization(name='bath1d'+str(maxDepth-depth))(io)
        io = Activation(acti)(io)
        
        #print('e 1 depth=',depth,io.shape)
        
        io = Conv2D(nn, (3, 3), padding='same', name='conv2d'+str(maxDepth-depth))(io)
        if batch : io = BatchNormalization(name='bath2d'+str(maxDepth-depth))(io)
        
        ii = Conv2D(nn, (1, 1), strides=2, name='ii1d'+str(maxDepth-depth))(i)
        if batch : io = BatchNormalization(name='bath3d'+str(maxDepth-depth))(io) # 2017-12-11 add batch
        
        io = Add()([io, ii]); io1 = io;
        
        io = Activation(acti)(io)   # 2017-12-11 change point add before activation after batch
        
        io = Conv2D(nn, (3, 3), padding='same', name='conv3d'+str(maxDepth-depth))(io)
        if batch : io = BatchNormalization(name='bath4d'+str(maxDepth-depth))(io)
        io = Activation(acti)(io)                
        
        io = Conv2D(nn, (3, 3), padding='same', name='conv4d'+str(maxDepth-depth))(io)
        if batch : io = BatchNormalization(name='bath5d'+str(maxDepth-depth))(io)
        
        io = Add()([io, io1]);     # 2017-12-11 change point add before activation after batch
        
        io = Activation(acti)(io)                
        
        ##io = Concatenate()([io, io1]);
        
        return (io)
        

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Decoder block
    #
    
    def blockDecoder(i, depth, maxDepth, mm, nn) :
        
        io = i
        
        io = Conv2D(mm//4, (1, 1), padding='same', name='convd1d'+str(maxDepth-depth))(io)
        if batch : io = BatchNormalization(name='bathd1d'+str(maxDepth-depth))(io)
        io = Activation(acti)(io)                
        
        io = Conv2DTranspose(mm//4, (3, 3), padding='same', name='convd2d'+str(maxDepth-depth))(io)
        if batch : io = BatchNormalization(name='bathd2d'+str(maxDepth-depth))(io)
        io = Activation(acti)(io)    
        
        io = UpSampling2D((2,2))(io)
        
        io = Conv2D(nn, (1, 1), padding='same', name='convd3d'+str(maxDepth-depth))(io)
        if batch : io = BatchNormalization(name='bathd3d'+str(maxDepth-depth))(io)
        io = Activation(acti)(io)                
        
        return (io)
        

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Level block (recursive call build levels) 
    #
    
    def blockLevel(i, depth, maxDepth):

        #if batch : m = BatchNormalization(name='bath1d'+str(maxDepth-depth))(m)
        #if batch and depth==maxDepth : m = BatchNormalization(name='bath1d'+str(maxDepth-depth))(m)
        
        if depth == maxDepth : return(i);
        
        emm, enn =  emm0[depth-1],  enn0[depth-1]
        dmm, dnn =  dmm0[depth-1],  dnn0[depth-1]
        
        #print('l 0 depth=',depth,i.shape)
        en  = blockEncoder(i,depth,maxDepth,emm,enn)
        
        #print('l 1 depth=',depth,en.shape)
        le  = blockLevel(en,depth+1,maxDepth)
        
        
        
        if printOK : print('l 2 depth={} en={} le={}'.format(depth,en.shape,le.shape))
        io  = Add()([en, le]);
        #io  = Concatenate()([en, le]);
        de  = blockDecoder(io,depth,maxDepth,dmm,dnn)
        #print('l L depth=',depth,en.shape)
        
        return de
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Main function block
    #
    
    maxDepth = 4
    depth    = 0
    
    emm0 = [64, 64,128,256]
    enn0 = [64,128,256,512]
    
    ##emm0 = [64, 64,128,256,512]
    ##enn0 = [64,128,256,512,1024]
    
    dmm0 = enn0
    dnn0 = emm0
    
    i = Input(shape=img_shape, name='input'); io = i
    if True : # always exists  
        io = BatchNormalization(name='bath0d'+str(maxDepth-depth))(io)
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initial function block
    #
    
    #print(io.shape)

    io = Conv2D(64, (7, 7), strides=2, padding='same', name='conv1d'+str(maxDepth-depth))(io)
    if batch : io = BatchNormalization(name='bath1d'+str(maxDepth-depth))(io)
    io = Activation(acti)(io)                
        
    io = MaxPooling2D((3, 3), strides=2, name='pool1d'+str(maxDepth-depth))(io)
    
    ##print('before',io.shape)
    
    # levels block
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Main build levels
    #
    
    io = blockLevel(io,depth+1,maxDepth)
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Final function block
    #

    io = Conv2DTranspose(32, (3, 3), padding='same', name='conv2d'+str(maxDepth-depth))(io)
    if batch : io = BatchNormalization(name='bath2d'+str(maxDepth-depth))(io)
    io = Activation(acti)(io)                
    
    io = UpSampling2D((2,2))(io)
        
    io = Conv2D(32, (3,3), padding='same', name='conv3d'+str(maxDepth-depth))(io)
    if batch : io = BatchNormalization(name='bath3d'+str(maxDepth-depth))(io)
    io = Activation(acti)(io)                
    
    if dropout : io = SpatialDropout2D(rate=dropout, name='dropLd'+str(maxDepth-depth))(io)
        
    ###io = UpSampling2D((2,2))(io) -- ?????? do bad output result 
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # The End. Build output
    #
    
    io = Conv2DTranspose(n_out, (2, 2), strides=2, padding='same', name='conv4d'+str(maxDepth-depth))(io)
    if batch : io = BatchNormalization(name='bath4d'+str(maxDepth-depth))(io)
    io = Activation(acti)(io)                

    io = Conv2D(n_out, (3,3), padding='same', name='conv5d'+str(maxDepth-depth))(io)
    if batch : io = BatchNormalization(name='bath5d'+str(maxDepth-depth))(io)
    io = Activation(acti_last)(io)

    io = Conv2D(n_out, (1,1), padding='same', name='convLd'+str(maxDepth-depth))(io)
    o = Activation(acti_last)(io)

    return Model(inputs=i, outputs=o, name='LinkNetBoba')

if 0 :

    model10 = LinkNetBoba((512,512,3), n_out=1, dropout=0.20, batch=False, acti='elu')
    model10.summary()
