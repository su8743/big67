#!/usr/bin/env /c/Apps/Anaconda3/Scripts/Rscript

# Letter visualization
# Author : Yibeck Lee(yibeck.lee@gmail.com)

library(tensorflow)
library(ggplot2)

# 손글씨 데이터 가져오기
datasets = tf$contrib$learn$datasets

# local에 손글씨 데이터 저장
mnist = datasets$mnist$read_data_sets("./data/mnist/input_data", one_hot = TRUE)
print(mnist)
par(mfrow = c(3, 3)) # set plot options back to default
imageId = 8 # letter = 0
imageId = 21 # letter = 1
imageId = 12 # letter = 3
imageId = 34 # letter = 4
imageId = 120 # letter = 8

# image #10의 length =  784
print(c('length of image#10', length(mnist$train$images[imageId,])))
# digit [0,1,2,3,4,5,6,7,8,9]
# 784 = 28 * 28
imgMatrix = matrix(as.numeric(mnist$train$images[imageId,]), nrow = 28, byrow = TRUE)
# digit = matrix(as.numeric(mnist$train$images[imageId,]), ncol = 28)
imgMatrix = apply(imgMatrix, 2, rev)
png("letter_1.png", height = 800, width = 600)
image(t(imgMatrix), col = grey.colors(255))
dev.off()
system('explorer letter_1.png')
print(mnist$train$labels[imageId,])
print(which.max(mnist$train$labels[imageId,]) - 1)

# Sampling 6000개 손글씨  
letters = 6000
x_train = mnist$train$images[1:letters,]
y_train = mnist$train$labels[1:letters,]
# 특정 image visualization
imageId = 90
print(c('length of image#10', length(x_train[imageId,])))
imgMatrix = matrix(as.numeric(x_train[imageId,]), nrow = 28, byrow = TRUE)
imgMatrix = apply(imgMatrix, 2, rev)
png("letter_2.png", height = 800, width = 600)
image(t(imgMatrix), col = grey.colors(255))
dev.off()
system('explorer letter_2.png')
print(y_train[imageId,])
print(which.max(y_train[imageId,]) - 1)
