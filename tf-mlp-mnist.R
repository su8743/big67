#!/usr/bin/env /c/Apps/Anaconda3/Scripts/Rscript

# 손글씨 Recognition
# modified
# Author : Yibeck Lee(yibeck.lee@gmail.com)
# references
# 1) https://tensorflow.rstudio.com/tensorflow/
# install.packages("tensorflow")
library(tensorflow)
library(reticulate)

datasets = tf$contrib$learn$datasets
mnist = datasets$mnist$read_data_sets("./data/mnist/input_data", one_hot = TRUE)

num_obs = 5000

learningRate = 0.1
nFeatures = 784L
nLabel = 10L
nHidden01Neuron = 784L
nHidden02Neuron = 784L

nLoop = 10L

features = tf$placeholder(tf$float32, shape(NULL, nFeatures), name="placeholder_for_features")
label = tf$placeholder(tf$float32, shape(NULL, nLabel), name = "placeholder_for_label")

Weight_f_to_h01 = tf$Variable(tf$random_normal(shape(nFeatures, nHidden01Neuron)))
bias_f_to_h01 = tf$Variable(tf$random_normal(shape(nHidden01Neuron)))
layer01 = tf$add(tf$matmul(features, Weight_f_to_h01), bias_f_to_h01)
layer01 = tf$nn$tanh(layer01)

Weight_f_to_h02 = tf$Variable(tf$random_normal(shape(nHidden01Neuron, nHidden02Neuron)))
bias_f_to_h02 = tf$Variable(tf$random_normal(shape(nHidden02Neuron)))
layer02 = tf$add(tf$matmul(features, Weight_f_to_h02), bias_f_to_h02)
layer02 = tf$nn$tanh(layer02)


Weight_h02_to_out = tf$Variable(tf$zeros(shape(nHidden02Neuron, nLabel)))
bias_h02_to_out = tf$Variable(tf$zeros(shape(nLabel)))

hypothesis = tf$add(tf$matmul(layer02, Weight_h02_to_out), bias_h02_to_out) 
hypothesis = tf$nn$softmax(hypothesis)


cost = tf$reduce_mean(-tf$reduce_sum(label * tf$log(hypothesis), reduction_indices=1L))
optimizer = tf$train$GradientDescentOptimizer(learningRate)$minimize(cost)

epochs = 10
num_mini_batch = 50L

# session 초기화 및 변수 초기화
sess = tf$Session()
sess$run(tf$global_variables_initializer())
nloop = as.integer(num_obs/num_mini_batch)
# batch size 50개씩 ???회 Learning
for (epoch in 1:epochs){
    trainData = mnist$train$next_batch(num_mini_batch)
    for (i in 1:nloop) {
        trainFeatures = trainData[[1]]
        trainLabels = trainData[[2]]

        sess$run(optimizer,
                 feed_dict = dict(features = trainFeatures, label = trainLabels))
        if(i %% num_mini_batch == 0) {
    		correct_prediction = tf$equal(tf$argmax(hypothesis, 1L), tf$argmax(label, 1L))
    		accuracy = tf$reduce_mean(tf$cast(correct_prediction, tf$float32))
    		# print(c(i,sess$run(accuracy, feed_dict=dict(x = mnist$train$images, label = mnist$train$labels))))
    		cat(sprintf('epoch %d - iteration#%5d : accuracy=%.4f\n',epoch,i,sess$run(accuracy,feed_dict=dict(features = mnist$train$images, label = mnist$train$labels))))
        }
    }
}
# test minist를 대상으로 예측 및 정확도 산출
correct_prediction = tf$equal(tf$argmax(hypothesis, 1L), tf$argmax(label, 1L))
accuracy = tf$reduce_mean(tf$cast(correct_prediction, tf$float32))
sess$run(accuracy, feed_dict=dict(features = mnist$test$images, label = mnist$test$labels))

# with(tf$Session() %as% sess, {
#     tf$get_variable_scope()$reuse_variables()
#     sess$run(tf$global_variables_initializer()) # 변수 초기화

# 	for (i in 1:num_obs*epochs) {
# 	    batches = mnist$train$next_batch(num_mini_batch)
# 	    batch_xs = batches[[1]]
# 	    batch_ys = batches[[2]]
# 	    sess$run(train_step,
# 	             feed_dict = dict(x = batch_xs, label = batch_ys))
# 	    if(i %% 500 ==0) {
# 			correct_prediction = tf$equal(tf$argmax(model, 1L), tf$argmax(label, 1L))
# 			accuracy = tf$reduce_mean(tf$cast(correct_prediction, tf$float32))
# 			# print(c(i,sess$run(accuracy, feed_dict=dict(x = mnist$test$images, label = mnist$test$labels))))
# 			cat(sprintf('iteration#%5d : accuracy=%.4f\n',i,sess$run(accuracy,feed_dict=dict(x = mnist$test$images, label = mnist$test$labels))))
# 	    }
# 	}
# 	# test minist를 대상으로 예측 및 정확도 산출
# 	correct_prediction = tf$equal(tf$argmax(model, 1L), tf$argmax(label, 1L))
# 	accuracy = tf$reduce_mean(tf$cast(correct_prediction, tf$float32))
# 	# print(c(i,sess$run(accuracy, feed_dict=dict(x = mnist$test$images, label = mnist$test$labels))))
# 	cat(sprintf('iteration#%d accuracy=%.4f\n',i,sess$run(accuracy,feed_dict=dict(xcorrect_pred= mnist$test$labels))))
# })

testData = mnist$test$next_batch(1L)
testFeatures = testData[[1]]
testLabel = testData[[2]]

imgMatrix = matrix(as.numeric(testFeatures), nrow = 28, byrow = TRUE)
imgMatrix = apply(imgMatrix, 2, rev)
image(t(imgMatrix), col = grey.colors(255))

answer = sess$run(hypothesis, feed_dict = dict( features = testFeatures ))
print('test label')
print(testLabel)
print(c("test of letter : ", which.max(testLabel) - 1))
print('answer')
print(answer)
print(c("answer of letter : ", which.max(answer) - 1))
