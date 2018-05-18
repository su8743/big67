#!/usr/bin/env /c/Apps/Anaconda3/Scripts/Rscript

# XOR function
# Author : Yibeck Lee(yibeck.lee@gmail.com)
# features, label
# nFeature, nLabel
# learningRate
# nHidden01Neuron
# plFeatures, plLabel

args <- commandArgs(TRUE)

library(tensorflow)

features = c(0,1,0,1,0,0,1,1)
features = array(c(features),dim=c(4,2))
print(features)
# one hot encoder : 0 = (1,0) 1 = (0,1)
label = c(0,1,1,0,1,0,0,1)
label = array(c(label),dim=c(4,2))
print(label)

nFeature = 2
nLabel = 2

# learningRate = 0.2
# nHidden01Neuron = 10L
# nLoop = 50

learningRate = as.double(args[1])
nHidden01Neuron = as.double(args[2])
nLoop = as.double(args[3])


plFeatures = tf$placeholder(tf$float32, shape(NULL, nFeature), name="placeholder_for_features")
plLabel = tf$placeholder(tf$float32, shape(NULL, nLabel), name = "placeholder_for_label")
print(c(plFeatures,plLabel))

Weight_f_to_h01 = tf$Variable(tf$random_normal(shape(nFeature, nHidden01Neuron)))
bias_f_to_h01 = tf$Variable(tf$random_normal(shape(nHidden01Neuron)))
layer01 = tf$add(tf$matmul(plFeatures, Weight_f_to_h01), bias_f_to_h01)
layer01 = tf$nn$tanh(layer01)


Weight_h01_to_out = tf$Variable(tf$zeros(shape(nHidden01Neuron, nLabel)))
bias_h01_to_out = tf$Variable(tf$zeros(shape(nLabel)))

hypothesis = tf$add(tf$matmul(layer01, Weight_h01_to_out), bias_h01_to_out) 
hypothesis = tf$nn$softmax(hypothesis)


cost = tf$reduce_mean(-tf$reduce_sum(plLabel * tf$log(hypothesis), reduction_indices=1L))

forwardpropagation = tf$train$GradientDescentOptimizer(learningRate)$minimize(cost)
# forwardpropagation = tf$train$AdamOptimizer(learning_rate)$minimize(cost)


correct_predLabel = tf$equal(tf$argmax(hypothesis, 1L), tf$argmax(plLabel, 1L))
accuracy = tf$reduce_mean(tf$cast(correct_predLabel, tf$float32))

init = tf$global_variables_initializer()

with(tf$Session() %as% sess, {

    sess$run(init)

    for (i in 1:nLoop) {

        correct_predLabel = tf$equal(tf$argmax(hypothesis, 1L), tf$argmax(label, 1L))
        accuracy = tf$reduce_mean(tf$cast(correct_predLabel, tf$float32))

		current_cost = sess$run(cost, feed_dict = dict(plFeatures = features, plLabel = label))

        cat(sprintf('i=%d : cost=%f accuracy=%f\n'
            ,i
            ,current_cost
            ,sess$run(accuracy, feed_dict=dict(plFeatures = features, plLabel = label))
            ))
    }

input_x_ = c(0,0)
input_x = array(c(input_x_),dim=c(1,2))
result = sess$run(hypothesis, feed_dict=dict(plFeatures = input_x))
cat(sprintf('%d, %d, %f %f\n', input_x[1], input_x[2], result[1], result[2]))

input_x_ = c(0,1)
input_x = array(c(input_x_),dim=c(1,2))
result = sess$run(hypothesis, feed_dict=dict(plFeatures = input_x))
cat(sprintf('%d, %d, %f %f\n', input_x[1], input_x[2], result[1], result[2]))

input_x_ = c(1,0)
input_x = array(c(input_x_),dim=c(1,2))
result = sess$run(hypothesis, feed_dict=dict(plFeatures = input_x))
cat(sprintf('%d, %d, %f %f\n', input_x[1], input_x[2], result[1], result[2]))

input_x_ = c(1,1)
input_x = array(c(input_x_),dim=c(1,2))
result = sess$run(hypothesis, feed_dict=dict(plFeatures = input_x))
cat(sprintf('%d, %d, %f %f\n', input_x[1], input_x[2], result[1], result[2]))
})
