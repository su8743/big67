#!/usr/bin/env /c/Apps/Anaconda3/Scripts/Rscript

# 손글씨 CNN- Convolution Neural Network
# modified
# Author : Yibeck Lee(yibeck.lee@gmail.com)
# references
# 1) https://www.tensorflow.org/get_started/mnist/pros
# 2) http://rstudio-pubs-static.s3.amazonaws.com/272694_a5359dfe9cd9420292e358f519cfc94e.html

library(tensorflow)

datasets = tf$contrib$learn$datasets
mnist = datasets$mnist$read_data_sets("./data/mnist/input_data", one_hot = TRUE)

sess = tf$InteractiveSession()
features = tf$placeholder(tf$float32, shape(NULL, 784L), name="input")
label = tf$placeholder(tf$float32, shape(NULL, 10L))

features_image = tf$reshape(
    tensor = features
  , shape = shape(-1L, 28L, 28L, 1L) # input*width*height*channel= 1*28*28*1, -1L : inference - flattens(일직선으로 펼침)
)

# filter size = 5*5, input channels = 1, output channels = 32
# 가중치 : 정규분포의 2 standard deviations 내에 truncate
W_conv1 = tf$Variable(tf$truncated_normal(shape(5L, 5L, 1L, 32L), stddev=0.1))
b_conv1 = tf$Variable(tf$constant(0.1, shape=shape(32L)))

h_conv1 = tf$nn$relu(
  tf$nn$conv2d(
     input = features_image
    ,filter = W_conv1
    ,strides = c(1L,1L,1L,1L) 
    ,padding = 'SAME'
    )
  + b_conv1
)

h_pool1 = tf$nn$max_pool(
   value = h_conv1
  ,ksize=c(1L, 2L, 2L, 1L)
  ,strides=c(1L, 2L, 2L, 1L) 
  ,padding='SAME')

# filter size = 5*5, input channel = 32, output channel = 64
W_conv2 = tf$Variable(tf$truncated_normal(shape(5L, 5L, 32L, 64L), stddev=0.1))
b_conv2 = tf$Variable(tf$constant(0.1, shape=shape(64L)))
h_conv2 = tf$nn$relu(
  tf$nn$conv2d(
     input =  h_pool1
    ,filter = W_conv2
    ,strides = c(1L,1L,1L,1L) 
    ,padding = 'SAME'
    )
  + b_conv2
)
h_pool2 = tf$nn$max_pool(
   value = h_conv2
  ,ksize=c(1L, 2L, 2L, 1L)
  ,strides=c(1L, 2L, 2L, 1L) 
  ,padding='SAME')

# 7*7*64 
h_pool2_flat = tf$reshape(h_pool2, shape(-1L, 7L * 7L * 64L))

# fully connecting 7*7*64 to 1024 connected layer 
W_fc1 = tf$Variable(tf$truncated_normal(shape(7L * 7L * 64L, 1024L), stddev=0.1))
b_fc1 = tf$Variable(tf$constant(0.1, shape=shape(1024L)))
h_fc1 = tf$nn$relu(tf$matmul(h_pool2_flat, W_fc1) + b_fc1)
keep_prob = tf$placeholder(tf$float32)
h_fc1_drop = tf$nn$dropout(h_fc1, keep_prob)

# 1024 connected layer to 10 softmax layer
W_fc2 = tf$Variable(tf$truncated_normal(shape(1024L, 10L), stddev=0.1))
b_fc2 = tf$Variable(tf$constant(0.1, shape=shape(10L)))
# hypothesis : y_conv = h_fc1_drop * W_fc2 + b_fc2
hypothesis_conv = tf$nn$softmax(tf$matmul(h_fc1_drop, W_fc2) + b_fc2)


# cross_entropy = tf$reduce_mean(-tf$reduce_sum(label * tf$log(y_conv), reduction_indices=1L))
cost = tf$reduce_mean(-tf$reduce_sum(label * tf$log(hypothesis_conv)))
backpropagation = tf$train$AdamOptimizer(1e-4)$minimize(cost)
correct_prediction = tf$equal(tf$argmax(hypothesis_conv, 1L), tf$argmax(label, 1L))
accuracy = tf$reduce_mean(tf$cast(correct_prediction, tf$float32))
sess$run(tf$global_variables_initializer())

for (i in 1:1000) {
  # 학습 데이터 mini_patch
  batch = mnist$train$next_batch(50L)
  if (i %% 100 == 0) {
    train_accuracy = accuracy$eval(feed_dict = dict(
          features = batch[[1]]
        , label = batch[[2]]
        , keep_prob = 1.0))
    # 학습 정확도
    cat(sprintf("step %d, training accuracy %.4f\n", i, train_accuracy))
  }
  backpropagation$run(feed_dict = dict(
    features = batch[[1]], label = batch[[2]], keep_prob = 0.5))
}


# test data를 적용한 정확도
test_accuracy = accuracy$eval(feed_dict = dict(
       features = mnist$test$images
     , label = mnist$test$labels
     , keep_prob = 1.0))
cat(sprintf("test accuracy %g", test_accuracy))


# testData = mnist$test$next_batch(1L)
# testFeatures = testData[[1]]
# testLabel = testData[[2]]

# answer = sess$run(hypothesis_conv, feed_dict = dict( features = testFeatures ))

# print('test label')
# print(testLabel)
# print(c("test of letter : ", which.max(testLabel) - 1))
# print('answer')
# print(answer)
# print(c("answer of letter : ", which.max(answer) - 1))
