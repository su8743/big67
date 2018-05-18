#!/usr/bin/env /c/Apps/Anaconda3/Scripts/Rscript

# Regression
# modified
# Author : Yibeck Lee(yibeck.lee@gmail.com)
# references
# 1) https://tensorflow.rstudio.com/tensorflow/
# install.packages("tensorflow")

library(tensorflow)

# p = 3L # x1,x2,x3
nFeature = 1L # x1
features = tf$placeholder("float", shape = shape(NULL, nFeature), name = "input")
label = tf$placeholder("float", name = "label")


W = tf$Variable(tf$zeros(list(nFeature, 1L)))
b = tf$Variable(tf$zeros(list(1L)))
model = tf$add(tf$matmul(features, W), b)
cost = tf$reduce_mean(tf$square(model - label))
optimizer = tf$train$GradientDescentOptimizer(learning_rate = 0.01)$minimize(cost)

sess = tf$Session()
sess$run(tf$global_variables_initializer())

# Generate some data. The 'true' model will be 'y = 2x + 1';
set.seed(123)
n = 250
trainFeatures = matrix(runif(nFeature * n), nrow = n)
trainLabel = matrix(2 * trainFeatures[, 1] + 1 + (rnorm(n, sd = 0.25)))
plot(trainFeatures,trainLabel)
epochs = 1
num_mini_batch=10

for (i in 1:3000) {
    sess$run(optimizer, feed_dict = dict(features = trainFeatures, label = trainLabel))
	current_cost = sess$run(cost, feed_dict = dict(features = trainFeatures, label = trainLabel))
	if (i%%50 == 0){
		cat(sprintf('iteration#%d : cost=%.4f weight=%.4f bias=%.4f\n',i,current_cost,sess$run(W),sess$run(b)))
	}
}

Weight = sess$run(W)
bias = sess$run(b)
print(c('Weight',Weight[1]))
print(c('bias',bias))
print(trainFeatures[1,])
print(trainFeatures[2,])

ols = lm(trainLabel ~ trainFeatures)
print(ols$coefficients[1])
print(ols$coefficients[2])

plot(trainFeatures,trainLabel)
regression_line = trainFeatures*Weight[1] + bias[1]
lines(trainFeatures,regression_line)
lines(trainFeatures,ols$coefficients[2]*trainFeatures+ols$coefficients[1])

# testFeatures = trainFeatures
testFeatures = matrix(c(0.28,1,10,0,0.01), nrow=5)
estimatedLabel = sess$run(model, feed_dict = dict(features = testFeatures))
print(cbind(testFeatures, estimatedLabel))

