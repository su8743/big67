#!/usr/bin/env /c/Apps/Anaconda3/Scripts/Rscript

library(tensorflow)


trainFile = 'c:/Home/tfR/data/in/economic-level-train.csv'
trainData = read.csv(trainFile, header = F)[, 1:5]
testFile = 'c:/Home/tfR/data/in/economic-level-test.csv'
testData = read.csv(testFile, header = F)[, 1:5]

num_rows = nrow(trainData)
print(num_rows)

# print(cbind(nrow(trainData),ncol(trainData)))
# print(cbind(nrow(testData),ncol(testData)))

# lstm parameter
inputDim = as.integer(1L)
seqLength = as.integer(5L)
hiddenDim = as.integer(100L)
# print(sprintf('dim %f', hiddenDim))

# trainData의 feature와 label 할당
features = as.matrix(trainData[,2:5])
features = array(c(features),dim=c(138,4))
print(features)
# print(features)
label_ = as.matrix(trainData[,1])
# print(label)
print(label_[138])

# one_hot = function(x) {
# if(x==1){array(c(1,0,0),dim=c(1,3))}
# else if(x==2){array(c(0,1,0),dim=c(1,3))}
# else {array(c(0,0,1),dim=c(1,3))}
# }

one_hot = function(x) {
if(x==1){c(1,0,0)}
else if(x==2){c(0,1,0)}
else {c(0,0,1)}
}


one_hot_label = c()
for (i in 1:138){one_hot_label = c(one_hot_label,one_hot(label_[i,]))}
label = aperm(array(one_hot_label,dim=c(3,138)))
print(label)

nFeature = 4
nLabel = 3

learningRate = 0.2
nHidden01Neuron = 10L
nLoop = 50

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

backpropagation = tf$train$GradientDescentOptimizer(learningRate)$minimize(cost)
# backpropagation = tf$train$AdamOptimizer(learningRate)$minimize(cost)

correct_predLabel = tf$equal(tf$argmax(hypothesis, 1L), tf$argmax(plLabel, 1L))
accuracy = tf$reduce_mean(tf$cast(correct_predLabel, tf$float32))

init = tf$global_variables_initializer()

with(tf$Session() %as% sess, {

    sess$run(init)

    for (i in 1:nLoop) {

		sess$run(backpropagation, feed_dict = dict(plFeatures = features, plLabel = label))        

		current_cost = sess$run(cost, feed_dict = dict(plFeatures = features, plLabel = label))


        cat(sprintf('i=%d : cost=%f accuracy=%f\n'
            ,i
            ,current_cost
            ,sess$run(accuracy, feed_dict=dict(plFeatures = features, plLabel = label))
            ))
    }

# 3	1.501954908	0.51776495	1.273780419	1.601626357
input_x = c(1.501954908,-0.20710598,0.970499367,1.350625211)
input_x = array(c(input_x),dim=c(1,4))
result = sess$run(hypothesis, feed_dict=dict(plFeatures = input_x))
cat(sprintf('%f, %f, %f, %f, %f, %f, %f\n', input_x[1], input_x[2], input_x[3], input_x[4], result[1], result[2], result[3]))
})


# 1	-0.969003166	1.001012236	-1.213124209	-0.782884525
# 1	-0.823652691	1.725883166	-1.031155577	-1.033885671
# 1	-1.259704116	-0.20710598	-1.33443663	-1.159386243
# 1	-0.823652691	1.725883166	-1.213124209	-1.284886816
# 1	-1.550405066	0.276141307	-1.33443663	-1.284886816
# 1	-0.532951741	1.484259523	-1.273780419	-1.284886816
# 1	-0.969003166	0.51776495	-1.33443663	-1.284886816
# 2	-0.969003166	-1.898471483	-0.181968631	-0.280882234
# 2	-0.096900317	-0.93197691	0.363937263	0.095619484
# 2	0.048450158	-0.20710598	0.363937263	-0.029881089
# 2	0.048450158	-0.448729623	0.363937263	0.095619484
# 2	0.775202533	-0.448729623	0.424593473	0.095619484
# 2	-0.823652691	-1.415224196	-0.363937263	-0.155381661
# 2	0.048450158	-0.690353266	0.303281052	0.095619484
# 3	1.501954908	0.51776495	1.273780419	1.601626357
# 3	1.501954908	-0.20710598	0.970499367	1.350625211
# 3	0.920553008	-1.415224196	0.849186946	0.84862292
# 3	1.211253958	-0.20710598	0.970499367	0.974123493
# 3	0.775202533	0.759388593	1.091811788	1.350625211
# 3	0.339151108	-0.20710598	0.909843157	0.723122348
# 3	1.647305383	0.276141307	1.39509284	1.350625211




