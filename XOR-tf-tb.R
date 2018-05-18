#!/usr/bin/env /c/Apps/Anaconda3/Scripts/Rscript

# XOR function
# Author : Yibeck Lee(yibeck.lee@gmail.com)

library(tensorflow)

input_x_ = c(0,1,0,1,0,0,1,1)
input_x = array(c(input_x_),dim=c(4,2))
print(input_x)
input_label_ = c(1,0,0,1,0,1,1,0)
input_label = array(c(input_label_),dim=c(4,2))
print(input_label)

learning_rate = 0.2
n_hidden_1 = 10L
# n_hidden_2 = 10L
nloop = 50

x = tf$placeholder(tf$float32, shape(NULL, 2), name="input")
label = tf$placeholder(tf$float32, shape(NULL, 2), name = "label")

W_h1 = tf$Variable(tf$random_normal(shape(2, n_hidden_1)))
b1 = tf$Variable(tf$random_normal(shape(n_hidden_1)))
layer_1 = tf$add(tf$matmul(x, W_h1), b1)
layer_1 = tf$nn$relu(layer_1)

# W_h2 = tf$Variable(tf$random_normal(shape(n_hidden_1, n_hidden_2)))
# b2 = tf$Variable(tf$random_normal(shape(n_hidden_2)))
# layer_2 = tf$add(tf$matmul(layer_1, W_h2), b2)
# layer_2 = tf$nn$relu(layer_2)

# W_out = tf$Variable(tf$zeros(shape(n_hidden_2, n_classes)))

W_out = tf$Variable(tf$zeros(shape(n_hidden_1, 2)))
b_out = tf$Variable(tf$zeros(shape(2)))
model = tf$add(tf$matmul(layer_1, W_out), b_out)
model = tf$nn$softmax(model)


cost = tf$reduce_mean(-tf$reduce_sum(label * tf$log(model), reduction_indices=1L))

# with(tf$name_scope("cross_entropy"), {
#   diff <- label * tf$log(model)
#   with(tf$name_scope("total"), {
#     cost <- -tf$reduce_mean(diff)
#   })
#   tf$summary$scalar("cross entropy", cost)
# })



optimizer = tf$train$GradientDescentOptimizer(learning_rate)$minimize(cost)
# optimizer = tf$train$AdamOptimizer(learning_rate)$minimize(cost)

correct_prediction = tf$equal(tf$argmax(model, 1L), tf$argmax(label, 1L))
accuracy = tf$reduce_mean(tf$cast(correct_prediction, tf$float32))



# sess = tf$Session()
# sess$run(tf$global_variables_initializer())

# for (i in 1:nloop) {
#     # merged = tf$merge_all_summaries()
#     # writer = tf$train$SummaryWriter("./tb/xor", sess$graph)
#     opt = sess$run(optimizer,
#              feed_dict = dict(x = input_x, label = input_label))
#     # correct_prediction = tf$equal(tf$argmax(model, 1L), tf$argmax(label, 1L))
#     # accuracy = tf$reduce_mean(tf$cast(correct_prediction, tf$float32))
#     current_cost = sess$run(cost, feed_dict = dict(x = input_x, label = input_label))



#     # cat(sprintf('i=%d : Weight=%f bias=%f cost=%f accuracy=%f\n'
#     # 	,i
#     # 	,sess$run(W_out)
#     # 	,sess$run(b_out) 
#     # 	,current_cost
#     # 	,sess$run(accuracy, feed_dict=dict(x = input_x, label = input_label))
#     # 	))

#     cat(sprintf('i=%d : cost=%f accuracy=%f\n'
#     	,i
#     	,current_cost
#     	,sess$run(accuracy, feed_dict=dict(x = input_x, label = input_label))
#     	))
# }
# # correct_prediction = tf$equal(tf$argmax(model, 1L), tf$argmax(label, 1L))
# # accuracy = tf$reduce_mean(tf$cast(correct_prediction, tf$float32))

# input_x_ = c(0,0)
# input_x = array(c(input_x_),dim=c(1,2))
# result = sess$run(model, feed_dict=dict(x = input_x))
# cat(sprintf('%d, %d, %f %f\n', input_x[1], input_x[2], result[1], result[2]))

# input_x_ = c(0,1)
# input_x = array(c(input_x_),dim=c(1,2))
# result = sess$run(model, feed_dict=dict(x = input_x))
# cat(sprintf('%d, %d, %f %f\n', input_x[1], input_x[2], result[1], result[2]))


# input_x_ = c(1,0)
# input_x = array(c(input_x_),dim=c(1,2))
# result = sess$run(model, feed_dict=dict(x = input_x))
# cat(sprintf('%d, %d, %f %f\n', input_x[1], input_x[2], result[1], result[2]))

# input_x_ = c(1,1)
# input_x = array(c(input_x_),dim=c(1,2))
# result = sess$run(model, feed_dict=dict(x = input_x))
# cat(sprintf('%d, %d, %f %f\n', input_x[1], input_x[2], result[1], result[2]))


with(tf$Session() %as% sess, {
    tf$get_variable_scope()$reuse_variables()

    sess$run(tf$global_variables_initializer())
    tblog = tf$summary$FileWriter("./tb/XOR", sess$graph)
    for (i in 1:nloop) {

        opt = sess$run(optimizer,
                 feed_dict = dict(x = input_x, label = input_label))
        print(opt)
        correct_prediction = tf$equal(tf$argmax(model, 1L), tf$argmax(label, 1L))
        accuracy = tf$reduce_mean(tf$cast(correct_prediction, tf$float32))
        current_cost = sess$run(cost, feed_dict = dict(x = input_x, label = input_label))



        # cat(sprintf('i=%d : Weight=%f bias=%f cost=%f accuracy=%f\n'
        #   ,i
        #   ,sess$run(W_out)
        #   ,sess$run(b_out) 
        #   ,current_cost
        #   ,sess$run(accuracy, feed_dict=dict(x = input_x, label = input_label))
        #   ))

        cat(sprintf('i=%d : cost=%f accuracy=%f\n'
            ,i
            ,current_cost
            ,sess$run(accuracy, feed_dict=dict(x = input_x, label = input_label))
            ))
    }

input_x_ = c(0,0)
input_x = array(c(input_x_),dim=c(1,2))
result = sess$run(model, feed_dict=dict(x = input_x))
cat(sprintf('%d, %d, %f %f\n', input_x[1], input_x[2], result[1], result[2]))

input_x_ = c(0,1)
input_x = array(c(input_x_),dim=c(1,2))
result = sess$run(model, feed_dict=dict(x = input_x))
cat(sprintf('%d, %d, %f %f\n', input_x[1], input_x[2], result[1], result[2]))

input_x_ = c(1,0)
input_x = array(c(input_x_),dim=c(1,2))
result = sess$run(model, feed_dict=dict(x = input_x))
cat(sprintf('%d, %d, %f %f\n', input_x[1], input_x[2], result[1], result[2]))

input_x_ = c(1,1)
input_x = array(c(input_x_),dim=c(1,2))
result = sess$run(model, feed_dict=dict(x = input_x))
cat(sprintf('%d, %d, %f %f\n', input_x[1], input_x[2], result[1], result[2]))

})
