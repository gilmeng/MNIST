# -*- coding: utf-8 -*-
"""
Created on Tue Jun 10 10:34:01 2018

@author: Choi Yoo jung , 12141635, Computer Engineering Dept.
"""
import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import time
#Visualize Sample Test Option
Visualize_sample_test = True
## Hyperparameters
hidden_layers = 1;  #은닉 계층 수
hidden_layer_nodes = 200  #은닉 계층의 뉴런 수
optimizer = 1 #0: Gradient Descent, 1: Adam Optimizer
learning_rate = 0.01 #학습률
batch_size = 500 #학습 샘플 사이즈
iter_count = 4000   #반복 학습 수
dropout = 0.8   # Dropout 비율

# 그래프에 출력될 하이퍼파라미터 변수 스트링
hyperparam = "Learn rate = %6.3f\nBatch size = %d\nInterations = %d\nDropout = %5.2f\nHidden Nodes = %d\nOptimizer = %s" % (learning_rate, batch_size, iter_count, dropout, hidden_layer_nodes,  "SGD" if optimizer==0 else "Adam" )

# Load and prepare data by the following 2 lines of code (a piece of cake!)
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

seed = 3
tf.set_random_seed(seed)

# Declare the placeholder for the data (i.e., images) and target (i.e., 10-length vector)
x_data = tf.placeholder(shape=[None, 784], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 10], dtype=tf.float32)

#Dropout 비율을 입력할 Placeholder 선언
keep_prob = tf.placeholder(tf.float32)

#가중치 초기값을 설정하기위한 he initializer선언
he_initializer = tf.contrib.keras.initializers.he_normal() 

# 다중계층 신경망을 선언한다.

if hidden_layers == 1:  #3-Layer NN
#    with tf.variable_scope("2NN", reuse = tf.AUTO_REUSE ):
    W1 = tf.get_variable("W1", shape=[784, hidden_layer_nodes], initializer=he_initializer) 
    b1 = tf.Variable(tf.random_normal(shape=[hidden_layer_nodes]))
    W2 = tf.get_variable("W2", shape=[hidden_layer_nodes,10], initializer=he_initializer) 
    b2 = tf.Variable(tf.random_normal(shape=[10]))
    hidden_output = tf.nn.relu(tf.add(tf.matmul(x_data, W1), b1))
    hidden_output = tf.nn.dropout(hidden_output, keep_prob)
    final_output = tf.add(tf.matmul(hidden_output, W2), b2)
    
elif hidden_layers == 2: #4-Layer NN
#    with tf.variable_scope("4NN", reuse = tf.AUTO_REUSE ):
    W1 = tf.get_variable("W1", shape=[784, hidden_layer_nodes], initializer=he_initializer) 
    b1 = tf.Variable(tf.random_normal(shape=[hidden_layer_nodes]))
    W2 = tf.get_variable("W2", shape=[hidden_layer_nodes, hidden_layer_nodes], initializer=he_initializer) 
    b2 = tf.Variable(tf.random_normal(shape=[hidden_layer_nodes]))
    W3 = tf.get_variable("W3", shape=[hidden_layer_nodes, 10], initializer=he_initializer) 
    b3 = tf.Variable(tf.random_normal(shape=[10]))
    hidden_output_1 = tf.nn.relu(tf.add(tf.matmul(x_data, W1), b1))
    hidden_output_1 = tf.nn.dropout(hidden_output_1, keep_prob)
    hidden_output_2 = tf.nn.relu(tf.add(tf.matmul(hidden_output_1, W2), b2))
    hidden_output_2 = tf.nn.dropout(hidden_output_2, keep_prob)
    final_output = tf.add(tf.matmul(hidden_output_2, W3), b3) 
else:   #1-Layer NN (은닉계층 없음)
#    with tf.variable_scope("1NN", reuse = tf.AUTO_REUSE ):
    W = tf.get_variable("W", shape=[784, 10], initializer=he_initializer) 
    b = tf.Variable(tf.random_normal(shape=[10]))
    final_output = tf.add(tf.matmul(x_data, W), b)

# 손실함수(Loss function) : Softmax activation 함수 출력과 Cross entropy Cost Fuction 사용
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_target, logits=final_output))

# 최적화 알고리즘 선언 (Gradient Descent 혹은 Adam)
if optimizer == 0:
    opt = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
else:
    opt = tf.train.AdamOptimizer(learning_rate=learning_rate)

train_step = opt.minimize(loss)

#정확도(Accuracy) 계산을 위한 Computation Graph 선언
correct_prediction = tf.equal(tf.argmax(y_target, 1), tf.argmax(final_output, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init = tf.global_variables_initializer()
sess = tf.Session()

#learning_rates = [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]


sess.run(init)

# 학습결과를 저장할 배열들
time_trains = []    #학습에 걸린 시간들을 저장하는 배열
losses = []         #학습 오차값을 저장하는 배열
train_accuracies = []   #학습 정확도를 저장하는 배열
test_accuracies = []    #시험 정확도를 저장하는 배열

# Training iterations
for i in range(iter_count):
    # run the training step
    rand_x, rand_y = mnist.train.next_batch(batch_size)
    
    start_time = time.time()    #시작시간 기록
    _, temp_loss = sess.run([train_step, loss], feed_dict={x_data: rand_x, y_target: rand_y, keep_prob: dropout})
    end_time = time.time()      #끝시간 기록
    
    time_train = end_time - start_time  #train_step에 걸린 시간 계산
    time_trains.append(time_train)      #train_step에 걸린 시간 배열에 입력
    
    losses.append(temp_loss)            #오차 배열에 입력

    train_accuracy = sess.run(accuracy, feed_dict={x_data: rand_x, y_target: rand_y, keep_prob: dropout})
    train_accuracies.append(train_accuracy)

    test_accuracy = sess.run(accuracy, feed_dict={x_data: mnist.test.images, y_target: mnist.test.labels, keep_prob: 1.0})
    test_accuracies.append(test_accuracy)

    
    if (i+1) % 100 == 0:
        print("Iteration: %d. Train accuracy = %f. Test accuracy = %f.  loss = %f. Train time = %f" %(i+1, train_accuracy, test_accuracy, temp_loss, time_train))
   

print("Max Train accuracy = %f, Max Test accuracy = %f Time to train = %f" %(max(train_accuracies), max(test_accuracies), sum(time_trains)))
# 그래프에 표시할 결과값 스트링 
result = "Max Train acc. = %6.3f\nMax Test acc. = %6.3f\nTime to train = %7.4f" %(max(train_accuracies), max(test_accuracies), sum(time_trains))
result_loss = "Min Train loss = %6.3f\nTime to train = %7.4f" %(min(losses), sum(time_trains))

f, (ax1, ax2) = plt.subplots(1, 2)
f.set_size_inches(14,4)

# 인식 정확도 그래프
ax1.plot(train_accuracies, 'k-', label='Train Accuracy')
ax1.plot(test_accuracies, 'r--', label='Test Accuracy')
ax1.set_title('Match Accuracy')
ax1.set_xlabel('Iteration')
ax1.set_ylabel('Accuracy')
ax1.legend(loc='lower right')
ax1.text(0.67,0.23, hyperparam, transform=ax1.transAxes, bbox=dict(facecolor='gray', alpha=0.1)) #하이퍼파라미터 출력
ax1.text(0.30,0.1, result, transform=ax1.transAxes) #결과값을 그래프에 출력

#손실오차 그래프
ax2.plot(losses, 'k-', label='Train loss')
ax2.set_title('Loss ')
ax2.set_xlabel('Iteration')
ax2.set_ylabel('loss')
ax2.legend(loc='upper right')
ax2.text(0.67,0.55, hyperparam, transform=ax2.transAxes, bbox=dict(facecolor='gray', alpha=0.1)) #하이퍼파라미터 출력
ax2.text(0.30,0.8, result_loss, transform=ax2.transAxes) #결과값 출력

# 예측 결과 예시 화면 출력
if Visualize_sample_test == True:
    labels = sess.run(final_output, feed_dict={x_data: mnist.test.images, y_target: mnist.test.labels, keep_prob: 1.0})
    fig = plt.figure()
    for i in range(10):
        rnd = random.randrange(1,1000)
        subplot = fig.add_subplot(2,5,i+1)
        subplot.set_xticks([])
        subplot.set_yticks([])
        #title 출력 : 예측된 레이블값
        subplot.set_title('%d' % np.argmax(labels[rnd]))
        #이미지 출력
        subplot.imshow(mnist.test.images[rnd].reshape((28,28)),cmap=plt.cm.gray_r)
    plt.show()
    
#Tensorflow Graph and Variable 초기화
tf.reset_default_graph()