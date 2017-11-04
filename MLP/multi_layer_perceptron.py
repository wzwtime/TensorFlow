# coding=utf=8
"""TensorFlow使用Dropout、Adagrad、ReLU等辅助组件实现多层感知机"""
# 第一步-定义算法公式即神经网络forward时的计算
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # 隐藏提示警告
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
sess = tf.InteractiveSession()

# 隐含层的参数设置Variable并进行初始化
in_units = 784      # 输入节点数
h1_units = 300      # 隐含层的输出节点数
W1 = tf.Variable(tf.truncated_normal([in_units, h1_units], stddev=0.1))  # 隐含层的权重truncated：截断的正态分布 标差：0.1
b1 = tf.Variable(tf.zeros([h1_units]))          # 隐含层的偏置
W2 = tf.Variable(tf.zeros([h1_units, 10]))
b2 = tf.Variable(tf.zeros([10]))

# 定义输入及Dropout比率的占位符
x = tf.placeholder(tf.float32, [None, in_units])
keep_prob = tf.placeholder(tf.float32)          # Dropout的比率即保留节点的概率

# 定义模型结构
hidden1 = tf.nn.relu(tf.matmul(x, W1) + b1)         # 隐含层relu()激活函数：解决梯度弥散
hidden1_drop = tf.nn.dropout(hidden1, keep_prob)    # dropout(): 减轻过拟合，通过增大样本量、减少特征数量来防止过拟合
y = tf.nn.softmax(tf.matmul(hidden1_drop, W2) + b2)

# 第二步-定义损失函数和选择优化器来优化loss
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.AdagradOptimizer(0.3).minimize(cross_entropy)  # Adagrad:自适应学习速率优化器，希望学习率开始大后期小

# 第四步-对模型进行准确率评测
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 第三步-训练步骤
tf.global_variables_initializer().run()     # 没有.run()
for i in range(3000):
    batch_xs, batch_ys = mnist.train.next_batch(100)        # 没有.next_batch()
    train_step.run({x: batch_xs, y_: batch_ys, keep_prob: 0.75})    # 没有.run()
    if i % 100 == 0:
        print(accuracy.eval({x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))  # 没有.eval()






