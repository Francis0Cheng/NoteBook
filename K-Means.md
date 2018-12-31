  本小节中，我们将使用TensorFlow实现K-Means算法， 并利用它对手写字图像进行分类。这个例子将使用MNIST数据集。



1.导入库

```python
import numpy as np
import tensorflow as tf
from tensorflow.contrib.factorization import KMeans
```



2.忽略所有GPU, tensorflow的随机森林不会从中受益

```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
```



3.导入MNIST数据集

```python
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data",one_hot=True)
full_data_x = mnist.train.images
```



4.参数

```python
num_steps = 50  #步长
batch_size = 1024  #每次batch样本的数量
k = 25  #聚类中心数目
num_classes = 10  #MNIST数据集10种数字 0 到 9 
num_features = 784  #每幅图片的大小是 28x28 像素点
```



5.图片和标签的占位符(placeholder)

```python
#图片
X = tf.placeholder(tf.float32, shape=[None, num_features])
#标签
Y = tf.placeholder(tf.float32, shape=[None, num_classes])
```



6.K-Mean的一些参数

```python
kmeans = KMeans(inputs=X, num_clusters=k, distance_metric='cosine', use_mini_batch=True)
```



7.创建KMeans图

```python
training_graph = kmeans.training_graph()
(all_scores, cluster_idx, scores, cluster_centers_initialized, init_op, train_op) = training_graph
```

```
cluster_idx = cluster_idx[0]  #cluster_idx是一个只有一个元素的元组，这里将其取出
avg_distance = tf.reduce_mean(scores)	#平均距离
```



8.TensorFlow初始化, 新建一个会话

```python
init = tf.global_variables_initializer()
sess = tf.Session()
```



9.初始化

```python
sess.run(init)
sess.run(init_op, feed_dict={X:full_data_x})
```



10.训练

```
for i in range(1, num_steps + 1):
	_, d, idx = sess.run([train_op, avg_distance, cluster_idx], feed_dict={X:full_data_x})
	if i % 10 ==0 or i == 1:
		print("step %i, 平均距离: %g" % (i, d))
```



11.为每个中心分配一个标签。用每个训练样本的标签到最近的质心（由'idx'给出）计算每个质心的标签总数，

```python
counts = np.zeros(shape=(k, num_classes))
for i in range(len(idx)):
    counts[idx[i]] += mnist.train.labels[i]
```

```python
labels_map = [np.argmax(c) for c in counts]
labels_map = tf.convert_to_tensor(labels_map)
```



13.将最多出现的标签分配给中心

```python
cluster_label = tf.nn.embedding_lookup(labels_map, cluster_idx)
```



14.选取索引

```python
correct_prediction = tf.equal(cluster_label, tf.cast(tf.argmax(Y, 1), tf.int32))
accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
```



15.测试模型

```python
test_x, test_y = mnist.test.images, mnist.test.labels
print("Test Accuracy:", sess.run(accuracy_op, feed_dict={X: test_x, Y: test_y}))
```

