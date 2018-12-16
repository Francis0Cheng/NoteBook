我们将使用TensorFlow是实现一个GBDT(Gradient Boosted Decision Tree)去识别手写字图像。我们将MNIST手写字作为训练样本



导入库

```python
import tensorflow as tf
from tensorflow.contrib.boosted_trees.estimator_batch.estimator import GradientBoostedDecisionTreeClassifier
from tensorflow.contrib.boosted_trees.proto import learner_pb2 as gbdt_learner
```



忽略所有GPU，因为当前版本的TF GBDT不支持GPU

```python
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = ""
```



导入MNIST数据集

将设置日志详细程度为仅出现错误时显示，是为了过滤掉Warnings

```python
tf.logging.set_verbosity(tf.logging.ERROR)
```

导入数据集

直接用tensorflow中的 input_data就好了，不过因为数据集在墙外的原因，可能会下载失败，可以手动下载4个压缩包。放到同一目录下的"MNIST_data"文件夹中

```python
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data",one_hot=False)
```



定义一些参数

```
batch_size = 4096   
num_classes = 10    #有0到9十个数字类型
num_features = 784   #每张图片是28*28 像素的
max_steps = 10000
```



定义GBDT的参数

```python
learning_rate = 0.1
l1_regul = 0.
l2_regul = 1.
examples_per_layer = 1000
num_trees = 10
max_depth = 16
```



将GBDT的参数填进配置原型中

```python
learner_config = gbdt_learner.LearnerConfig()
learner_config.learning_rate_tuner.fixed.learning_rate = learning_rate
learner_config.regularization.l1 = l1_regul
learner_config.regularization.l2 = l2_regul / examples_per_layer
learner_config.constraints.max_tree_depth = max_depth
growing_mode = gbdt_learner.LearnerConfig.LAYER_BY_LAYER
learner_config.growing_mode = growing_mode

run_config = tf.contrib.learn.RunConfig(save_checkpoints_secs=300)
learner_config.multi_class_strategy = (
    gbdt_learner.LearnerConfig.DIAGONAL_HESSIAN)

#创建一个TensorFlow GBDT评估器
gbdt_model = GradientBoostedDecisionTreeClassifier(
    model_dir=None, 
    learner_config=learner_config,
    n_classes=num_classes,
    examples_per_layer=examples_per_layer,
    num_trees=num_trees,
    center_bias=False,
    config=run_config)
```



显示TensorFlow 信息日志

```python
tf.logging.set_verbosity(tf.logging.INFO)
```



训练模型**

​	定义训练的输入函数

```python
input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'images': mnist.train.images}, y=mnist.train.labels,
    batch_size=batch_size, num_epochs=None, shuffle=True)
```

​	训练模型

```python
gbdt_model.fit(input_fn=input_fn, max_steps=max_steps)
```



**评估模型**

​	定义评估的输入函数

```python
input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'images': mnist.test.images}, y=mnist.test.labels,
    batch_size=batch_size, shuffle=False)
```

​	使用Estimator的`evaluate`方法

```python
e = gbdt_model.evaluate(input_fn=input_fn)
```

