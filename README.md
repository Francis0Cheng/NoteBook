# TensorFlow Examples

本教程旨在通过示例轻松深入了解TensorFlow。 为了便于阅读，它包括笔记本和源代码以及解释。

它适合想要找到关于TensorFlow的清晰简洁示例的初学者。 除了传统的“原始”TensorFlow实现，您还可以找到最新的TensorFlow API实践（例如`layers`,`estimator`,`datasets`)

**联系译者**：francis000cheng@gmail.com

[译者博客](https://francis0cheng.github.io/)

[译者Bilibili](http://space.bilibili.com/150239294?)

## 教程目录

#### 0 - Prerequisite

- [机器学习简介](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/notebooks/0_Prerequisite/ml_introduction.ipynb).
- [MNIST数据集介绍](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/notebooks/0_Prerequisite/mnist_dataset_intro.ipynb).

#### 1 -简介

- **Hello World** ([notebook](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/notebooks/1_Introduction/helloworld.ipynb)) ([code](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/1_Introduction/helloworld.py)). 非常简单的例子，学习如何使用TensorFlow打印“Hello World”
- **基础运算** ([notebook](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/notebooks/1_Introduction/basic_operations.ipynb)) ([code](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/1_Introduction/basic_operations.py)). 一个涵盖TensorFlow基本操作的简单示例。
- **TensorFlow Eager API 基础知识** ([notebook](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/notebooks/1_Introduction/basic_eager_api.ipynb)) ([code](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/1_Introduction/basic_eager_api.py)). 初识TensorFlow 的 Eager API.

#### 2 - 基础模型

- **线性回归** ([notebook](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/notebooks/2_BasicModels/linear_regression.ipynb)) ([code](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/2_BasicModels/linear_regression.py)).使用TensorFlow实现线性回归。
- **线性回归 (eager api)** ([notebook](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/notebooks/2_BasicModels/linear_regression_eager_api.ipynb)) ([code](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/2_BasicModels/linear_regression_eager_api.py)).使用TensorFlow的Eager API实现线性回归。
- **Logistic回归** ([notebook](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/notebooks/2_BasicModels/logistic_regression.ipynb)) ([code](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/2_BasicModels/logistic_regression.py)). 。使用TensorFlow实现Logistic回归。
- **Logistic 回归 (eager api)** ([notebook](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/notebooks/2_BasicModels/logistic_regression_eager_api.ipynb)) ([code](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/2_BasicModels/logistic_regression_eager_api.py)). Implement a Logistic Regression using TensorFlow's Eager API.
- **邻近算法** ([notebook](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/notebooks/2_BasicModels/nearest_neighbor.ipynb)) ([code](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/2_BasicModels/nearest_neighbor.py)). 使用TensorFlow实现最近邻算法。
- **K-Means算法** ([notebook](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/notebooks/2_BasicModels/kmeans.ipynb)) ([code](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/2_BasicModels/kmeans.py)). 使用TensorFlow构建随机森林分类器。
- **R随机森林算法** ([notebook](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/notebooks/2_BasicModels/random_forest.ipynb)) ([code](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/2_BasicModels/random_forest.py)). 使用TensorFlow构建随机森林分类器。
- **Gradient Boosted Decision Tree (GBDT)** ([notebook](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/notebooks/2_BasicModels/gradient_boosted_decision_tree.ipynb)) ([code](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/2_BasicModels/gradient_boosted_decision_tree.py)).使用TensorFlow构建GBDT
- **Word2Vec (Word Embedding词嵌入)** ([notebook](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/notebooks/2_BasicModels/word2vec.ipynb)) ([code](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/2_BasicModels/word2vec.py)). 使用TensorFlow从Wikipedia数据构建Word Embedding模型（Word2Vec）

#### 3 - 神经网络

##### 监督式

- **简单神经网络** ([notebook](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/notebooks/3_NeuralNetworks/neural_network_raw.ipynb)) ([code](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/neural_network_raw.py)). 构建一个简单的神经网络（a.k.a多层感知器）来对MNIST数字数据集进行分类。
- **简单神经网络（tf.layers / estimator api）** ([notebook](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/notebooks/3_NeuralNetworks/neural_network.ipynb)) ([code](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/neural_network.py)). 使用TensorFlow `layers`和`estimator`API构建一个简单的神经网络
- **简单神经网络（eager api）** ([notebook](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/notebooks/3_NeuralNetworks/neural_network_eager_api.ipynb)) ([code](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/neural_network_eager_api.py)). 使用TensorFlow Eager API构建一个简单的神经网络（a.k.a多层感知器）来对MNIST数字数据集进行分类。
- **卷积神经网络** ([notebook](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/notebooks/3_NeuralNetworks/convolutional_network_raw.ipynb)) ([code](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/convolutional_network_raw.py)). 构建卷积神经网络以对MNIST数字数据集进行分类。(低层TensorFlow API实现)
- **卷积神经网络（tf.layers / estimator api）** ([notebook](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/notebooks/3_NeuralNetworks/convolutional_network.ipynb)) ([code](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/convolutional_network.py)). 使用TensorFlow`layers`和`estimator`API构建卷积神经网络，对MNIST数字数据集进行分类。
- **回归神经网络（LSTM）** ([notebook](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/notebooks/3_NeuralNetworks/recurrent_network.ipynb)) ([code](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/recurrent_network.py)).构建递归神经网络（LSTM）以对MNIST数字数据集进行分类。
- **双向递归神经网络（LSTM）** ([notebook](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/notebooks/3_NeuralNetworks/bidirectional_rnn.ipynb)) ([code](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/bidirectional_rnn.py)). 构建双向递归神经网络（LSTM）以对MNIST数字数据集进行分类。
- **动态递归神经网络（LSTM）** ([notebook](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/notebooks/3_NeuralNetworks/dynamic_rnn.ipynb)) ([code](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/dynamic_rnn.py)).构建一个递归神经网络（LSTM），执行动态计算以对不同长度的序列进行分类。

##### 无监督式

- **自动编码器** ([notebook](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/notebooks/3_NeuralNetworks/autoencoder.ipynb)) ([code](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/autoencoder.py)). 构建自动编码器以将图像编码为较低维度并重新构建它。
- **变分自编码器** ([notebook](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/notebooks/3_NeuralNetworks/variational_autoencoder.ipynb)) ([code](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/variational_autoencoder.py)). 构建变分自动编码器（VAE），对噪声进行编码和生成图像。
- **GAN（Generative Adversarial Networks对抗网络）** ([notebook](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/notebooks/3_NeuralNetworks/gan.ipynb)) ([code](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/gan.py)). 构建生成对抗网络（GAN）以从噪声中生成图像。
- **DCGAN (Deep Convolutional Generative Adversarial Networks)** ([notebook](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/notebooks/3_NeuralNetworks/dcgan.ipynb)) ([code](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/dcgan.py)). 构建深度卷积生成对抗网络（DCGAN）以从噪声中生成图像。

#### 4 - 实用工具

- **保存并恢复模型** ([notebook](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/notebooks/4_Utils/save_restore_model.ipynb)) ([code](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/4_Utils/save_restore_model.py)). 使用TensorFlow保存和恢复模型。
- **Tensorboard - Graph（图） 和 损失可视化** ([notebook](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/notebooks/4_Utils/tensorboard_basic.ipynb)) ([code](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/4_Utils/tensorboard_basic.py)).使用Tensorboard可视化计算图并绘制损失。
- **Tensorboard - 高级可视化** ([notebook](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/notebooks/4_Utils/tensorboard_advanced.ipynb)) ([code](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/4_Utils/tensorboard_advanced.py)). 深入了解Tensorboard;可视化变量，渐变等......

#### 5 - 数据管理

- **构建图像数据集** ([notebook](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/notebooks/5_DataManagement/build_an_image_dataset.ipynb)) ([code](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/5_DataManagement/build_an_image_dataset.py)). 使用TensorFlow数据队列，从图像文件夹或数据集文件构建您自己的图像数据集。
- **TensorFlow数据集API** ([notebook](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/notebooks/5_DataManagement/tensorflow_dataset_api.ipynb)) ([code](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/5_DataManagement/tensorflow_dataset_api.py)). I引入TensorFlow数据集API以优化输入数据管道。

#### 6 - 多GPU运算

- **多GPU的基本操作**([notebook](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/notebooks/6_MultiGPU/multigpu_basics.ipynb)) ([code](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/6_MultiGPU/multigpu_basics.py)). 在TensorFlow中引入多GPU的简单示例。
- **在多个GPU上训练神经网络** ([notebook](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/notebooks/6_MultiGPU/multigpu_cnn.ipynb)) ([code](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/6_MultiGPU/multigpu_cnn.py)). 一个清晰简单的TensorFlow实现，用于在多个GPU上训练卷积神经网络。

## 数据集

一些示例需要MNIST数据集进行训练和测试。不用担心，运行示例时会自动下载此数据集。
MNIST是手写数字的数据库，为了快速描述该数据集，您可以查看这个jupyter notebook。 [this notebook](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/notebooks/0_Prerequisite/mnist_dataset_intro.ipynb).

官方网站: [http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/)

## 安装

要下载所有示例，只需要克隆这个库:

```
git clone https://github.com/aymericdamien/TensorFlow-Examples
```

要运行它们，您还需要最新版本的TensorFlow。安装：

```
pip install tensorflow
```

TensorFlow GPU版本

```
pip install tensorflow_gpu
```

有关TensorFlow安装的更多详细信息，可以查看TensorFlow安装指南 [TensorFlow 安装指南](https://www.tensorflow.org/install/)

## 更多例子

以下示例来自 [TFLearn](https://github.com/tflearn/tflearn), 一个为TensorFlow提供简化界面的库。. 这里有很多[例子](https://github.com/tflearn/tflearn/tree/master/examples) and [预建的操作和网络层](http://tflearn.org/doc_index/#api).

### Tutorials

- [TFLearn Quickstart](https://github.com/tflearn/tflearn/blob/master/tutorials/intro/quickstart.md). 通过具体的机器学习任务了解TFLearn的基础知识。构建并训练深度神经网络分类器。

### Examples

- [TFLearn Examples](https://github.com/tflearn/tflearn/blob/master/examples). 使用TFLearn的大量示例集合。