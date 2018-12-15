TensorFlow的Eager API基础介绍

  Eager：急切的、渴望的；很多文章翻译为：动态图，Eager execution（立即执行）引申为动态图。

  Eager使得Tensorflow可以立刻执行运算：并返回具体值，替换（之前）建立计算图（它）过一会才执行。动态图）是一个命令式的编程环境，不建立图而是立即运算求值：运算返回具体值替换（以前）先构建运算图然后执行的机制。使得（使用）Tensorflow和调试模型变得简单，而且减少了多余（模板化、公式化操作）



动态图是一个灵活的机器学习平台，用于研究和实验，提供以下功能：

- *An intuitive interface*—Structure your code naturally and use Python data structures. Quickly iterate on small models and small data.
- 直观的接口-方便编码使用使用，基于Python数据结构。快速迭代小模型和小数据
- *Easier debugging*—Call ops directly to inspect running models and test changes. Use standard Python debugging tools for immediate error reporting.
- 调试简单-直接调用ops来检查运行模型和测试变更。使用标准Python调试工具进行即时错误报告。
- *Natural control flow*—Use Python control flow instead of graph control flow, simplifying the specification of dynamic models.
- 自然控制流-使用Python控制流替换图控制流，简化动态模型规范。



动态图支持大部分Tensorflow运算和GPU（图像设备接口，显卡）加速，运行动态库实例见：



导入库

```python
import numpy as np
import tensorflow as tf
```



设置Eager API

```python
print('正在设置Eager模式')
tf.enable_eager_execution()
tfe = tf.contrib.eager
```

定义常张量

```python
print('定义常张量')
a = tf.constant(2)
print('a = %i' % a)
b = tf.constant(3)
print('b = %i' % b) 
```



在不需要打开一个tf.Session()的情况下运行计算

```pypthon
print('在不需要tf.Session()的情况下运行计算')
c = a + b
print('a + b = %i' % c)
d = a * b
print('a * b = %i' % d)
```



与Numpy全兼容

定义常张量

```python
print('混合TensorFlow Tensor和Numpy向量的运算')

a = tf.constant([[2., 1,],
				 [1., 0.]], dtype=tf.float32)
print('Tensor:\n a = %s' % a)
b = np.array([[3., 0.],
    			[5., 1.]], dtype=np.float32)
print('NumpyArray:\n b = %s' % b)
```

在不需要tf.Session()的情况下运行计算

```python
c = a + b
print('a + b = %s' % c)
d = tf.matmul(a, b)
print('a * b = %s' % d)
```



迭代张量a

```python
print('迭代a张量')
for i in range(a.shape[0]):
	for j in range(a.shape[1]):
		print(a[i][j])

```

