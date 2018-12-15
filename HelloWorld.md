一个简单的Hello World “TensorFlow”程序

1.导入库

```python
import tensorflow as tf
```

2.创建一个常量

```python
hello = tf.constant('Hello, TensorFlow')
```

3.新建一个Session（会话）

```python
sess = tf.Session()
```

4.运行计算

```
print(sess.run(hello))
```

