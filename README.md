
![](https://img.shields.io/badge/licence-Apache2.0-ff69b4.svg)
![](https://img.shields.io/badge/build-pass-0f9d58.svg)

## 运行环境


1. Python 3.x.x
2. 需要使用的第三方库在**requirements.txt**，如下：
- numpy
- imutils
- sklearn
- tensorflow
- keras
- opencv2-python

```
pip3 install -r requirements.txt
```

## 解压

- 将**gen_captcha.7z**解压

## 提取

- 从验证码图像中提取单个字符

```sh
python extract_single_letters_from_captchas.py
```

提取出来的单个字符集会包含在**ext_letter**这个文件夹中

## 训练

- 训练神经网络模型去识别单个字符

```sh
python train_model.py
```

```sh
C:Python36\lib\site-packages\sklearn\externals\joblib\externals\cloudpickle\cloudpickle.py:47: DeprecationWarning: the imp module is deprecated in favour of importlib; see the module's documentation for alternative uses
  import imp
Using TensorFlow backend.
Train on 29058 samples, validate on 9686 samples
Epoch 1/10
2018-12-11 14:59:24.134434: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2
29058/29058 [==============================] - 75s 3ms/step - loss: 0.2216 - acc: 0.9467 - val_loss: 0.0177 - val_acc: 0.9953
Epoch 2/10
29058/29058 [==============================] - 71s 2ms/step - loss: 0.0119 - acc: 0.9971 - val_loss: 0.0126 - val_acc: 0.9962
Epoch 3/10
29058/29058 [==============================] - 72s 2ms/step - loss: 0.0073 - acc: 0.9982 - val_loss: 0.0080 - val_acc: 0.9974
Epoch 4/10
29058/29058 [==============================] - 73s 2ms/step - loss: 0.0043 - acc: 0.9985 - val_loss: 0.0131 - val_acc: 0.9964
Epoch 5/10
29058/29058 [==============================] - 72s 2ms/step - loss: 0.0043 - acc: 0.9990 - val_loss: 0.0091 - val_acc: 0.9978
Epoch 6/10
29058/29058 [==============================] - 72s 2ms/step - loss: 0.0021 - acc: 0.9996 - val_loss: 0.0626 - val_acc: 0.9839
Epoch 7/10
29058/29058 [==============================] - 73s 3ms/step - loss: 0.0059 - acc: 0.9981 - val_loss: 0.0059 - val_acc: 0.9983
Epoch 8/10
29058/29058 [==============================] - 73s 3ms/step - loss: 3.1436e-04 - acc: 0.9999 - val_loss: 0.0061 - val_acc: 0.9982
Epoch 9/10
29058/29058 [==============================] - 80s 3ms/step - loss: 3.8522e-05 - acc: 1.0000 - val_loss: 0.0054 - val_acc: 0.9988
Epoch 10/10
29058/29058 [==============================] - 74s 3ms/step - loss: 3.6896e-06 - acc: 1.0000 - val_loss: 0.0053 - val_acc: 0.9988
```
PS C:\Users\j8\Pictures\captcha\solving_captchas>
PS C:\Users\j8\Pictures\captcha\solving_captchas> python .\solve_captchas_with_model.py
Using TensorFlow backend.
2018-12-11 15:12:26.439293: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2
验证码包含的文本内容是: MXSB
验证码包含的文本内容是: V47H
验证码包含的文本内容是: H9WC
验证码包含的文本内容是: HEGD
验证码包含的文本内容是: JSEL
验证码包含的文本内容是: 4JMJ
验证码包含的文本内容是: G2D4
验证码包含的文本内容是: JPNX
验证码包含的文本内容是: XVSZ
PS C:\Users\j8\Pictures\captcha\solving_captchas>

运行完毕会出现**captcha_model.hdf5**和**model_labels.dat**


## 识别

- 使用训练出来的模型去识别验证码

```sh
python3 solve_captchas_with_model.py
```

- 如下：

<img src="img\captcha.gif"></img>

## Q.E.D

Enjoy it!