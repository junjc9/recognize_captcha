
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
C:\Python36\lib\site-packages\sklearn\externals\joblib\externals\cloudpickle\cloudpickle.py:47: DeprecationWarning: the imp module is deprecated in favour of importlib; see the module's documentation for alternative uses
  import imp
Using TensorFlow backend.
Train on 29058 samples, validate on 9686 samples
Epoch 1/10
2018-10-21 13:14:38.141068: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2
29058/29058 [==============================] - 71s 2ms/step - loss: 0.2356 - acc: 0.9422 - val_loss: 0.0273 - val_acc: 0.9935
Epoch 2/10
29058/29058 [==============================] - 69s 2ms/step - loss: 0.0148 - acc: 0.9959 - val_loss: 0.0130 - val_acc: 0.9959
Epoch 3/10
29058/29058 [==============================] - 69s 2ms/step - loss: 0.0064 - acc: 0.9981 - val_loss: 0.0069 - val_acc: 0.9982
Epoch 4/10
29058/29058 [==============================] - 69s 2ms/step - loss: 0.0054 - acc: 0.9986 - val_loss: 0.0647 - val_acc: 0.9783
Epoch 5/10
29058/29058 [==============================] - 68s 2ms/step - loss: 0.0071 - acc: 0.9983 - val_loss: 0.0249 - val_acc: 0.9926
Epoch 6/10
29058/29058 [==============================] - 68s 2ms/step - loss: 0.0035 - acc: 0.9991 - val_loss: 0.0084 - val_acc: 0.9976
Epoch 7/10
29058/29058 [==============================] - 67s 2ms/step - loss: 0.0012 - acc: 0.9997 - val_loss: 0.0055 - val_acc: 0.9987
Epoch 8/10
29058/29058 [==============================] - 71s 2ms/step - loss: 0.0047 - acc: 0.9988 - val_loss: 0.0130 - val_acc: 0.9953
Epoch 9/10
29058/29058 [==============================] - 70s 2ms/step - loss: 0.0044 - acc: 0.9989 - val_loss: 0.0098 - val_acc: 0.9978
Epoch 10/10
29058/29058 [==============================] - 70s 2ms/step - loss: 0.0038 - acc: 0.9988 - val_loss: 0.0106 - val_acc: 0.9974
```

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