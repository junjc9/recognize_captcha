import cv2
import pickle
import os.path
import numpy as np
from imutils import paths
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Flatten, Dense
from helpers import resize_to_fit


IMG_DIR_OUTPUT_CHARACTER = "img_output_character"
MODEL_DIR_CAPTCHA = "model_captcha.hdf5"
MODEL_DIR_LABEL = "model_labels.dat"


# initialize the data and labels
# 初始化数据和标签
data = []
labels = []

# loop over the input images
# 遍历输入图像
for img_character in paths.list_images(IMG_DIR_OUTPUT_CHARACTER):
    # Load the image and convert it to grayscale
    # 加载图像并将其转换为灰度
    image = cv2.imread(img_character)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Resize the letter so it fits in a 20x20 pixel box
    # 调整字母大小，使其适合20x20像素框
    image = resize_to_fit(image, 20, 20)

    # Add a third channel dimension to the image to make Keras happy
    # 为图像添加第三个通道尺寸，适应keras
    image = np.expand_dims(image, axis=2)

    # Grab the name of the letter based on the folder it was in
    # 根据它所在的文件夹抓取该字母的名称
    label = img_character.split(os.path.sep)[-2]

    # Add the letter image and it's label to our training data
    # 将字母图像及其标签添加到我们的训练数据中
    data.append(image)
    labels.append(label)


# scale the raw pixel intensities to the range [0, 1] (this improves training)
# 将原始像素强度缩放到[0,1]范围（这样可以提高训练效果）
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# Split the training data into separate train and test sets
# 将训练数据拆分为单独的训练和测试集
(X_train, X_test, Y_train, Y_test) = train_test_split(data, labels, test_size=0.25, random_state=0)

# Convert the labels (letters) into one-hot encodings that Keras can work with
# 将标签（字母）转换为Keras可以使用的一位有效编码
lb = LabelBinarizer().fit(Y_train)
Y_train = lb.transform(Y_train)
Y_test = lb.transform(Y_test)

# Save the mapping from labels to one-hot encodings.
# We'll need this later when we use the model to decode what it's predictions mean
# 保存从标签到一位有效编码的映射。
# 当我们使用模型来解码它的预测意味着什么时，我们将需要这个
with open(MODEL_DIR_LABEL, "wb") as f:
    pickle.dump(lb, f)

# Build the neural network!
# 建立神经网络！
model = Sequential()

# First convolutional layer with max pooling
# 第一个具有最大池的卷积层
model.add(Conv2D(20, (5, 5), padding="same", input_shape=(20, 20, 1), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# Second convolutional layer with max pooling
# 具有最大池的第二卷积层
model.add(Conv2D(50, (5, 5), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# Hidden layer with 500 nodes
# 具有500个节点的隐藏层
model.add(Flatten())
model.add(Dense(500, activation="relu"))

# Output layer with 32 nodes (one for each possible letter/number we predict)
# 输出层有32个节点（我们预测的每个可能的字母/数字一个）
model.add(Dense(32, activation="softmax"))

# Ask Keras to build the TensorFlow model behind the scenes
# 要求Keras在幕后构建TensorFlow模型
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train the neural network
# 训练神经网络
model.fit(X_train, Y_train, validation_data=(X_test, Y_test), batch_size=32, epochs=10, verbose=1)

# Save the trained model to disk
# 将训练过的模型保存到磁盘
model.save(MODEL_DIR_CAPTCHA)
