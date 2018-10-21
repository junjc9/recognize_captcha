from keras.models import load_model
from helpers import resize_to_fit
from imutils import paths
import numpy as np
import imutils
import cv2
import pickle


MODEL_FILENAME = "captcha_model.hdf5"
MODEL_LABELS_FILENAME = "model_labels.dat"
CAPTCHA_IMAGE_FOLDER = "gen_captcha"


# 加载模型标签（这样我们就可以将模型预测转换为实际字母）
with open(MODEL_LABELS_FILENAME, "rb") as f:
    lb = pickle.load(f)

# 加载训练好的神经网络
model = load_model(MODEL_FILENAME)

# 抓取一些随机的CAPTCHA图像进行测试。
# 在现实世界中，您将用代码替换此部分以获取真实内容
# 来自实时网站的CAPTCHA图片。
captcha_image_files = list(paths.list_images(CAPTCHA_IMAGE_FOLDER))
captcha_image_files = np.random.choice(captcha_image_files, size=(10,), replace=False)

# loop over the image paths
# 在图像路径上遍历
for image_file in captcha_image_files:
    # Load the image and convert it to grayscale
    # 加载图像并将其转换为灰度
    image = cv2.imread(image_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 在图像周围添加一些额外的填充
    image = cv2.copyMakeBorder(image, 20, 20, 20, 20, cv2.BORDER_REPLICATE)

    # threshold the image (convert it to pure black and white)
    # 阈值图像（将其转换为纯黑色和白色）
    thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    # find the contours (continuous blobs of pixels) the image
    # 找到图像的轮廓（连续斑点像素）
    contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Hack for compatibility with different OpenCV versions
    # Hack兼容不同的OpenCV版本
    contours = contours[0] if imutils.is_cv2() else contours[1]

    letter_image_regions = []

    # Now we can loop through each of the four contours and extract the letter
    # inside of each one
    # 现在我们可以从每一幅图内部循环遍历四个字母的轮廓中并提取字母
    for contour in contours:
        # Get the rectangle that contains the contour
        # 获取包含轮廓的矩形
        (x, y, w, h) = cv2.boundingRect(contour)

        # Compare the width and height of the contour to detect letters that
        # are conjoined into one chunk
        # 比较轮廓的宽度和高度以检测字母
        # 被连成一个块
        if w / h > 1.25:
            # This contour is too wide to be a single letter!
            # Split it in half into two letter regions!
            # 这个轮廓太宽，不能成为一个字母！
            # 将它分成两半字母区域！
            half_width = int(w / 2)
            letter_image_regions.append((x, y, half_width, h))
            letter_image_regions.append((x + half_width, y, half_width, h))
        else:
            # This is a normal letter by itself
            # 这种就是符合标准的字母
            letter_image_regions.append((x, y, w, h))

    # If we found more or less than 4 letters in the captcha, our letter extraction
    # didn't work correcly. Skip the image instead of saving bad training data!
    # 如果我们在验证码中发现了多于或少于4个字母，我们的字母提取
    # 没有正常工作。跳过图像而不是保存不良的训练数据！
    if len(letter_image_regions) != 4:
        continue

    # Sort the detected letter images based on the x coordinate to make sure
    # we are processing them from left-to-right so we match the right image
    # with the right letter
    # 根据x坐标对检测到的字母图像进行排序以确保
    # 我们正在从左到右处理它们，所以我们匹配正确的图像
    # 用正确的字母
    letter_image_regions = sorted(letter_image_regions, key=lambda x: x[0])

    # Create an output image and a list to hold our predicted letters
    # 创建输出图像和列表以保存我们预测的字母
    output = cv2.merge([image] * 3)
    predictions = []

    # loop over the letters
    # 遍历字母集
    for letter_bounding_box in letter_image_regions:
        # Grab the coordinates of the letter in the image
        # 抓取图像中字母的坐标
        x, y, w, h = letter_bounding_box

        # Extract the letter from the original image with a 2-pixel margin around the edge
        # 从原始图像中提取字母，边缘周围有2像素的边距
        letter_image = image[y - 2:y + h + 2, x - 2:x + w + 2]

        # Re-size the letter image to 20x20 pixels to match training data
        # 将字母图像重新调整为20x20像素以匹配训练数据
        letter_image = resize_to_fit(letter_image, 20, 20)

        # Turn the single image into a 4d list of images to make Keras happy
        # 将单张图像转换为4d图像列表适应keras
        letter_image = np.expand_dims(letter_image, axis=2)
        letter_image = np.expand_dims(letter_image, axis=0)

        # Ask the neural network to make a prediction
        # 要求神经网络做出预测
        prediction = model.predict(letter_image)

        # Convert the one-hot-encoded prediction back to a normal letter
        # 将一位有效编码预测转换回普通字母
        letter = lb.inverse_transform(prediction)[0]
        predictions.append(letter)

        # draw the prediction on the output image
        # 在输出图像上绘制预测
        cv2.rectangle(output, (x - 2, y - 2), (x + w + 4, y + h + 4), (0, 255, 0), 1)
        cv2.putText(output, letter, (x - 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

    # Print the captcha's text
    # 打印验证码的文本
    captcha_text = "".join(predictions)
    print("CAPTCHA text is: {}".format(captcha_text))

    # Show the annotated image
    # 显示带注释的图像
    cv2.imshow("Output", output)
    cv2.waitKey()
