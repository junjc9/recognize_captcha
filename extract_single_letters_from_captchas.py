import os
import os.path
import cv2
import glob
import imutils


CAPTCHA_IMAGE_FOLDER = "gen_captcha"
OUTPUT_FOLDER = "ext_letter"


# Get a list of all the captcha images we need to process
# 获取我们需要处理的所有验证码图像的列表
captcha_image_files = glob.glob(os.path.join(CAPTCHA_IMAGE_FOLDER, "*"))
counts = {}

# loop over the image paths
# 在图像路径上遍历
for (i, captcha_image_file) in enumerate(captcha_image_files):
    print("[INFO] processing image {}/{}".format(i + 1, len(captcha_image_files)))

    # Since the filename contains the captcha text (i.e. "2A2X.png" has the text "2A2X"),
    # grab the base filename as the text
    # 由于文件名包含验证码文本（即“2A2X.png”具有文本“2A2X”），
    # 抓取基本文件名作为文本
    filename = os.path.basename(captcha_image_file)
    captcha_correct_text = os.path.splitext(filename)[0]

    # Load the image and convert it to grayscale
    # 加载图像并将其转换为灰度
    image = cv2.imread(captcha_image_file)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Add some extra padding around the image
    # 在图像周围添加一些额外的填充
    gray = cv2.copyMakeBorder(gray, 8, 8, 8, 8, cv2.BORDER_REPLICATE)

    # threshold the image (convert it to pure black and white)
    # 阈值图像（将其转换为纯黑色和白色）
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

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

    # Save out each letter as a single image
    # 将每个字母保存为单个图像
    for letter_bounding_box, letter_text in zip(letter_image_regions, captcha_correct_text):
        # Grab the coordinates of the letter in the image
        # 抓取图像中字母的坐标
        x, y, w, h = letter_bounding_box

        # Extract the letter from the original image with a 2-pixel margin around the edge
        # 从原始图像中提取字母，边缘周围有2像素的边距
        letter_image = gray[y - 2:y + h + 2, x - 2:x + w + 2]

        # Get the folder to save the image in 
        # 获取用于保存图像的文件夹
        save_path = os.path.join(OUTPUT_FOLDER, letter_text)

        # if the output directory does not exist, create it
        # 如果输出目录不存在，创建
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # write the letter image to a file
        # 将字母图像写入文件
        count = counts.get(letter_text, 1)
        p = os.path.join(save_path, "{}.png".format(str(count).zfill(6)))
        cv2.imwrite(p, letter_image)

        # increment the count for the current key
        # 递增当前键的计数
        counts[letter_text] = count + 1
