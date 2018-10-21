import imutils
import cv2


def resize_to_fit(image, width, height):
    """
    A helper function to resize an image to fit within a given size
    :param image: image to resize
    :param width: desired width in pixels
    :param height: desired height in pixels
    :return: the resized image
    帮助函数调整图像大小以适合给定大小
    param image：要调整大小的图像
    param width：所需宽度(以像素为单位)
    param height：所需高度(以像素为单位)
    return：调整大小后的图像
    """

    # grab the dimensions of the image, then initialize
    # the padding values
    # 抓取图像的尺寸，然后初始化填充值
    (h, w) = image.shape[:2]

    # if the width is greater than the height then resize along
    # the width
    # 如果宽度大于高度，则调整宽度
    if w > h:
        image = imutils.resize(image, width=width)

    # otherwise, the height is greater than the width so resize
    # along the height
    # 否则，高度大于宽度，则调整高度
    else:
        image = imutils.resize(image, height=height)

    # determine the padding values for the width and height to
    # obtain the target dimensions
    # 确定宽度和高度的填充值
    # 获得目标尺寸
    padW = int((width - image.shape[1]) / 2.0)
    padH = int((height - image.shape[0]) / 2.0)

    # pad the image then apply one more resizing to handle any
    # rounding issues
    # 填充图像，然后再应用一个调整大小来处理任何图像
    # 四舍五入的问题
    image = cv2.copyMakeBorder(image, padH, padH, padW, padW,
        cv2.BORDER_REPLICATE)
    image = cv2.resize(image, (width, height))

    # return the pre-processed image
    # 返回预处理的图像
    return image