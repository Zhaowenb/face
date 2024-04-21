# **安装过程：**

``` sh
sudo apt update
sudo apt install python3==3.8.10 python3-pip -y
# 等待下载完成
git clone https://github.com/Zhaowenb/face.git
# 等待下载完成
cd face
pip3 install requirements.txt
# 等待下载完成
python3 face.py
```

# **Python源码：**

``` python
import cv2
import numpy as np

# 加载预训练的模型
modelFile = "/home/zwbyyds/zwbyyds_work/worktext/opencv_ppt/res10_300x300_ssd_iter_140000.caffemodel"
configFile = "/home/zwbyyds/zwbyyds_work/worktext/opencv_ppt/deploy.prototxt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
print("模型加载成功。")

def see(img):
# 读取图像
    image = img
    # 获取图像尺寸
    (h, w) = image.shape[:2]
    # 从图像创建一个blob
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
        (300, 300), (104.0, 177.0, 123.0))
    # 输入blob到网络中
    net.setInput(blob)
    detections = net.forward()
    # 循环检测
    for i in range(0, detections.shape[2]):
        # 提取与检测相关的置信度（即概率）
        confidence = detections[0, 0, i, 2]
        # 过滤掉弱检测，确保置信度大于最小置信度
        if confidence > 0.5:
            # 计算面部边界框的 (x, y) 坐标
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            # 绘制面部边界框及置信度
            text = "{:.2f}%".format(confidence * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(image, (startX, startY), (endX, endY),
                        (0, 255, 0), 2)
            cv2.putText(image, text, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
# 显示输出图像
cv2.namedWindow('Face Detection', cv2.WINDOW_AUTOSIZE)
cap = cv2.VideoCapture('/home/zwbyyds/zwbyyds_work/worktext/opencv_ppt/images/12.mp4')
#文件是否打开
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()
while True:
    ret, frame = cap.read()
    if ret == False:
        break
    see(frame)
    cv2.imshow('Face Detection', frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
```

---
## **这个 Python 程序使用了 OpenCV 库来实现人脸识别功能。程序的主要步骤和组件的说明和注释如下：**

1. **导入必要的库**：
   - `cv2`：OpenCV 库，用于计算机视觉任务。
   - `numpy`：用于数值计算的库。

2. **加载预训练模型**：
   - `modelFile` 和 `configFile` 分别指向 Caffe 框架的预训练模型文件和配置文件。
   - `net` 是通过 OpenCV 的 `dnn` 模块加载的深度学习网络。

3. **定义人脸检测函数 `see`**：
   - 函数接收一个图像 `img` 作为输入。
   - 获取图像的高度和宽度 `(h, w)`。
   - 创建一个 blob，这是一个预处理后的图像，它被缩放并减去平均值，以便网络可以识别。
   - 将 blob 输入到网络中，执行前向传播，得到检测结果 `detections`。
   - 循环遍历检测结果，对于每个检测到的对象：
     - 提取置信度 `confidence`。
     - 如果置信度大于 0.5，则认为检测有效。
     - 计算并绘制边界框和置信度标签。

4. **视频流处理**：
   - 使用 `cv2.VideoCapture` 打开视频文件。
   - 检查视频是否成功打开。
   - 在一个循环中读取视频帧，并对每一帧使用 `see` 函数进行人脸检测。
   - 显示检测结果。
   - 如果按下 'q' 键，则退出循环。

5. **资源释放和窗口关闭**：
   - 释放视频捕获对象 `cap`。
   - 关闭所有 OpenCV 创建的窗口。

这个程序是一个完整的人脸检测系统，可以从视频流中实时识别人脸，并在每个检测到的人脸周围绘制边界框和置信度。