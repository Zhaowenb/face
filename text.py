import cv2
import numpy as np
# 加载预训练的模型
modelFile = "res10_300x300_ssd_iter_140000.caffemodel"
configFile = "deploy.prototxt"
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
cap = cv2.VideoCapture('12.mp4')
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
