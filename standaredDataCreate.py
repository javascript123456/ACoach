
#——————————————————————
#用于标准数据集的生成
#——————————————————————

import numpy as np
from pandas.core.apply import frame_apply
from sympy.stats.sampling.sample_numpy import numpy
from torch.onnx.symbolic_opset9 import tensor
from ultralytics import YOLO

import cv2
import matplotlib.pyplot as plt
#%matplotlib inline

from tqdm import*
import torch

import math

# 载入模型
model = YOLO('yolov8n-pose.pt')

#角度计算函数
def degreeCalculate(point1,point2,point3):

    a = math.sqrt((point2[0] - point3[0]) * (point2[0] - point3[0]) + (point2[1] - point3[1]) * (point2[1] - point3[1]))
    b = math.sqrt((point1[0] - point3[0]) * (point1[0] - point3[0]) + (point1[1] - point3[1]) * (point1[1] - point3[1]))
    c = math.sqrt((point2[0] - point1[0]) * (point2[0] - point1[0]) + (point2[1] - point1[1]) * (point2[1] - point1[1]))
    if a and b and c != 0:
        ang = math.degrees(math.acos((a * a + c * c - b * b) / (2 * a * c)))
        return format(ang,'.2f')
    return 0

#帧图像处理
def process_frame(img_bgr):

    arr = np.array

    results = model(img_bgr, verbose=False)  # verbose设置为False，不单独打印每一帧预测结果

    # 预测框的个数
    num_bbox = len(results[0].boxes.cls)

    # 关键点的 xy 坐标
    bboxes_keypoints = results[0].keypoints.cpu().numpy()


    #img_bgr = cv2.imread(img_bgr)
    for idx in range(num_bbox):  # 遍历每个框

        bbox_keypoints = bboxes_keypoints[idx]  # 该框所有关键点坐标和置信度

        # 关键点角度计算
        arr = np.array(
            [
                # 左侧手臂
                degreeCalculate(bbox_keypoints.xy[0][10],bbox_keypoints.xy[0][8],bbox_keypoints.xy[0][6]),
                # 右侧手臂
                degreeCalculate(bbox_keypoints.xy[0][9],bbox_keypoints.xy[0][7],bbox_keypoints.xy[0][5]),
                # 左侧腿部
                degreeCalculate(bbox_keypoints.xy[0][16],bbox_keypoints.xy[0][14],bbox_keypoints.xy[0][12]),
                # 右侧腿部
                degreeCalculate(bbox_keypoints.xy[0][15],bbox_keypoints.xy[0][13],bbox_keypoints.xy[0][11]),
                # 左侧腋下
                degreeCalculate(bbox_keypoints.xy[0][12],bbox_keypoints.xy[0][10],bbox_keypoints.xy[0][8]),
                # 右侧腋下
                degreeCalculate(bbox_keypoints.xy[0][11],bbox_keypoints.xy[0][9],bbox_keypoints.xy[0][7])
            ]
        )

    return arr


#帧处理模板
def generate_video(input_path='videos/robot.mp4' , standared_data_path='data.txt'):

    # 导入标准数据表
    standaredData = []

    with open(standared_data_path, 'r') as file:

        data = file.readlines()

        for line in data:

            line = line.strip()

            row = line.split(' ')

            row = [float(val) for val in row]

            standaredData.append(row)

    print('视频开始处理', input_path)

    # 获取视频总帧数
    cap = cv2.VideoCapture(input_path)
    frame_count = 0
    while (cap.isOpened()):
        success, frame = cap.read()
        frame_count += 1
        if not success:
            break
    cap.release()
    print('视频总帧数为', frame_count)

    cap = cv2.VideoCapture(input_path)

    dataArr = []

    videoFrameCount = [frame_count-1,-1,-1,-1,-1,-1]
    dataArr.append(videoFrameCount)

    arr = np.array

    # 进度条绑定视频总帧数
    with tqdm(total=frame_count - 1) as pbar:

        try:
            while (cap.isOpened()):
                success, frame = cap.read()

                if not success:
                    break

                try:

                    arr = process_frame(frame)

                    #样例数据生成
                    arr = [float(val) for val in arr]
                    dataArr.append(arr)

                except:
                    print('error')
                    pass

                if success == True:

                    # 进度条更新一帧
                    pbar.update(1)

        except:
            print('中途中断')
            pass

    cv2.destroyAllWindows()
    cap.release()

    #样例数据保存
    np.savetxt(input_path+'.data.txt',dataArr,fmt='%s')

generate_video('sample.mp4','data.txt')