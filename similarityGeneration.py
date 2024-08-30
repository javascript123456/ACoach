
#——————————————————————
#用于视频流姿态识别匹配度生成
#——————————————————————

from turtledemo.penrose import start

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

# 框（rectangle）可视化配置
bbox_color = (150, 0, 0)  # 框的 BGR 颜色
bbox_thickness = 2  # 框的线宽

# 框类别文字
bbox_labelstr = {
    'font_size': 2,  # 字体大小6
    'font_thickness': 2,  # 字体粗细14
    'offset_x': 0,  # X 方向，文字偏移距离，向右为正
    'offset_y': 0,  # Y 方向，文字偏移距离，向下为正
}

#点厚度与框厚度
r = 2
thick = 2

# 关键点 BGR 配色
kpt_color_map = {
    0: {'name': 'Nose', 'color': [0, 0, 255], 'radius': r},  # 鼻尖
    1: {'name': 'Right Eye', 'color': [255, 0, 0], 'radius': r},  # 右边眼睛
    2: {'name': 'Left Eye', 'color': [255, 0, 0], 'radius': r},  # 左边眼睛
    3: {'name': 'Right Ear', 'color': [0, 255, 0], 'radius': r},  # 右边耳朵
    4: {'name': 'Left Ear', 'color': [0, 255, 0], 'radius': r},  # 左边耳朵
    5: {'name': 'Right Shoulder', 'color': [193, 182, 255], 'radius': r},  # 右边肩膀
    6: {'name': 'Left Shoulder', 'color': [193, 182, 255], 'radius': r},  # 左边肩膀
    7: {'name': 'Right Elbow', 'color': [16, 144, 247], 'radius': r},  # 右侧胳膊肘
    8: {'name': 'Left Elbow', 'color': [16, 144, 247], 'radius': r},  # 左侧胳膊肘
    9: {'name': 'Right Wrist', 'color': [1, 240, 255], 'radius': r},  # 右侧手腕
    10: {'name': 'Left Wrist', 'color': [1, 240, 255], 'radius': r},  # 左侧手腕
    11: {'name': 'Right Hip', 'color': [140, 47, 240], 'radius': r},  # 右侧胯
    12: {'name': 'Left Hip', 'color': [140, 47, 240], 'radius': r},  # 左侧胯
    13: {'name': 'Right Knee', 'color': [223, 155, 60], 'radius': r},  # 右侧膝盖
    14: {'name': 'Left Knee', 'color': [223, 155, 60], 'radius': r},  # 左侧膝盖
    15: {'name': 'Right Ankle', 'color': [139, 0, 0], 'radius': r},  # 右侧脚踝
    16: {'name': 'Left Ankle', 'color': [139, 0, 0], 'radius': r},  # 左侧脚踝
}

# 点类别文字
kpt_labelstr = {
    'font_size': 1,  # 字体大小4
    'font_thickness': 2,  # 字体粗细10
    'offset_x': 0,  # X 方向，文字偏移距离，向右为正
    'offset_y': 0,  # Y 方向，文字偏移距离，向下为正
}

# 骨架连接 BGR 配色
skeleton_map = [
    {'srt_kpt_id': 15, 'dst_kpt_id': 13, 'color': [0, 100, 255], 'thickness': thick},  # 右侧脚踝-右侧膝盖
    {'srt_kpt_id': 13, 'dst_kpt_id': 11, 'color': [0, 255, 0], 'thickness': thick},  # 右侧膝盖-右侧胯
    {'srt_kpt_id': 16, 'dst_kpt_id': 14, 'color': [255, 0, 0], 'thickness': thick},  # 左侧脚踝-左侧膝盖
    {'srt_kpt_id': 14, 'dst_kpt_id': 12, 'color': [0, 0, 255], 'thickness': thick},  # 左侧膝盖-左侧胯
    {'srt_kpt_id': 11, 'dst_kpt_id': 12, 'color': [122, 160, 255], 'thickness': thick},  # 右侧胯-左侧胯
    {'srt_kpt_id': 5, 'dst_kpt_id': 11, 'color': [139, 0, 139], 'thickness': thick},  # 右边肩膀-右侧胯
    {'srt_kpt_id': 6, 'dst_kpt_id': 12, 'color': [237, 149, 100], 'thickness': thick},  # 左边肩膀-左侧胯
    {'srt_kpt_id': 5, 'dst_kpt_id': 6, 'color': [152, 251, 152], 'thickness': thick},  # 右边肩膀-左边肩膀
    {'srt_kpt_id': 5, 'dst_kpt_id': 7, 'color': [148, 0, 69], 'thickness': thick},  # 右边肩膀-右侧胳膊肘
    {'srt_kpt_id': 6, 'dst_kpt_id': 8, 'color': [0, 75, 255], 'thickness': thick},  # 左边肩膀-左侧胳膊肘
    {'srt_kpt_id': 7, 'dst_kpt_id': 9, 'color': [56, 230, 25], 'thickness': thick},  # 右侧胳膊肘-右侧手腕
    {'srt_kpt_id': 8, 'dst_kpt_id': 10, 'color': [0, 240, 240], 'thickness': thick},  # 左侧胳膊肘-左侧手腕
    {'srt_kpt_id': 1, 'dst_kpt_id': 2, 'color': [224, 255, 255], 'thickness': thick},  # 右边眼睛-左边眼睛
    {'srt_kpt_id': 0, 'dst_kpt_id': 1, 'color': [47, 255, 173], 'thickness': thick},  # 鼻尖-左边眼睛
    {'srt_kpt_id': 0, 'dst_kpt_id': 2, 'color': [203, 192, 255], 'thickness': thick},  # 鼻尖-左边眼睛
    {'srt_kpt_id': 1, 'dst_kpt_id': 3, 'color': [196, 75, 255], 'thickness': thick},  # 右边眼睛-右边耳朵
    {'srt_kpt_id': 2, 'dst_kpt_id': 4, 'color': [86, 0, 25], 'thickness': thick},  # 左边眼睛-左边耳朵
    {'srt_kpt_id': 3, 'dst_kpt_id': 5, 'color': [255, 255, 0], 'thickness': thick},  # 右边耳朵-右边肩膀
    {'srt_kpt_id': 4, 'dst_kpt_id': 6, 'color': [255, 18, 200], 'thickness': thick}  # 左边耳朵-左边肩膀
]

#角度计算函数
def degreeCalculate(point1,point2,point3):

    a = math.sqrt((point2[0] - point3[0]) * (point2[0] - point3[0]) + (point2[1] - point3[1]) * (point2[1] - point3[1]))
    b = math.sqrt((point1[0] - point3[0]) * (point1[0] - point3[0]) + (point1[1] - point3[1]) * (point1[1] - point3[1]))
    c = math.sqrt((point2[0] - point1[0]) * (point2[0] - point1[0]) + (point2[1] - point1[1]) * (point2[1] - point1[1]))
    if a and b and c != 0:
        ang = math.degrees(math.acos((a * a + c * c - b * b) / (2 * a * c)))
        return format(ang,'.2f')
    return 0

# 匹配度计算函数
# 返回的是不匹配值
def dataCompare(arr1,arr2):

    similarity = 0.0
    counter = 0

    for index in range(len(arr1)):
        if arr1[index]+arr2[index] == 0:
            similarity += 1
            pass
        similarity += abs(arr1[index]-arr2[index])/(arr1[index]+arr2[index])
        print('arr1: ',arr1[index],'  arr2: ',arr2[index],"   similarity: ",similarity)
        counter+=1

    similarity /= counter

    return similarity


#帧图像处理
def process_frame(img_bgr):

    arr = np.array

    results = model(img_bgr, verbose=False)  # verbose设置为False，不单独打印每一帧预测结果

    # 预测框的个数
    num_bbox = len(results[0].boxes.cls)

    # 预测框的 xyxy 坐标
    bboxes_xyxy = results[0].boxes.xyxy.cpu().numpy().astype('uint32')

    # 关键点的 xy 坐标
    bboxes_keypoints = results[0].keypoints.cpu().numpy()

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
        # 获取该框坐标
        bbox_xyxy = bboxes_xyxy[idx]

        # 获取框的预测类别（对于关键点检测，只有一个类别）
        bbox_label = str()

        # 画框
        img_bgr = cv2.rectangle(img_bgr, (bbox_xyxy[0], bbox_xyxy[1]), (bbox_xyxy[2], bbox_xyxy[3]), bbox_color,
                                bbox_thickness)

        # 写框类别文字：图片，文字字符串，文字左上角坐标，字体，字体大小，颜色，字体粗细
        img_bgr = cv2.putText(img_bgr, bbox_label,
                              (bbox_xyxy[0] + bbox_labelstr['offset_x'], bbox_xyxy[1] + bbox_labelstr['offset_y']),
                              cv2.FONT_HERSHEY_SIMPLEX, bbox_labelstr['font_size'], bbox_color,
                              bbox_labelstr['font_thickness'])

        # 画该框的骨架连接
        for skeleton in skeleton_map:
            # 获取起始点坐标
            srt_kpt_id = skeleton['srt_kpt_id']
            srt_kpt_x = int(bbox_keypoints.xy[0][srt_kpt_id][0])
            srt_kpt_y = int(bbox_keypoints.xy[0][srt_kpt_id][1])

            # 获取终止点坐标
            dst_kpt_id = skeleton['dst_kpt_id']
            dst_kpt_x = int(bbox_keypoints.xy[0][dst_kpt_id][0])
            dst_kpt_y = int(bbox_keypoints.xy[0][dst_kpt_id][1])

            # 获取骨架连接颜色
            skeleton_color = skeleton['color']

            # 获取骨架连接线宽
            skeleton_thickness = skeleton['thickness']

            # 画骨架连接 忽略超出屏幕外的点
            if (srt_kpt_x or srt_kpt_y !=0) and (dst_kpt_x or dst_kpt_y != 0):
                img_bgr = cv2.line(img_bgr, (srt_kpt_x, srt_kpt_y), (dst_kpt_x, dst_kpt_y), color=skeleton_color,
                                   thickness=skeleton_thickness)

            # 画该框的关键点
        for kpt_id in kpt_color_map:
            # 获取该关键点的颜色、半径、XY坐标
            kpt_color = kpt_color_map[kpt_id]['color']
            kpt_radius = kpt_color_map[kpt_id]['radius']
            kpt_x = int(bbox_keypoints.xy[0][kpt_id][0])
            kpt_y = int(bbox_keypoints.xy[0][kpt_id][1])

            # 画圆：图片、XY坐标、半径、颜色、线宽（-1为填充）
            img_bgr = cv2.circle(img_bgr, (kpt_x, kpt_y), kpt_radius, kpt_color, -1)

            kpt_label = str()

                # 写关键点类别文字：图片，文字字符串，文字左上角坐标，字体，字体大小，颜色，字体粗细
            img_bgr = cv2.putText(img_bgr, kpt_label,
                                  (kpt_x + kpt_labelstr['offset_x'], kpt_y + kpt_labelstr['offset_y']),
                                  cv2.FONT_HERSHEY_SIMPLEX, kpt_labelstr['font_size'], kpt_color,
                                  kpt_labelstr['font_thickness'])

    return img_bgr,arr


#帧处理模板
def generate_video(input_path='videos/robot.mp4' , standared_data_path='data.txt'):

    # 导入标准数据表
    standaredData = []

    with open(standared_data_path, 'r') as file:

        data = file.readlines()

        for line in data:

            line = line.strip()

            if float(line.split(' ')[1]) < 0:
                sampleFrame = line.split(' ')
                sampleFrame = [float(val) for val in sampleFrame]
                print('sampleFrame: ', sampleFrame)
                continue

            row = line.split(' ')
            print('row: ',row)

            row = [float(val) for val in row]

            standaredData.append(row)

    filehead = input_path.split('/')[-1]
    output_path = "out-" + filehead

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
    frame_size = (cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)

    out = cv2.VideoWriter(output_path, fourcc, fps, (int(frame_size[0]), int(frame_size[1])))

    dataArr = []
    arr = np.array

    # 进度条绑定视频总帧数
    with tqdm(total=frame_count - 1) as pbar:
        try:
            while (cap.isOpened()):
                success, frame = cap.read()
                if not success:
                    break

                try:
                    frame , arr = process_frame(frame)

                    #样例数据生成
                    arr = [float(val) for val in arr]
                    dataArr.append(arr)

                except:
                    print('error')
                    pass

                if success == True:

                    out.write(frame)

                    # 进度条更新一帧
                    pbar.update(1)

        except:
            print('中途中断')
            pass

    cv2.destroyAllWindows()
    out.release()
    cap.release()

    # 匹配度计算
    # 获取动作起始帧
    startPoint = 0
    startPointSet = []
    #sampleFrame = 100
    isGetStartPoint = False
    while(1):

        startPiece = []

        # 传入视频的每一帧都与标准视频的前三帧匹配，得到可能的开始节点
        for index in range(3):
            startPiece.append(dataCompare(dataArr[startPoint], standaredData[index]))


        simCounter = min(startPiece[0],startPiece[1],startPiece[2])

        # 获得所有可能的运动开始帧
        if simCounter < 0.2:

            startPointSet.append(startPoint)
            isGetStartPoint = True

        startPoint += 1

        if startPoint == frame_count -4 and isGetStartPoint == False:
            print('动作匹配失败，请重新上传视频')
            return

        if startPoint == frame_count -4 and isGetStartPoint == True:
            break

    # 计算匹配度
    result = 100
    for point in startPointSet:
        frameCount = min(frame_count - point, int(sampleFrame[0])-4)
        print('frameCount: ',frameCount)

        tmpSim = 0
        for index in range(frameCount - 4):
            pieceResult = []

            for framePiece in range(3):
                pieceResult.append(dataCompare(dataArr[point + index], standaredData[index + framePiece]))

            tmpSim += min(pieceResult[0], pieceResult[1], pieceResult[2])

        tmpSim /= frameCount

        result = min(result,tmpSim)

    print('本次运动您的动作匹配度为: ',int((1-result)*100),'%')

    print('视频已保存', output_path)

generate_video('trainee.mp4','sample.mp4.data.txt')