import cv2
import numpy as np

# 识别圆形靶并输出坐标
def find_circle_target_center_realtime(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2) # 核大小和标准差可能需要调整

    # 2. Hough圆检测
    # dp: 累加器分辨率与图像分辨率的反比。
    # minDist: 检测到的圆的中心之间的最小距离。对于视频流，如果帧率较高，这个值可以适当减小，
    #          但如果靶子移动不快，保持一定距离避免重复检测。
    # param1: Canny边缘检测的高阈值。
    # param2: 累加器阈值。它越小，就越能检测到假圆圈。
    # minRadius: 圆半径的最小值。
    # maxRadius: 圆半径的最大值。
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=70,
                               param1=100, param2=35, minRadius=20, maxRadius=150)
                               # 请根据实际情况调整 minDist, param2, minRadius, maxRadius
    center_coords = None
    radius = None

    # 3. 提取并绘制检测到的圆
    if circles is not None:
        circles = np.uint16(np.around(circles))
        # 假设我们关注最大的那个圆，或者第一个检测到的显著圆
        # 在实际应用中，可能需要更复杂的逻辑来跟踪特定靶子
        # 这里我们简单取第一个检测到的圆
        target_circle = circles[0, 0]
        center_x, center_y, r = target_circle[0], target_circle[1], target_circle[2]
        center_coords = (center_x, center_y)
        radius = r

        # 在帧上绘制检测到的圆和圆心
        cv2.circle(frame, (center_x, center_y), r, (0, 255, 0), 2)  # 绘制圆轮廓
        cv2.circle(frame, (center_x, center_y), 2, (0, 0, 255), 3)   # 绘制圆心
        cv2.putText(frame, f"X: {center_x}, Y: {center_y}", (center_x - 50, center_y - r - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return frame, center_coords, radius

def get_DroidCam_url(ip, port=4747, res='1080p'):
    res_dict = {
        '240p': '320x240',
        '480p': '640x480',
        '720p': '1280x720',
        '1080p': '1920x1080',  
    }
    url = f'http://{ip}:{port}/mjpegfeed?{res_dict[res]}'
    return url

cap = cv2.VideoCapture(get_DroidCam_url('10.79.64.117', 4747, '1080p')) #仅需更改此处变量即可！

if __name__ == "__main__":

    if not cap.isOpened():
        exit()

    print("实时检测中... 按 'q' 键退出。")

    while True:
        # 逐帧捕获
        ret, frame = cap.read()

        if not ret:
            print("无法接收帧 (视频流结束?). 退出...")
            break

        # 调用函数进行检测
        processed_frame, center, r = find_circle_target_center_realtime(frame.copy()) # 传递帧的副本以避免修改原始帧

        if center:
            print(f"检测到靶子: 中心 X={center[0]}, Y={center[1]}, 半径={r}")
        else:
            pass #静默

        # 显示结果
        cv2.imshow('Real-time Circle Target Detection', processed_frame)

        # 按 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()