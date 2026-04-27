from ultralytics import YOLO
import cv2

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

if __name__ == '__main__':
    model = YOLO("runs/detect/train/weights/best.pt")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(
            frame,
            conf=0.7,
            iou=0.1,
            imgsz=960,
            max_det=20
        )

        frame_show = results[0].plot()
        cv2.imshow("手机增强识别", frame_show)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()