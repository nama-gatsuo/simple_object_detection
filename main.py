import cv2
from ultralytics import YOLO
from pythonosc.udp_client import SimpleUDPClient
import time
import sys

import logging

# ログレベルをWARNING以上に設定
logging.getLogger('ultralytics').setLevel(logging.WARNING)

# OSCクライアントの設定
OSC_IP = "127.0.0.1"  # OSC受信側のIPアドレス
OSC_PORT = 8000  # OSC受信側のポート
osc_client = SimpleUDPClient(OSC_IP, OSC_PORT)

# YOLOv11のモデルをロード
model = YOLO("yolo11n.pt", verbose=False) 

# Webカメラの起動
CAPTURE_DEVICE_ID = 2
cap = cv2.VideoCapture(CAPTURE_DEVICE_ID)

# フレームレート設定（例: 30FPS）
frame_rate = 30
prev_time = 0

while cap.isOpened():
    current_time = time.time()
    if (current_time - prev_time) < (1.0 / frame_rate):
        continue
    prev_time = current_time

    ret, frame = cap.read()
    if not ret:
        break

    # YOLOで物体検出を行う
    results = model(frame)

    # オブジェクト情報をOSCで送信
    for result in results[0].boxes.data.tolist():
        x1, y1, x2, y2, confidence, class_id = result
        label = model.names[int(class_id)]
        osc_client.send_message("/object", [label, confidence, x1, y1, x2, y2])

    # コンソールの出力を1行で更新
    sys.stdout.write(f"\rDetected objects: {len(results[0].boxes)}")
    sys.stdout.flush()

    # 結果をフレームに描画して表示
    annotated_frame = results[0].plot()

    # ウィンドウに描画した結果を表示
    cv2.imshow("YOLO Detection", annotated_frame)

    # 'q'を押すと終了
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
