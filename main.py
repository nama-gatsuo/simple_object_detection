from ultralytics import YOLO
import cv2
import time
import sys
import zmq
import msgpack
import logging
from PIL import Image
import io
import numpy as np


# ログレベルをWARNING以上に設定
logging.getLogger('ultralytics').setLevel(logging.WARNING)

# ZeroMQの設定
ZMQ_IP = "127.0.0.1"  # ZeroMQ受信側のIPアドレス
ZMQ_PORT = 5555  # ZeroMQ受信側のポート
context = zmq.Context()
socket = context.socket(zmq.PUB)
socket.bind(f"tcp://{ZMQ_IP}:{ZMQ_PORT}")

# YOLOv11のモデルをロード
model = YOLO("yolo11n-seg.pt", verbose=False) 

# Webカメラの起動
CAPTURE_DEVICE_ID = 11
cap = cv2.VideoCapture(CAPTURE_DEVICE_ID, cv2.CAP_DSHOW)

# フレームレート設定（例: 30FPS）
frame_rate = 30
prev_time = 0

# 検出の信頼度しきい値
confidence_threshold = 0.1

# ラベルごとの固定カラー定義
label_colors = {
    "person": (255, 0, 0),    # 青
    "car": (0, 255, 0),       # 緑
    "dog": (0, 0, 255),       # 赤
    # 他のラベルと色を必要に応じて追加
}

while cap.isOpened():
    current_time = time.time()
    if (current_time - prev_time) < (1.0 / frame_rate):
        continue
    prev_time = current_time

    ret, frame = cap.read()
    if not ret:
        break

    # YOLOで物体検出を行う
    # results = model.track(frame, conf=confidence_threshold, persist=True)
    results = model.track(frame, conf=confidence_threshold)

    # フレーム内のオブジェクト情報を収集
    frame_data = []
    mask_overlay = np.zeros_like(frame, dtype=np.uint8)  # マスクを描画するための空白フレーム
    for i, box in enumerate(results[0].boxes):
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        confidence = box.conf[0].item()
        class_id = int(box.cls[0].item())
        tracking_id = box.id[0].item() if box.id is not None else -1  # トラッキングID
        tracking_id = int(tracking_id)
        if confidence < confidence_threshold:
            continue  # しきい値未満の検出を無視
        label = model.names[class_id]
        frame_data.append({"label": label, "confidence": confidence, "x1": x1, "y1": y1, "x2": x2, "y2": y2, "tracking_id": tracking_id})

        # セグメンテーションマスクをフレームに描画
        if results[0].masks is not None:
            mask = results[0].masks.data[i].cpu().numpy()  # マスクを取得 (uint8 のまま保持)
            color = label_colors.get(label, (255, 255, 255))  # 定義されていないラベルは白
            for c in range(3):
                mask_overlay[:, :, c] = mask * color[c] + mask_overlay[:, :, c]

    # 元のフレームとマスクを重ねる
    combined_frame = cv2.addWeighted(frame, 0.7, mask_overlay, 0.3, 0)

    # フレームをJPEG形式に変換
    pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    buffer = io.BytesIO()
    pil_frame.save(buffer, format="JPEG")
    jpeg_frame = buffer.getvalue()

    pil_mask = Image.fromarray(cv2.cvtColor(mask_overlay, cv2.COLOR_BGR2RGB))
    buffer = io.BytesIO()
    pil_mask.save(buffer, format="JPEG")
    jpeg_mask = buffer.getvalue()
    
    # フレームデータとJPEG画像をMessagePackでシリアライズしてZeroMQで送信
    packed_data = msgpack.packb({"frame_data": frame_data, "frame": jpeg_frame, "mask": jpeg_mask})
    socket.send(packed_data)

    # コンソールの出力を1行で更新
    sys.stdout.write(f"\rDetected objects: {len(results[0].boxes)}...")
    sys.stdout.flush()

    # 結果をフレームに描画して表示
    annotated_frame = results[0].plot()

    # ウィンドウに描画した結果を表示
    cv2.imshow("YOLO Detection", combined_frame)

    # 'q'を押すと終了
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
socket.close()
context.term()
