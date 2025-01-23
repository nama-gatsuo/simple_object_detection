import cv2
import wmi

def get_camera_names():
    """WMIを使用して接続されているカメラの名前を取得"""
    c = wmi.WMI()
    camera_names = []
    for device in c.Win32_PnPEntity():
        # device.Name が None の場合に備えて処理
        device_name = getattr(device, "Name", None)
        if device_name and ("camera" in device_name.lower() or "video" in device_name.lower()):
            camera_names.append(device_name)
    return camera_names

def find_available_cameras(max_cameras=20):
    """OpenCVでカメラIDを確認"""
    available_cameras = []
    for camera_id in range(max_cameras):
        cap = cv2.VideoCapture(camera_id)
        if cap.isOpened():
            available_cameras.append(camera_id)
            cap.release()
    return available_cameras

def match_camera_names_and_ids():
    """カメラ名称とOpenCVのカメラIDを対応付けて出力"""
    camera_names = get_camera_names()
    available_ids = find_available_cameras()

    print("接続されているカメラデバイス:")
    for camera_id in available_ids:
        # 名前をIDに適切に対応付ける（近い順序で推定）
        name = camera_names[camera_id] if camera_id < len(camera_names) else "不明なデバイス"
        print(f"カメラID: {camera_id}, デバイス名: {name}")

# 実行
match_camera_names_and_ids()
