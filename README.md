# YOLO v11

* [ultralytics](https://github.com/ultralytics/ultralytics)
* python>=3.8, pytorch>=1.8
* 全てのコードをChat-GPT 4oで生成

## CondaでのYoloの基本環境構築
python=3.11.5  
conda=24.11.1  
[cuda=12.4](https://developer.nvidia.com/cuda-12-4-0-download-archive)

* condaを最新化
    ```
    conda update -n base conda
    ```
* [ultralytics, その依存をインストール](https://docs.ultralytics.com/quickstart/#install-ultralytics)

    ```
    conda create -n objectdetection 
    activate objectdetection
    conda install -c pytorch -c nvidia -c conda-forge pytorch torchvision pytorch-cuda=12.4 ultralytics
    ```

### テストコマンド
```
yolo predict model=yolo11n.pt source='https://ultralytics.com/images/bus.jpg'
```
画像が生成・格納されればOK

![](./runs/detect/predict/bus.jpg)


## OSCの送信機能をつける

```
conda install pip
pip install python-osc
```

```python
 for result in results[0].boxes.data.tolist():
        x1, y1, x2, y2, confidence, class_id = result
        label = model.names[int(class_id)]
        osc_client.send_message("/object", [label, confidence, x1, y1, x2, y2])
```