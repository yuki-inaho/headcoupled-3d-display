# 添付コンポーネントとの接続

## 1. tagcal

同梱場所: `components/tagcal/`

このコンポーネントはカメラ内部パラメータ `K, D` を作ります。カメラ・ディスプレイ外部姿勢は
別工程です。

```bash
cd components/tagcal
uv sync --all-extras
uv run tagcal panel
```

生成例:

```text
artifacts/session/calibration.json
artifacts/session/calibration_opencv.yaml
artifacts/session/camera_info.yaml
```

制御系へ統合:

```bash
cd ../..
uv run headcoupled import-tagcal \
  components/tagcal/artifacts/session/calibration.json \
  --profile config/hardware_profile.demo.json \
  --output config/hardware_profile.measured.json
```

対応形式:

- tagcal `calibration.json`
- OpenCV FileStorage YAML
- ROS CameraInfo YAML

## 2. facemesh_tracking

同梱場所: `components/facemesh_tracking/`

元コンポーネントはPython 3.10限定です。GPU構成を維持したまま制御系を同じ仮想環境へ
editable installする方法を推奨します。

```bash
cd components/facemesh_tracking
uv sync
uv pip install -e ../..

export FACEMESH_TRACKING_SOURCE="$PWD/src"
uv run headcoupled serve \
  --profile ../../config/hardware_profile.measured.json \
  --user-profile ../../config/user_profile.demo.json \
  --source facemesh \
  --backend cuda \
  --camera-index 0
```

CPU確認:

```bash
uv run headcoupled serve --source facemesh --backend cpu
```

事前診断:

```bash
just doctor
```

注意事項:

- モデル重みは初回取得されます。
- GPU版はCUDA 11.8/cuDNN 8系の仮想環境内ライブラリを使います。
- カメラの解像度、フォーカス、デジタルズームが内部較正時と一致している必要があります。
- 汎用顔モデルの眼球中心は近似です。精度用途では利用者較正が必要です。
- 実機・GPU試験は本生成環境では実施していません。

## 3. データ経路

FaceMesh統合時もブラウザー契約は変わりません。

```text
FaceMeshTrackingProvider
  → TrackingState
  → /ws/pose JSON

annotated camera frame
  → JPEG
  → /ws/camera binary
```

これにより、表示層はモデル、CUDA、OpenCV、カメラデバイスを認識しません。
