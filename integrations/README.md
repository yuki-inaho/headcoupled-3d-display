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

元コンポーネントはPython 3.10限定で、制御系はPython 3.11以上です。したがって、現在の
`FaceMeshTrackingProvider` を同一プロセスで使う旧手順（`uv pip install -e ../..`）は成立しません。
実カメラの確認はFaceMesh側の実行環境を使い、headcoupledのブラウザー表示は合成デモとして扱います。

この機体では複数のV4L2デバイスがあり、数値インデックス0は失敗することがあるため、必ず
`/dev/video0` をパスとして渡します。

```bash
cd "$FACEMESH_TRACKING_PROJECT"
just cam /dev/video0
# または headcoupled-3d-display から: just facemesh-live /dev/video0
```

復元済みの個人モデルは次です。どちらも実寸mmで、`shape.ply`は外部メッシュビューア
（CloudCompare/MeshLabなど）、`shape.pcd`はheadcoupledの個人眼位置プロファイル用です。

```bash
$FACEMESH_TRACKING_PROJECT/recordings/me/shape.ply
$FACEMESH_TRACKING_PROJECT/recordings/me/shape.pcd
```

事前診断:

```bash
just doctor
```

注意事項:

- モデル重みは初回取得されます。
- GPU版はCUDA 11.8/cuDNN 8系のFaceMesh仮想環境内ライブラリを使います。
- カメラの解像度、フォーカス、デジタルズームが内部較正時と一致している必要があります。
- browser表示へ実ランドマークを渡すには、Python 3.10 FaceMeshプロセスからのIPCブリッジが別途必要です。

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
