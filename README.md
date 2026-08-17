# Head-Coupled 3D Display

単眼カメラの顔・眼位置追跡を用い、観察者の両眼中点に追従する非対称透視投影で
`bunny.pcd` を表示する実装です。カメラ内部較正、カメラ・ディスプレイ外部姿勢、
利用者固有の眼位置を別プロファイルとして扱います。

## 重要な前提

添付資料には、実測済みの「ディスプレイ中央からカメラまでの高さ」または
「ディスプレイ法線からの下向き角」を含む外部較正結果は入っていませんでした。
`config/hardware_profile.demo.json` の値は、動作確認専用の人工値です。

- カメラ左右位置: 中央（0.0 cm）
- ディスプレイ中央から上: **20.0 cm**
- ディスプレイ面より手前: **2.5 cm**
- ディスプレイ法線から下向き: **10.0°**
- 出自: `synthetic_demo_not_measured`

物理機器で使用する前に、実測値または頭部ターゲット較正結果へ置き換えてください。
`headcoupled profile-summary` は、設定値ではなく最終的な4×4変換行列から上記の値を
逆算して表示します。

## 構成

```text
camera / synthetic source
        │
        ├── FaceMesh 478点（任意、添付コンポーネント）
        │       └── solvePnP → 頭部姿勢 → 左右眼球中心 → 両眼中点
        │
        └── TrackingState（画面座標系）
                ├── /ws/pose      JSON
                ├── /ws/camera    JPEG binary
                └── off-axis projection → WebGL2 / Canvas2D → bunny.pcd

AprilTag tagcal
        └── camera intrinsics K, D

5点/9点 head-ray calibration
        └── camera ↔ display external transform
```

ブラウザー側に顔推論を入れず、Python側は姿勢JSONとJPEGを別WebSocketで配信します。
表示側は外部依存のないES Modules実装で、WebGL2を優先し、管理対象ブラウザー等で
WebGLが使えない場合は同一の幾何式を用いるCanvas2Dへ切り替わります。

詳細は次を参照してください。

- `docs/source_review.md` — 添付会話・FaceMesh・tagcalの確認結果
- `docs/architecture.md` — 座標系、変換、コンポーネント、処理フロー
- `docs/specification.md` — API、プロファイル、品質基準、運用仕様
- `docs/test_report.md` — 合成データ試験とPlaywright試験結果
- `integrations/README.md` — 添付FaceMesh/tagcalとの接続手順

## セットアップ

Python制御系の既定環境は3.13です。`uv` と `just` を使用します。
`just` がない環境では `scripts/tasks.py` を代替として使用できます。

```bash
cd headcoupled-3d-display
just setup
just setup-browsers   # test-e2e / playwright-cli 用の Chromium 取得（初回のみ）
just serve
```

手動で同じ環境を作る場合:

```bash
uv venv --python 3.13
uv pip sync requirements.lock
PYTHONPATH=src .venv/bin/python -m headcoupled_display.cli serve \
  --profile config/hardware_profile.demo.json \
  --source synthetic
```

ブラウザーで `http://127.0.0.1:8000` を開きます。

### 主なタスク

```bash
just profile-summary
just synthetic-calibration
just test
just test-e2e
just playwright-cli
just check
```

`just` がない場合:

```bash
PYTHONPATH=src python scripts/tasks.py serve
PYTHONPATH=src python scripts/tasks.py check
PYTHONPATH=src python scripts/tasks.py playwright-cli
```

## 実測プロファイルの作成

### 1. カメラ内部較正

添付のtagcalを同梱しています。

```bash
just setup-tagcal
just tagcal-panel
```

CLI例:

```bash
cd components/tagcal
uv run tagcal screen show artifacts/screen --monitor DP-2 --tag-size-mm 40
uv run tagcal record capture.mp4 --camera 0 --width 1920 --height 1080
uv run tagcal process capture.mp4 artifacts/screen/pattern.json artifacts/session
cd ../..
```

結果をハードウェアプロファイルへ統合します。

```bash
PYTHONPATH=src .venv/bin/python -m headcoupled_display.cli import-tagcal \
  components/tagcal/artifacts/session/calibration.json \
  --profile config/hardware_profile.demo.json \
  --output config/hardware_profile.measured.json
```

内部較正値 `K, D` と外部姿勢は独立です。この操作だけではカメラ高さ・下向き角は
測定されません。

### 2. カメラ設置値

実測値を直接使用する場合、次を編集します。

```json
{
  "camera_mount": {
    "horizontal_offset_m": 0.0,
    "height_above_center_m": 0.2,
    "forward_offset_m": 0.025,
    "pitch_down_deg": 10.0,
    "yaw_right_deg": 0.0,
    "roll_clockwise_deg": 0.0
  }
}
```

外部較正で4×4行列が得られた場合は `camera_to_display_matrix` を指定します。
行列が存在する場合は行列を正とし、`camera_mount` は可読な初期値・説明値になります。

```bash
PYTHONPATH=src .venv/bin/python -m headcoupled_display.cli profile-summary config/hardware_profile.measured.json
```

### 3. FaceMesh実入力

添付FaceMeshはPython 3.10、ONNX Runtime GPU 1.18、CUDA 11.8/cuDNN 8系の独立環境です。
接続手順は `integrations/README.md` を参照してください。合成モードと異なり、実カメラ、
モデル重み、GPU/CPU推論の試験は対象機器上で行う必要があります。

## API

```text
GET  /api/health
GET  /api/profile
GET  /api/runtime
GET  /api/frame.jpg
POST /api/calibration/synthetic
POST /api/calibration/fit
GET  /api/calibration/status
WS   /ws/pose       TrackingState JSON
WS   /ws/camera     JPEG binary frame
```

OpenAPIは `/docs` で確認できます。

## テスト

実施済み試験:

- Python単体・API: 10件
- Playwright Python E2E: 1件
- Playwright CLIスクリーンショット: 1件
- 合成較正: 36標本、9画面点
- 代表結果: 高さ誤差0.09 mm、ピッチ誤差0.19°、平均レイ残差1.81 mm

スクリーンショットは `artifacts/` に保存されます。数値は決定的乱数シード
`20260817` の人工データに対する結果であり、実機精度を意味しません。

## プライバシー

本実装は人物の同定を行う「顔認識」ではなく、顔ランドマーク・頭部姿勢・眼位置の
追跡を目的とします。既定では映像や顔特徴を永続保存しません。JPEGは最新フレームを
メモリ上で配信するだけです。
