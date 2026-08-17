# facemesh_tracking

**onnxruntime-gpu バックエンドで顔検出 → FaceMesh 478点ランドマーク推定**を行う uv プロジェクト。

```
frame(BGR) ─► YOLOv8-Face ─► BBox + 5点キーポイント ─► FaceMesh V2_478 ─► FaceLandmarks(478×3)
                                   (目2点で crop を正立)
```

モデルは [UniFace](https://github.com/yakhyo/uniface) が提供する ONNX 重み（計 19 MB）。
検出・推定の各段は Protocol で抽象化されており、独立に差し替えられる。

## すぐ動かす

```bash
just setup            # uv sync
just doctor           # GPU / ORT プロバイダ / モデル / カメラを確認
just cam              # USB カメラをリアルタイム表示 (ESC で終了)
just image ~/face.jpg # 画像を推論して outputs/ に保存
```

重みは初回実行時に自動取得される（`models/` にキャッシュ）。
**初回推論だけ PTX の JIT で数秒かかる**（2 枚目以降は通常速度）。

`just --list` で全レシピ。主なもの:

| レシピ | 内容 |
| --- | --- |
| `just cam [DEVICE]` | カメラをリアルタイム表示 |
| `just image IMG` / `just show IMG` | 画像を保存 / ウィンドウ表示 |
| `just video VID` | 動画を推論して mp4 に保存 |
| `just inspect IMG` | 黒背景に輪郭+点のみ（当たり具合の確認用） |
| `just json SRC` | ランドマークを JSON 出力 |
| `just bench [IMG]` | 段階別の速度計測 |
| `just bench-backends IMG` | CUDA と CPU を比較 |
| `just check` | lint + test |

## CLI

```bash
uv run facemesh run --source 0 --show --width 640 --height 480
uv run facemesh run --source input.mp4 --output out.mp4 --mode full
uv run facemesh bench --source face.jpg --iterations 30
```

| オプション | 説明 |
| --- | --- |
| `--backend {cuda,tensorrt,cpu}` | 実行プロバイダ（既定 `cuda`） |
| `--mode {full,partial,points}` | 描画（全メッシュ 2556 辺 / 輪郭 124 辺 + 点 / 点のみ）。虹彩は常に赤点 |
| `--detection-threshold` / `--landmark-threshold` | 検出・顔存在確率の下限（既定 0.5） |
| `--margin-ratio` | BBox の追加拡張（既定 0.0、後述） |
| `--no-background` / `--no-boxes` | 黒背景に描画 / 検出枠を非表示 |
| `--max-frames N` | N フレームで打ち切り |

## ライブラリとして使う

```python
from facemesh_tracking import Backend, FaceMeshPipeline, render

pipeline = FaceMeshPipeline.create(backend=Backend.CUDA)
result = pipeline.process(frame_bgr)
canvas = render(frame_bgr, result.faces, result.boxes)

for face in result.faces:
    face.xy  # (478, 2) 元画像ピクセル座標 (float32, サブピクセル)
    face.points  # (478, 3) X, Y, Z
    face.irises  # (10, 2) 虹彩点
    face.score  # 顔存在確率 [0, 1]
```

段の差し替えは Protocol を満たすクラスを渡すだけ:

```python
from facemesh_tracking import FaceMeshPipeline, UnifaceFaceMesh

pipeline = FaceMeshPipeline(MyOwnDetector(), UnifaceFaceMesh())
```

- `FaceDetector` — `detect(image_bgr) -> list[BBox]`
- `LandmarkEstimator` — `estimate(image_bgr, boxes) -> list[FaceLandmarks]`

## 構成

| モジュール | 責務 |
| --- | --- |
| `protocols.py` | `FaceDetector` / `LandmarkEstimator` の 2 インタフェース |
| `geometry.py` | `BBox`（+5点キーポイント）/ `FaceLandmarks`（468 or 478点） |
| `uniface_models.py` | UniFace の YOLOv8-Face / FaceMesh ラッパ、モデルキャッシュ設定 |
| `runtime.py` | バックエンド選択と CUDA ライブラリ解決 |
| `pipeline.py` | 2 段の合成 |
| `visualize.py` | メッシュ・虹彩・枠の描画（推論非依存） |
| `media.py` | 画像 / 動画 / カメラの入出力 |
| `cli.py` | `facemesh {run,bench}` |

## GPU 実行に関する注意（Pascal 世代）

`onnxruntime-gpu==1.18.0`（PyPI wheel）は **CUDA 11.8 + cuDNN 8.9** ビルド。
cuDNN 9 には Pascal（sm_61 / GTX 1070 など）向けの Conv 実行カーネルが無く、
システム側が cuDNN 9 だと畳み込みが `CUDNN_STATUS_EXECUTION_FAILED` で落ちる。

そのため `nvidia-*-cu11` wheel で **venv 内に CUDA 11.8 ランタイムを持ち**、
`runtime.preload_cuda_libraries()` が onnxruntime の import 前に `RTLD_GLOBAL` でそれらを
先にロードする。UniFace が内部で onnxruntime を import する前にも同じ関数を通すので自動で効く。
UniFace の ORT 要件は `>=1.16.0,<1.24`（Python 3.10）で、1.18.0 ピンと矛盾しない。

TensorRT バックエンドは `TensorrtExecutionProvider` が使える環境でのみ選択すること。

## モデル

`models/` に集約される（UniFace のキャッシュを `set_cache_dir()` でここに向けている）。

| ファイル | 内容 |
| --- | --- |
| `yolov8n_face.onnx` (12 MB) | YOLOv8-Face、WIDER FACE Easy 94.6% / Medium 92.3% / Hard 79.6% |
| `face_landmarker.onnx` (4.6 MB) | MediaPipe FaceMesh V2_478（Google の `face_landmarker.task` 由来） |

再生成は `python tools/extract_tessellation.py`。478点モデルでは先頭 468 点に適用され、虹彩は点で描かれる。

## テスト

```bash
uv run pytest       # 42 件。重みが無い場合、モデル依存の統合テストは自動 skip
uv run ruff check .
uv run ruff format --check .
```

統合テストはモデルを CPU で読み、座標空間・バッチ整合・キーポイント伝播
（整列キーポイントの有無で結果が変わること）を検証する。
