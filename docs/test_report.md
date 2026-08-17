# 試験報告

## 1. 試験環境

```text
OS/arch:          Linux x86_64
Python:           3.13.5
uv:               0.10.0
just:             1.46.0（成果物作成時のjustfile検証用）
Playwright:        1.57.0
Chromium:          142.0.7444.175
NumPy:             2.3.5
SciPy:             1.17.0
OpenCV:            4.13.0
FastAPI:           0.128.2
```

GPU、物理カメラ、FaceMeshモデルは使用せず、決定的人工データで試験しました。

## 2. 実行コマンド

```bash
PYTHONPATH=src pytest -q
ruff check src tests scripts
PYTHONPATH=src python scripts/playwright_cli_smoke.py

# justfile自体の検証
just --list
just --dry-run setup
just check
just playwright-cli
```

成果物環境には当初`just`が入っていなかったため、公式配布バイナリを一時的に使用して
justfileを実行確認しました。バイナリは成果物へ同梱していません。

## 3. Python試験

結果:

```text
11 passed in 3.74s
```

内訳:

| 分類 | 件数 | 対象 |
|---|---:|---|
| 幾何 | 4 | 設置値→行列→設置値、剛体逆変換、対称/非対称frustum |
| 較正 | 1 | 9点×4姿勢の外部姿勢復元 |
| プロファイル | 1 | tagcal JSON読込 |
| API | 4 | profile/health、姿勢WS、JPEG WS、較正API、JPEG HTTP |
| ブラウザーE2E | 1 | UI、PCD、WebSocket、較正操作、スクリーンショット |

## 4. 人工較正試験

### 条件

- 画面点: 3×3の9点
- 頭部位置: 各点4位置
- 合計: 36標本
- 方向ノイズ: 0.12°相当
- 乱数シード: `20260817`
- 真値: 高さ20.0 cm、手前2.5 cm、下向き10.0°、左右中央
- 初期値: 真値から高さ+1.8 cm、手前-1.2 cm、ピッチ-2.5°等を意図的に付加

### 結果

| 指標 | 結果 | 受入基準 |
|---|---:|---:|
| 平均点・レイ残差 | 1.810 mm | < 2.5 mm |
| 中央値点・レイ残差 | 1.724 mm | 参考 |
| 最大点・レイ残差 | 5.288 mm | 参考 |
| 平均角度誤差 | 0.147° | 参考 |
| 高さ誤差 | 0.086 mm | < 0.5 mm |
| 全並進誤差 | 0.464 mm | < 1.5 mm |
| ピッチ誤差 | 0.193° | < 0.35° |
| 全回転誤差 | 0.561° | 参考 |

最適化後の表示値:

```text
画面中央から上:        20.0086 cm
画面より手前:           2.4973 cm
下向きピッチ:           9.8075°
左右オフセット:         -0.0455 cm
左右中央判定:           true
```

これは人工真値への回帰精度であり、実機の測定精度を示しません。

## 5. Playwright Python E2E

確認項目:

- FastAPIサーバー起動と`/api/health`
- ページ初期化完了
- ASCII `bunny.pcd` 13,810点読込
- 姿勢WebSocketのsequence更新
- JPEG WebSocketのblob表示
- 高さ20.0 cm、下向き10.0°のUI表示
- 「合成較正を実行」ボタン
- 較正成功結果の表示
- フルページスクリーンショット生成
- JavaScript未処理例外なし

出力:

```text
artifacts/playwright-e2e-dashboard.png
```

成果物作成環境のChromiumは管理ポリシーでWebGLを無効化していたため、E2E時は
Canvas2D代替経路を使用しました。アプリケーションはWebGL2を最初に試み、使用可能な
ブラウザーではWebGL2を選びます。Python側で非対称frustum行列を別途単体試験しています。

また、この環境はChromiumのURL全体ブロックポリシーを持っていました。試験補助コードは
書込可能な場合だけlocalhostを試験中に一時許可し、`finally`で元のJSONをバイト単位で
復元します。通常の開発環境では変更しません。

## 6. Playwright CLI試験

実際の`playwright screenshot`コマンドを使用しました。

```text
Navigating to http://127.0.0.1:<ephemeral-port>
Waiting for selector body[data-ready="true"]...
Waiting for timeout 1500...
Capturing screenshot into artifacts/playwright-cli-dashboard.png
```

出力:

```text
artifacts/playwright-cli-dashboard.png
artifacts/playwright-cli-command.txt
artifacts/playwright-cli-report.txt
```

## 7. justfile試験

次を実行し、レシピの解釈と実行を確認しました。

```text
just --list             成功
just --dry-run setup    成功
just check              ruff + 11 pytest成功
just playwright-cli     成功
```

`setup`はネットワークを使用する新規インストールを避けるためdry-runです。成果物作成環境に
既にある完全固定バージョン群で`check`と`playwright-cli`を実行しました。
`requirements.lock`には試験環境の直接・推移依存を完全固定しています。

## 8. 未実施項目

- 物理カメラ入力
- 実機tagcal収録・内部較正
- 実ディスプレイ上の5点/9点外部較正
- FaceMeshモデルの取得・推論
- CUDA 11.8/cuDNN 8/ONNX Runtime GPU動作
- 実利用者の眼球中心・IPD較正
- 実機のエンドツーエンド遅延、ジッター、熱・長時間試験

これらは対象カメラ、ディスプレイ、GPU、利用者を必要とするため、同梱の実機手順に従い
対象環境で実施してください。
