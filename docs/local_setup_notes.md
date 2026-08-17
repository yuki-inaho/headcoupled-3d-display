# ローカル環境での受け入れ結果と修正記録

配布ZIP `headcoupled-3d-display.zip`
(SHA-256 `564398e7ea117e669bfcefd306f883368fa31feb76efcaddbcf473ab4cc85ef2`)
を展開し、この機体で実行できる状態にするまでに行った修正の記録です。

実行環境: Linux x86_64 / Python 3.13 (uv venv) / GTX 1070 / Chromium 143 (Playwright 管理)

---

## 1. 配布物の主張と実際の差分

| 配布物の主張 | 実際 | 対応 |
| :--- | :--- | :--- |
| `artifacts/ruff-report.txt`: "All checks passed!" | ruff 0.15.4 (同一版) で 8 件の指摘 | 全件修正し、レポートを本環境で再生成 |
| `docs/test_report.md`: WebSocket 姿勢配信 成功 | `requirements.lock` に WebSocket 実装が無く、uvicorn では `/ws/pose` `/ws/camera` が接続不能 | `websockets` を依存とlockへ追加 |
| Playwright CLI 試験 成功 | `just playwright-cli` が CLI を発見できず即時失敗 | venv 内 entry point の解決方法を修正 |
| ブラウザE2E 成功（Canvas2D経路） | `/usr/bin/chromium` 決め打ちで、本機には存在せず起動不能 | Playwright 同梱 Chromium を既定に変更 |

いずれも「配布元サンドボックスでは通っていたが、この環境では通らない」ではなく、
**配布された `requirements.lock` の構成では原理的に通らない**ものが含まれます。

---

## 2. 修正内容

### 2.1 WebSocket 実装の欠落（機能上もっとも重大）

uvicorn は WebSocket プロトコル実装を同梱しません。`websockets` も `wsproto` も
入っていない環境では `/ws/pose` `/ws/camera` が確立できず、ブラウザ側は
「再接続中」のまま姿勢が一切更新されません。

`tests/api/test_api.py` が通っていたのは Starlette `TestClient` が
ASGI を直接叩き uvicorn のプロトコル層を経由しないためで、実サーバでの
配信は検証できていませんでした。

- `pyproject.toml` に `websockets>=13,<18` を追加
- `requirements.lock` に `websockets==17.0.1` を追加

### 2.2 Playwright CLI の発見失敗

`scripts/playwright_cli_smoke.py`

```python
local_cli = Path(sys.executable).resolve().parent / "playwright"
```

uv が作る venv では `sys.executable` が共有インタプリタへのシンボリックリンクのため、
`resolve()` すると venv の外（entry point が存在しない場所）を指します。
`resolve()` を外しました。

### 2.3 Chromium 実行ファイルの決め打ち

`tests/e2e/test_browser.py` が `executable_path="/usr/bin/chromium"` 固定でした。
Playwright 管理の Chromium を既定とし、環境変数 `HEADCOUPLED_CHROMIUM` で上書き可能に
しました。取得用に `just setup-browsers` を追加しています。

### 2.4 Chromium ポリシー書き換えシムの既定無効化

`scripts/playwright_cli_smoke.py` の `provision_system_chromium_for_cli()` は、
system chromium を `~/.cache/ms-playwright/` 配下へシンボリックリンクします。
共有キャッシュを汚すため、`HEADCOUPLED_SYSTEM_CHROMIUM_SHIM=1` のときだけ動作するよう
変更しました。

なお `allow_localhost_for_managed_chromium()` は `/etc/chromium/policies/managed` が
存在するときのみ動作し、本機には存在しないため無効です。

### 2.5 lint 8 件

- `UP035` × 2: `typing.Iterator` → `collections.abc.Iterator`
- `UP017`: `datetime.now(timezone.utc)` → `datetime.now(UTC)`
- `F401`: 未使用 import
- `SIM105`: `contextlib.suppress` 化
- `B008` × 3: typer の引数既定値での関数呼び出しを、既定 `None` + 関数内解決へ変更
  （`default_hardware_profile_path()` は環境変数と cwd を参照するため、
  モジュール定数化ではなく遅延評価を維持した）

`UP017` の修正で `datetime.UTC` (3.11+) を使うため、`requires-python` を
`>=3.10` から `>=3.11` へ修正しました。ruff の `target-version` が元から `py311`
であり、宣言と実装が食い違っていたものを実装側に合わせています。

---

## 3. 本環境での検証結果

```text
just check          ruff All checks passed! / pytest 11 passed
just test-e2e       1 passed
just playwright-cli 成功（スクリーンショット生成）
```

### 3.1 合成較正（配布物の数値を再現）

```text
translation_error_mm  0.4641824454417735
height_error_mm       0.08574027645394389
rotation_error_deg    0.5609786045164163
pitch_error_deg       0.19253936527826632
```

`docs/test_report.md` 記載値と一致します。乱数シード `20260817` に対する決定的な結果です。

### 3.2 WebGL2 経路（配布元では未実行）

配布元サンドボックスの Chromium は WebGL が無効で Canvas2D 代替経路しか実行できて
いませんでしたが、本環境では **WebGL2 経路で 13,810 点の描画を確認**しました
（`artifacts/playwright-cli-dashboard.png` のフッタ表示が `WebGL2 / 13,810 points`）。

---

## 4. 未検証のまま残る項目

配布物の `docs/test_report.md` 8章の記載どおり、以下は実機接続が必要でした。
2026-08-17 の作業（`temp/workdoc_Aug17-2026_headcoupled_scene_latency.md`）で
一部が実測に置き換わったため、現状を分けて記します。

### 4.1 実測で検証済みになった項目

- **実機 tagcal 内部較正**: `apriltag-camera-calibrator/artifacts/eval_refine/calibration.json`
  を使用。1280×720、`plumb_bob`、RMS 1.0429 px、fx/fy = 1150.77 / 1150.62、
  cx/cy = 719.85 / 360.36。録画と解像度一致。
- **実カメラ入力と FaceMesh 推論**: GTX 1070 / CUDA 11.8 / onnxruntime-gpu 1.18.0 で
  静止画ベンチ mean 36.33 ms（detector 25.67 ms、FaceMesh 10.67 ms、27.5 FPS）。
  実セッションの provider が `CUDAExecutionProvider` 先頭であることを検査で強制。
- **実利用者の個人顔モデル**: `recordings/me/shape.pcd`（478点）が
  `UserProfile.face_model_path` から解決され、虹彩中心 468/473 が眼位置に使われることを
  実録画で確認。両眼中点は画面座標で約 `(0.005, 0.030, 0.547) m`。

### 4.2 依然として未検証の項目

- **実ディスプレイ上の 5点／9点 外部較正**: `config/hardware_profile.local.json` の設置値は
  ユーザーの実機目視確認であり、頭部レイ較正による実測ではない。
  `forward_offset_m = 0.0` も録画検証用の明示値である。
- **エンドツーエンド遅延・ジッタ（motion-to-photon）**: 録画入力では
  カメラ露光・キャプチャ時間を含められないため、実カメラでの受け入れが別途必要。
- **絶対深度の正しさ**: 録画時の `focus_absolute` が 332、較正時が 256 で不一致。
  相対運動と表示幾何の検証には使えるが、絶対距離の校正証拠にはしない。
  tagcal の主点 cx = 719.85 が画像中心 640 から 80 px ずれている点も併せて再確認が必要。

## 5. 同梱コンポーネントについての注意

`components/facemesh_tracking/` は `facemesh_tracking_reconstruction` の
**3次元復元機能を追加する前のスナップショット**です。以下を含みません。

```text
head_pose.py     canonical face + solvePnP 頭部姿勢
calibration.py   外部キャリブレーション読み込み
capture.py       v4l2 固定設定つき録画
capture_gui.py   キーフレーム撮影GUI
reconstruct.py   GTSAM バンドル調整・左右対称化・PD実寸化
```

このスナップショットの不足を補うため、後続の修正で
`src/headcoupled_display/tracking.py` に次を実装しました。

- 6点 `SOLVEPNP_ITERATIVE` を骨格に近い12点 `SOLVEPNP_SQPNP` へ置換し、
  `tvec.z > 0` のcheiralityを必須化。鏡像解を成功扱いしない。
- `UserProfile.face_model_path` で `reconstruct.py` 出力の478点 `shape.pcd` を指定可能にし、
  canonicalとのKabsch照合後に虹彩中心（468/473）を左右眼位置として使う。

PCDはOpenCV/mm座標で保存されることを実データで確認したため、head frameへ正規化してから
PnPと眼位置変換を行う。一般値は個人PCD未指定時だけのフォールバックである。
