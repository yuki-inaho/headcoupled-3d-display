# 実装仕様

## 1. 機能要件

| ID | 要件 | 実装状態 |
|---|---|---|
| FR-001 | ディスプレイ中央原点の座標系を定義する | 実装済み |
| FR-002 | カメラ高さ・前後・ヨー・ピッチ・ロールから4×4変換を作る | 実装済み |
| FR-003 | 4×4変換から高さcm・下向き角度を逆算する | 実装済み |
| FR-004 | tagcal JSON/OpenCV YAML/ROS YAMLから内部パラメータを読む | 実装済み |
| FR-005 | 合成追跡で実カメラなしに全系を動作させる | 実装済み |
| FR-006 | 添付FaceMeshを任意バックエンドとして接続する | アダプター実装済み、実機未試験 |
| FR-007 | 両眼中点をWebSocket JSON配信する | 実装済み |
| FR-008 | JPEG映像を別WebSocketでバイナリ配信する | 実装済み |
| FR-009 | ASCII PCDを読み込み非対称投影する | 実装済み |
| FR-010 | WebGL2不可時にCanvas2Dへ切り替える | 実装済み |
| FR-011 | 5点以上の頭部レイから外部姿勢を最適化する | 実装済み |
| FR-012 | 9点人工データで較正回帰試験を行う | 実装済み |
| FR-013 | 人工値と実測値の出自をUI/APIへ表示する | 実装済み |
| FR-014 | Playwright APIおよびCLIでUIを試験する | 実装済み |
| FR-015 | 録画映像とFaceMesh保存JSONを同じ3D表示経路で再生する | 実装済み |

## 2. 非機能要件

| ID | 要件 | 方針 |
|---|---|---|
| NFR-001 | 再現性 | `uv`, `.python-version`, `requirements.lock`, `justfile` |
| NFR-002 | オフライン表示 | npm/CDNなし、静的ES Modules、同梱PCD |
| NFR-003 | 低遅延 | latest-value状態、姿勢と映像の分離 |
| NFR-004 | 交換可能性 | `TrackingProvider`, 独立プロファイル、API契約 |
| NFR-005 | プライバシー | 同定なし、既定で映像・特徴量を保存しない |
| NFR-006 | 監査性 | 出自、品質指標、人工データシード、試験報告 |
| NFR-007 | 入力交換可能性 | `FaceMeshFrameSource` により実カメラ・録画・IPCを同一のPnP/配信経路へ接続 |

## 3. ハードウェアプロファイル

```json
{
  "schema_version": 1,
  "profile_id": "device-id",
  "provenance": "measured | estimated_from_head_targets | synthetic_demo_not_measured | user_confirmed_mount_synthetic_intrinsics",
  "display": {
    "pixel_width": 2560,
    "pixel_height": 1440,
    "width_m": 0.596,
    "height_m": 0.335
  },
  "camera": {
    "image_width_px": 1280,
    "image_height_px": 720,
    "camera_matrix": [[950,0,640],[0,950,360],[0,0,1]],
    "distortion_coefficients": [],
    "distortion_model": "plumb_bob",
    "rms_reprojection_error_px": 0.24
  },
  "camera_mount": {
    "horizontal_offset_m": 0,
    "height_above_center_m": 0.2,
    "forward_offset_m": 0.025,
    "pitch_down_deg": 10,
    "yaw_right_deg": 0,
    "roll_clockwise_deg": 0
  },
  "camera_to_display_matrix": [[...],[...],[...],[...]],
  "quality_metrics": {}
}
```

優先順位:

1. `camera_to_display_matrix` があればそれを使用。
2. なければ `camera_mount` から構成。
3. UI表示値は、最終的に選択された行列から逆算。

同梱プロファイル:

| ファイル | provenance | 内容 |
| :--- | :--- | :--- |
| `config/hardware_profile.demo.json` | `synthetic_demo_not_measured` | 動作確認用の人工値（上20 cm、下向き10°） |
| `config/hardware_profile.local.json` | `user_confirmed_mount_synthetic_intrinsics` | 実機で確認した設置値（水平中央、上15 cm、前後0 cm、下向き12°） |

`user_confirmed_mount_synthetic_intrinsics` は「設置は実機確認済みだが、カメラ内部パラメータ
`K, D` と表示サイズはまだdemoのプレースホルダ」を表します。全項目が実測になるまで `measured`
を名乗らせないための区別です。`forward_offset_m = 0.0` は録画検証用の明示値であり、実レンズ中心が
画面より手前であれば実カメラ受け入れ前に実測して更新します。

## 3.1 シーンプロファイル

```json
{
  "schema_version": 1,
  "scene_id": "bunny-on-display-plane",
  "point_cloud_asset": "/static/assets/bunny.pcd",
  "anchor_display_m": [0.0, 0.0, 0.0],
  "longest_edge_m": 0.24,
  "grid_spacing_m": 0.05,
  "back_wall_z_m": -0.3,
  "floor_y_m": -0.14,
  "floor_near_z_m": 0.05,
  "floor_far_z_m": -0.3
}
```

較正結果ではなく表示上の選択であるため、ハードウェアプロファイルとは別ファイルに保ちます。
探索順は `HEADCOUPLED_SCENE` → `./config/scene_profile.default.json` → パッケージ同梱リソース
の明示順で、カレントディレクトリへの暗黙fallbackは作りません。

拒否条件（いずれも明示エラー）:

- 未知のキー
- `longest_edge_m` または `grid_spacing_m` が0以下
- `back_wall_z_m` が0以上（後壁が画面より観察者側にある）
- `floor_far_z_m >= floor_near_z_m`（床が観察者側へ伸びている）
- `floor_far_z_m < back_wall_z_m`（床が後壁を越えている）

## 4. 利用者プロファイルの個人顔モデル

`UserProfile.face_model_path` は `facemesh_tracking reconstruct` の478点ASCII `shape.pcd` を指す任意項目です。
相対パスは利用者プロファイルJSONから解決します。PCDはmm/OpenCV座標（+X右、+Y下、+Z奥）であり、
実行時にhead frameへ正規化してから12点SQPNPへ渡します。canonical faceとのKabsch照合で、proper
rotation、8°以下の向き差、15 mm以下のRMSを満たさないファイルは拒否します。指定時の左右眼位置は
虹彩中心ランドマーク468/473、両眼中点はその平均です。

## 5. TrackingState

```json
{
  "sequence": 42,
  "timestamp_unix_s": 1786932091.7,
  "source": "synthetic | facemesh | replay",
  "confidence": 0.99,
  "cyclopean_eye_display_m": [0.02, 0.01, 0.67],
  "left_eye_display_m": [-0.012, 0.01, 0.67],
  "right_eye_display_m": [0.052, 0.01, 0.67],
  "head_forward_display": [0, 0, -1],
  "tracking_fps": 30,
  "inference_ms": 1.2,
  "stable": true,
  "diagnostics": {}
}
```

## 6. HTTP API

### `GET /api/profile`

ハードウェア・利用者プロファイル、変換行列から計算した設置サマリー、座標系、人工値警告を返します。

### `POST /api/calibration/synthetic`

決定的人工データを生成し、意図的にずらした初期姿勢から外部姿勢を復元します。

### `POST /api/calibration/fit`

`CalibrationDataset` を受け、外部姿勢を推定します。

最低条件:

- 10標本以上
- 5画面点以上
- `target_uv` は0〜1
- 方向ベクトルは正規化可能

### WebSocket

- `/ws/pose`: `{type: "tracking", payload: TrackingState}`
- `/ws/camera`: JPEG bytes

## 7. 較正品質基準

人工回帰試験の受入基準:

- 平均点・レイ残差 < 2.5 mm
- 高さ誤差 < 0.5 mm
- 全並進誤差 < 1.5 mm
- ピッチ誤差 < 0.35°
- 9画面点、36標本

実機では人工基準をそのまま保証値にせず、次を記録します。

- tagcal RMSと内部パラメータ標準偏差
- 外部較正の平均・中央値・最大レイ残差
- 中央・四隅・上下左右別残差
- 静止時眼位置ジッター
- 前後移動時スケールドリフト
- bunny.pcdの窓効果を見た目で検証した結果

## 8. 異常系

- 眼位置 `z <= 0`: 投影拒否。
- 変換回転が直交でない: プロファイル拒否。
- PCDがASCIIでない: 表示エラー。
- FaceMesh未導入: 合成モードを案内する明示例外。
- replay JSONが連番でない、ランドマークが不足・非有限、映像解像度/フレーム数が不一致:
  起動前に拒否。`replay` CLIでは個人 `shape.pcd` とtagcal `K,D` の指定も必須。
- カメラ未接続: ランタイム状態へエラーを出し、API自体は維持。
- WebSocketクライアント遅延: 古いフレームを蓄積せず最新値へ追従。
- WebGL2不可: Canvas2Dへ切替。

## 9. 実機化時の追加作業

- 対象カメラ解像度・フォーカス・ズーム固定。
- 対象ディスプレイの有効表示寸法を定規で確認。
- 実機tagcal結果を取り込む。
- カメラ光学中心の高さ・前後位置を測定、または5/9点外部較正を実行。
- 利用者ごとの眼球中心・IPD・正面軸を較正。
- GPU/CPUの実測レイテンシと追跡ジッターを計測。
- 全画面表示時にOSスケーリング、ブラウザー倍率、ディスプレイ選択を固定。
