# アーキテクチャ

## 1. 目的

観察者の頭部移動に応じて、ディスプレイ面を「窓」として見たときの正しい透視投影を
リアルタイム生成します。通常実行時に必要なのは両眼中点の3次元位置です。

## 2. 論理構成

```mermaid
flowchart LR
    CAM[USB Camera] --> FM[Face detector / FaceMesh 478]
    FM --> PNP[Metric head pose solvePnP]
    PNP --> EYE[Left / right eye centers]
    SYN[Synthetic provider] --> STATE[TrackingState]
    EYE --> TSC[T_display_camera]
    TSC --> STATE

    STATE --> POSEWS[/ws/pose JSON]
    CAM --> JPEG[JPEG encoder]
    SYN --> JPEG
    JPEG --> CAMWS[/ws/camera binary]

    POSEWS --> VIEW[Browser controller]
    VIEW --> FRUSTUM[Asymmetric frustum]
    PCD[bunny.pcd] --> LOADER[ASCII PCD loader]
    LOADER --> RENDER[WebGL2 / Canvas2D]
    FRUSTUM --> RENDER

    TAG[tagcal K,D] --> PROFILE[HardwareProfile]
    TARGET[5/9 display targets] --> SOLVER[Head-ray least squares]
    SOLVER --> PROFILE
    PROFILE --> TSC
```

## 3. 座標系

### 3.1 ディスプレイ座標系 S

- 原点: 有効表示領域の中央
- +X: 観察者から見て右
- +Y: 上
- +Z: ディスプレイ面から観察者側
- 画面平面: `z = 0`

### 3.2 カメラ座標系 C

OpenCV規約を使用します。

- +X: カメラ画像右
- +Y: カメラ画像下
- +Z: カメラ光軸前方（観察者側）

カメラが正面・無傾斜の場合、カメラ画像右は観察者の左、画像下は画面下になるため、
カメラ軸をSで表す回転は概ね `diag(-1, -1, +1)` です。

### 3.3 変換

`T_S_C` をカメラからディスプレイへの剛体変換とします。

```text
p_S = R_S_C p_C + t_S_C
```

プロファイルの `camera_to_display_matrix` はこの `T_S_C` です。
逆変換は次です。

```text
R_C_S = R_S_C^T
t_C_S = -R_S_C^T t_S_C
```

## 4. カメラ設置パラメータ

`CameraMount` から `T_S_C` を構成します。

```text
t_S_C = [左右オフセット, 中央からの高さ, 画面より手前]
```

光軸ベクトル `z_C` をSで表すと、下向きピッチ `p`、右向きヨー `y` に対して:

```text
z_C^S = [sin(y) cos(p), -sin(p), cos(y) cos(p)]
```

変換行列から表示する値:

```text
height_cm = 100 * t_y
forward_cm = 100 * t_z
pitch_down = atan2(-z_y, sqrt(z_x^2 + z_z^2))
total_tilt = acos(dot(z_C^S, display_normal))
```

左右中央判定は既定で `|t_x| <= 5 mm` です。

## 5. 頭部・眼位置推定

実入力時の処理:

```text
BGR frame
  → YOLOv8-Face
  → FaceMesh 478 points
  → bone-backed 12 landmark subset
  → SOLVEPNP_SQPNP + cheirality (tvec.z > 0)
  → T_C_H
  → reconstructed shape.pcd iris centres (468/473), or profile fallback
  → left/right/cyclopean eye in C
  → T_S_C
  → eye positions in S
```

標準点は鼻・両眼角・口角・顎など表情で動きにくい12点です。個人 `shape.pcd` が指定されると、
起動時にcanonical head frameとのKabsch照合を行い、PCDのOpenCV/mm座標を内部head frameへ
正規化します。照合失敗（反射、軸違い、単位違い）はカメラ開始前に拒否します。

## 6. 非対称投影

観察者の両眼中点 `e = [e_x, e_y, e_z]`、画面幅 `W`、高さ `H`、near面 `n` とします。

```text
left   = n (-W/2 - e_x) / e_z
right  = n ( W/2 - e_x) / e_z
bottom = n (-H/2 - e_y) / e_z
top    = n ( H/2 - e_y) / e_z
```

標準OpenGL frustum行列をこの非対称境界で作ります。ビュー行列は画面軸を固定したまま
`translate(-e)` です。仮想物体は画面後方の `z < 0` に置きます。

Canvas2D代替経路では、眼 `e` と仮想点 `p` を結ぶ線が画面平面 `z=0` と交わる位置を
直接計算します。

```text
ratio = e_z / (e_z - p_z)
screen_xy = e_xy + ratio * (p_xy - e_xy)
```

これは同じピンホール幾何です。

## 7. 外部姿勢較正

### 7.1 観測

各標本は次を持ちます。

```text
target_uv                 既知画面点
cyclopean_eye_camera_m    カメラ座標のレイ始点
head_forward_camera       カメラ座標の単位方向
confidence                標本重み
```

画面点 `q_S` を候補 `T_C_S` でカメラ座標へ変換し、レイとの直交距離を残差にします。

```text
q_C = R_C_S q_S + t_C_S
r = (I - d d^T) (q_C - e_C)
```

### 7.2 最適化

- SciPy `least_squares`
- `soft_l1` robust loss
- 変数: `T_C_S` の回転ベクトル3、並進3
- 弱い事前分布: 初期カメラ設置値
- 最低10標本、最低5画面点
- 推奨: 9画面点、各点複数の頭部位置

出力は明示的な `camera_to_display_matrix`、可読な設置値、レイ残差、角度誤差です。

## 8. プロファイル分離

### HardwareProfile

- ディスプレイ解像度・実寸
- カメラ内部パラメータ `K, D`
- カメラ設置事前値
- カメラ→ディスプレイ4×4行列
- 品質指標、出自

### UserProfile

- 瞳孔間距離
- 頭部座標内の左右眼球中心・両眼中点
- 任意の個人用478点 `shape.pcd` パス（指定時は虹彩中心を眼位置に使う）
- 頭部正面軸補正
- 個人較正品質

カメラを動かした場合はHardwareProfileを更新し、利用者だけが変わる場合はUserProfileだけを
切り替えます。

## 9. 実行時通信

### `/ws/pose`

JSONのlatest-value配信です。遅延を溜めないため、各クライアントは新しいsequenceだけを
受け取ります。

### `/ws/camera`

JPEGバイナリです。Base64化せず、姿勢配信と分離します。

### FastAPI runtime

追跡は専用バックグラウンドタスクで実行し、最新姿勢と最新JPEGだけを保持します。
永続記録は行いません。

## 10. 配置

```text
headcoupled-3d-display/
├── src/headcoupled_display/      制御/API/較正/表示
├── config/                       ハードウェア・利用者プロファイル
├── components/tagcal/            添付内部較正ツール
├── components/facemesh_tracking/ 添付FaceMeshツール
├── tests/                        単体/API/E2E
├── scripts/                      生成・試験・代替タスク
├── docs/                         仕様と調査記録
└── artifacts/                    試験結果
```
