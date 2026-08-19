# 実装計画: 実測較正・SE(3)姿勢フィルタ・点群配置拡張（改善項目1〜5）

対象: `~/Downloads/geometry_6dof_review.md`（= `headcoupled_geometry_6dof_review.md`、同一内容）の
改善項目 **1〜5**。項目6（WebGL点サイズの距離連動）と項目7（表示更新時刻への補間・予測）は**本計画の対象外**。

SE(3)フィルタは J. Solà, J. Deray, D. Atchuthan, "A micro Lie theory for state estimation in robotics"
(arXiv:1812.01537, v9) を参照実装の根拠とする。同論文に付属の C++ ライブラリ `manif` が参考実装。

最終更新: 2026-08-19

---

## 0. 対象項目と到達目標

| # | 項目 | 主な変更ファイル |
|---|---|---|
| 1 | カメラ・ディスプレイ・顔モデルのメートル系較正を実測値へ置き換える | `config/hardware_profile.local.json`, `api.py`, `profiles.py`, `tracking.py`, `app.js` |
| 2 | 生PnP結果を剛体姿勢 `SE(3)` として品質判定・平滑化・短時間予測する | **新規** `lie.py`, **新規** `filtering.py`, `tracking.py`, `models.py` |
| 3 | 平滑化・予測後の頭部姿勢から左右眼位置を再計算する | `tracking.py`, `models.py` |
| 4 | 点群モデル変換を「単位・向き・意味的ピボット・奥行き配置」へ拡張する | `models.py`(SceneProfile), `scene.py`, `static/renderer.js`, `static/app.js`, `pcd.js` |
| 5 | 通常表示は画面奥へ、画面貫通は検証用シーンとして分離する | `config/scene_profile.*.json`, `api.py`, `static/app.js`, `static/renderer.js` |

依存関係: 1 → 2・3（較正値が姿勢品質の前提）→ 4・5（姿勢が正しくなってから配置・演出を調整）。
項目4・5は項目2・3と独立に着手できるが、見た目の調整が誤差を相殺しないよう、**2・3の後に目視評価する**。

---

## 1. 共通の不変条件（壊さないライン）

- 座標系の規約は変えない。ディスプレイ座標 S（原点=表示領域中央、+X=観察者の右、+Y=上、+Z=観察者側、画面 z=0）、
  カメラ C は OpenCV 規約、頭部 H は +Y=上・+Z=顔前方。`HEAD_TO_OPENCV = diag(1,-1,-1)` (`face_model.py:27`) を維持。
- 較正値（HardwareProfile）と演出値（SceneProfile）の分離を維持。奥行きゲインや色は SceneProfile 側。
- 内部計算は 4×4 行列 or 正規化四元数。**オイラー角はUI表示専用**（既存 `geometry.py` の mount サマリはこのまま）。
- 閾値の緩和・未達の pass 扱いはしない。測れない項目は「未検証」と記録する（`performance_results.md` の方針を踏襲）。
- 生体情報（`shape.pcd`、録画、撮影フレーム）はリポジトリへ入れない。

---

## 2. 項目1: メートル系較正の実測化

### 2.1 現状（プレースホルダの所在）

| 項目 | 現状の値 | 場所 | 問題 |
|---|---|---|---|
| カメラ内部パラメータ K,D | fx=fy=950, cx=640, cy=360, dist=[-0.031,...] | `config/hardware_profile.local.json` の `camera` | デモ値。実測 tagcal は fx=1150.77, cx=719.85, RMS 1.0429 px（`docs/local_setup_notes.md` §4.1） |
| ディスプレイ実寸 | 0.596 × 0.335 m, 2560×1440 | 同上 `display` | demo から未確認のまま継承 |
| カメラ前後オフセット | `forward_offset_m = 0.0` | 同上 `camera_mount` | 録画検証用の明示プレースホルダ |
| 外部姿勢 T_S_C | `camera_to_display_matrix` 未設定 → `camera_mount` から構成 | `profiles.py:22` `resolved_camera_to_display` | 高さ15 cm・俯角12°は目視確認のみ |
| 顔モデルスケール | `UserProfile.ipd_m`(0.064) が**未参照**、前方軸補正2項目が未使用 | `tracking.py:184-185,193` | `face_model.py:60` の canonical は cm 基準→m 変換済み。個人 PCD は mm→m 変換済み(`face_model.py:154`) |

`api.py:173` `_profile_warning` は `synthetic_demo_not_measured` と `camera_intrinsics_imported` のみ警告し、
`user_confirmed_mount_synthetic_intrinsics` を警告しない。UI 上でプレースホルダ状態が見えにくい。

### 2.2 変更内容

**a. ディスプレイ実寸（`config/hardware_profile.local.json` の `display`）**
- 実測手順: 定規で有効表示領域（フルスクリーン時に実際に描画される矩形）の幅・高さを測り、
  ピクセル解像度も実運用値（OSスケーリング・ブラウザ倍率を考慮）へ合わせる。
- `docs/local_setup_notes.md` §4.2「未検証」→ 実測値に置き換えたら `quality_metrics` に測定日時・方法を追記。
- `app.js:12` の `PHYSICAL_ASPECT_RATIO = 0.596 / 0.335` 固定値を撤去し、
  `loadProfile()` (`app.js:27-55`) で受けた `profile.display.width_m / height_m` をモジュール変数へ保持し、
  `updateAspectState()` (`app.js:206`) がそれを使うようにする。

**b. カメラ内部パラメータ（tagcal）**
- 実測: `apriltag-camera-calibrator` の `calibration.json`（1280×720, plumb_bob, RMS 1.0429 px）。
- 取り込み経路は既存: `just import-tagcal` / `api.py:150` `_with_runtime_intrinsics` → `profiles.py:52` `load_tagcal_calibration`。
- 受け入れ時に確認すべき点（`docs/local_setup_notes.md` §4.2）:
  - 主点 cx=719.85 が画像中心 640 から 80 px ずれている → 意図的/測定誤差かを再確認。
  - 録画時 focus=332 と較正時 focus=256 の不一致 → 実カメラ運用時に同じ固定焦点で取り直す。

**c. カメラ・ディスプレイ外部姿勢 T_S_C**
- 選択肢（実測難度順）:
  1. メジャーで高さ・前後・俯角・ヨー・ロールを測り `camera_mount` を更新。
  2. 5点/9点の頭部レイ較正（`api.py` `/api/calibration/fit`、`calibration.py` 実装済み）で
     `camera_to_display_matrix` を推定し `config/hardware_profile.local.json` へ書く。
- 重要: 外部姿勢 1° の誤差は 0.67 m 先で約 12 mm の眼位置誤差、5 mm の並進誤差はそのまま眼位置誤差
  （`geometry_6dof_review.md` §4.1）。**この工程が項目2・3の精度上限を決める**。
- `forward_offset_m = 0.0` は実レンズ中心を測って置き換える（実機ではレンズが画面より手前にあるはず）。

**d. 顔モデルのメートルスケール（`tracking.py`・`UserProfile`）**
- `models.py:139` `ipd_m` を実装に接続する。canonical フォールバック時は
  `face_model.py:60` の canonical 全体を `ipd_m / canonical_ipd` で等方スケールしてから
  PnP 物体点と眼オフセットの**両方**に適用する（`geometry_6dof_review.md` §4.2: 眼だけ拡大すると不整合）。
  - ただし `face_model.py:60` はモジュールキャッシュなしで毎回ロードするので、スケールは
    `HeadPoseEstimator.__init__` (`tracking.py:167`) 内で適用する。
- **近似性の明示**: canonical 平均顔を `ipd_m` で等方スケールするのは「顔の比率が IPD に比例する」
  仮定の近似であり、真の実測は個人 `shape.pcd`（Kabsch 検証済み）。canonical フォールバック時は
  `UserProfile.notes` へ「距離推定は近似」と明示する（`geometry_6dof_review.md` §4.2 の指摘）。
- `models.py:147-148` `forward_axis_yaw_correction_deg` / `forward_axis_pitch_correction_deg` を
  `tracking.py:193` の `_forward_axis_opencv` 構築時に適用する（現在未使用）。
- 個人 `shape.pcd` 指定時（`face_model.py:146`）は Kabsch 検証済みの実測メッシュが単一情報源。
  `UserProfile.ipd_m` は personal 時には使用しない（警告を `notes` に出すのみ）。

**e. 出自・警告の強化（`api.py:173` `_profile_warning`）**
- `provenance == "user_confirmed_mount_synthetic_intrinsics"` のとき、
  「カメラ内部パラメータ・ディスプレイ寸法は未実測プレースホルダ」という警告を返す分岐を追加。
- 警告文字列は既存の `_with_runtime_intrinsics` 分岐より**後**に評価し、tagcal 取り込み済みなら
  内部パラメータの警告を省略して外部姿勢の警告だけ残す。

### 2.3 検証
- `tests/unit/test_profiles.py` に「`user_confirmed_mount_synthetic_intrinsics` で警告が出る」ことを追加。
- `tests/unit/test_tracking.py` に `ipd_m` スケール適用時の PnP 物体点と眼オフセットの整合性テストを追加。
- 実測後 `just profile-summary` で `camera_to_display_matrix` から逆算した設置値が測定値と一致すること。

---

## 3. 項目2: 生PnP → SE(3)品質判定・平滑化・短時間予測

### 3.1 現状
- `tracking.py:195` `HeadPoseEstimator.estimate` は**姿勢を返さず**眼位置4配列だけ返す。
  PnP は `cv2.solvePnP(..., SOLVEPNP_SQPNP)` (`tracking.py:209`)、cheirality は `tvec.z > 0` のみ。
- 再投影RMS・外れ値ゲート・時系列平滑化は**無し**。
- `tracking.py:588` `sample` はスコア最大の顔を選択（`tracking.py:592`）。複数顔で対象が切り替わりうる。
- `tracking.py:692` `stable` は `movement < 4mm && confidence >= 0.75` の簡易判定。

### 3.2 新規モジュール `src/headcoupled_display/lie.py`（論文の「micro Lie theory」）

論文 `arXiv:1812.01537` の適用範囲（v9 の節番号）:

| 機能 | 論文の箇所 | 実装 |
|---|---|---|
| SO(3) 指数/対数写像 | §5.1–5.2, eq.(16)–(24) | `log_so3`/`exp_so3`（Rodrigues 公式、θ≈0 の級数展開含む） |
| se(3) 指数写像（V 行列） | §6.1, eq.(55)–(58) | `exp_se3`（`T = [R, Vρ; 0 1]`） |
| se(3) 対数写像 | §6.1, eq.(59)–(60) | `log_se3`（θ≈0 の V⁻¹ 級数） |
| 右ヤコビアン J_r(θa) | §7.4, eq.(101)–(104) | `right_jacobian_so3` |
| SE(3) 随伴表現 Ad_T | §6.1, eq.(61) | `adjoint_se3`（twist の座標変換用） |
| ボックスプラス/マイナス ⊞/⊖ | §4, eq.(10)–(12) | `oplus(T, xi) = T @ exp(xi)`、`ominus(Y, X) = log(X⁻¹ @ Y)` |
| 姿勢+速度の EKF 例 | 「Example: EKF」節（v9 修正 Ex.5） | 下記 §3.5 のマニフォールド EKF |

規約は論文と同じく**右摂動**（`X ⊞ u = X exp(u)`）。回転の正規化・直交化チェックは
`geometry.py:75` `invert_rigid_transform` の流儀（許容誤差 1e-7）に合わせる。

### 3.3 `HeadPoseEstimator` の返却値拡張（`tracking.py:158`）

新設: 値オブジェクト `HeadPoseEstimate`（`@dataclass(frozen=True)`）
- `T_S_H`（4×4: 表示座標で表した頭部姿勢。`R_S_H = R_S_C @ R_C_H`, `t_S_H = R_S_C t_C_H + t_S_C`）
- `reprojection_rms_px`（12点を `cv2.projectPoints` で再投影し `image_points` と比較。`tracking.py:206-215` の直後で計算）
- `inlier_count`（残差 ≤ 閾値の点数。閾値は `UserProfile` か `HardwareProfile` の `quality_metrics` で指定、既定 3 px）
- `landmark_rms_px_per_point`（デバッグ用、`diagnostics` へ）
- 既存の `left/right/cyclopean/forward`（互換性のため残す）
- `timestamp_unix_ns`

`estimate()` のシグネチャは互換のため**維持**し、新設 `estimate_pose()` を本命とする。
既存テスト (`tests/unit/test_tracking.py`) は `estimate()` のまま通る。

### 3.4 品質ゲート（`tracking.py` の `sample` 周辺）

`FaceMeshPoseProvider.sample` (`tracking.py:588`) に以下を追加:

1. **再投影RMSゲート**: `reprojection_rms_px > 上限`（既定 4 px、`quality_metrics` で可変）なら姿勢を拒否し、
   最後の正常姿勢を保持して `confidence=0`・`diagnostics.rejection="reprojection_rms"` を出す。
2. **速度ゲート**: 前回フィルタ姿勢からの速度が物理的に不可能（線速度 > 1.5 m/s、角速度 > 4 rad/s 程度）なら拒否。
3. **複数顔の継続選択**: `tracking.py:592` の「スコア最大」をやめ、
   「前回予測姿勢から投影した予測眼位置との距離 + スコア」の複合で顔を選ぶ（`geometry_6dof_review.md` §4.3）。
4. **renderer 側の整合**（`renderer.js:131`）: `projectionForDisplay` の `Math.max(eye[2], 0.2)` による
   部分クランプをやめ、`TrackingState.tracking_valid` が false なら**姿勢全体を無効化**して最後の正常姿勢を保持する。

### 3.5 SE(3) フィルタ（新規 `src/headcoupled_display/filtering.py`）

**クラス**: `Se3PoseFilter`
**状態**: `X = (T_S_H, v_body)`。`v_body ∈ ℝ⁶`（se(3) のボディ系 twist `[ω; u]`）を並進/回転速度として持つ。
**参照**: 論文の EKF 例のマニフォールド化（predict は流れ場で T を更新、update はタンジェント空間で行う）。

predict（Δt、減衰 γ）:
- `v' = γ v`（γ≈0.95）
- `T' = T @ exp_se3(v_body * Δt)`（論文の `X ⊞ (v Δt)`）

correct（観測 `T_raw = T_S_H`、観測ノイズ R）:
- イノベーション `z = ominus(T_raw, T') = log(T'⁻¹ @ T_raw) ∈ se(3)`（右摂動、ボディ系）
- カルマンゲイン K（論文のヤコビアン J_r で線形化。簡易版では回転・並進を別ゲイン α_rot, α_pos でも可）
- `T_new = T' @ exp_se3(K_rot な z)`、`v_new = v' + K_vel z`

**実装上の落とし穴**（論文で特に注意とされる点）:
- `exp/log` の θ→0 特異点（級数展開）。
- 左右摂動の混同（本計画は**右摂動**で統一。`z` は必ず `T'⁻¹ @ T_raw` の対数）。
- 回転の正規化漏れ（`R = (R Rᵀ)⁻¹/² R` の直交化を update 後に1回）。

**短時間予測**（サーバー側）:
- `t_pred = t_pose + prediction_horizon_s`。horizon は既定 1フレーム分 ≈ 0.033 s、上限 0.1 s でクランプ。
- `T_pred = T_new @ exp_se3(v_new * clamp(horizon - elapsed, 0, horizon))`
- **入力が stale したら予測を止める**（`runtime.py:114` `_publish_stale` の既存経路が最後の姿勢を保持）。
- 注: 表示時刻への**毎フレーム補間・予測は項目6（対象外）**。ここでは1フレーム先固定予測のみ。

**初期化**: 最初の有効観測で `T = T_raw`、`v = 0`。最初の ~5 フレームは smooth 利得を上げて収束を速める。

### 3.6 配線（`tracking.py`）

- `FaceMeshPoseProvider.__init__` (`tracking.py:558`) で `Se3PoseFilter` を生成。
- `sample()` → `estimate_pose()` → ゲート → `filter.predict/correct` → 予測姿勢 `T_pred` を得る。
- `_build_state` (`tracking.py:662`) へ `T_pred` を渡す（眼位置は §4 で `T_pred` から再計算）。

### 3.7 `TrackingState` 拡張（`models.py:173`）

追加（すべてオプション/デフォルト付き。`StrictModel` の `extra="forbid"` と互換）:
- `pose_timestamp_unix_ns: int | None`
- `head_position_display_m: Vector3 | None`（頭部原点、`T_S_H` の並進）
- `head_orientation_display_xyzw: tuple[float,float,float,float] | None`（正規化四元数）
- `linear_velocity_display_m_s: Vector3 | None`
- `angular_velocity_display_rad_s: Vector3 | None`（表示座標系）
- `reprojection_rms_px: float | None`
- `inlier_count: int | None`
- `tracking_valid: bool`（renderer がこの姿勢を描画して良いか。false なら最後の正常値保持）
- `predicted_to_unix_ns: int | None`（サーバー側予測の適用時刻。ブラウザーはこの時刻を描画時刻と比較可能）

`cyclopean/left/right_eye_display_m` は互換性のため残すが、単一情報源は `T_S_H` + 眼オフセットにする。

### 3.8 検証
- `tests/unit/test_lie.py`（新規）: exp∘log の恒等、⊞/⊖ の可逆性、Ad_T の合成、特異点 θ≈0・θ≈π。
- `tests/unit/test_filtering.py`（新規）:
  - 定速度運動の追従（一定 twist を与え、遅延が一定で追従すること）。
  - 静止時ジッター低減（ノイズ付き静止観測 → 出力分散が観測より小さい）。
  - 外れ値1点 → 拒否され姿勢が跳ばない。
  - stale 後の予測停止・姿勢保持。
- `tests/unit/test_tracking.py`: `estimate_pose()` の返却値・reprojection RMS 計算・複数顔選択。
- 録画ベースライン (`just benchmark-recorded`) で平滑化前後のジッター・位相遅れを比較記録する。

---

## 4. 項目3: 平滑化・予測後姿勢から左右眼位置を再計算

### 4.1 現状
- 眼位置は `HeadPoseEstimator.estimate` 内の**生PnP姿勢**で計算される（`tracking.py:221-231`）。
  フィルタを挟んでも `sample()` はそのまま使うため、眼位置だけが平滑化されない。

### 4.2 変更
- `HeadPoseEstimator` に `eyes_from_pose(T_S_H) -> (left, right, cyclopean, forward)` を新設。
  実装は `tracking.py:221-231` の S←H 変換をそのまま移す（`left_eye_head_m` 等は `tracking.py:180-186` の既存値を保持）。
- `sample()` はフィルタ出力 `T_pred` に対して `eyes_from_pose(T_pred)` を呼ぶ。**生の `estimate` の眼位置は使わない**。
- `_PoseMeasurement` (`tracking.py:49`) に `T_S_H`・`reprojection_rms_px` を追加（または別構造へ置換）。
- `_build_state` (`tracking.py:662`) は §3.7 の新フィールドと、`eyes_from_pose` の結果を併記する。

### 4.3 効果と検証
- 剛体変換なので IPD が厳密に保存される（左右を別々にローパスした場合の IPD ゆらぎが消える）。
- 予測により眼位置が「頭部の実位置」から display までの遅延分だけ進む。
- `tests/unit/test_tracking.py`: 同一 `T_S_H` から出した左右眼間距離が `ipd` と一致すること、
  予測 horizon を変えたとき眼位置が予測方向へ進むこと。

---

## 5. 項目4: 点群モデル変換の拡張

### 5.1 現状
- `scene.py:18` `scene_model_matrix` = `T(anchor) @ S(最長辺) @ T(-aabb_center)`。単位・向き・ピボット・奥行き演出が無い。
- `renderer.js:153` `modelMatrixForBounds` も同型。
- `renderer.js:649` Canvas2D 経路は `model[0]`（等方スケール）と `model[12..14]`（並進）のみを読み、
  **回転・非等方・ピボットに対応していない**（`geometry_6dof_review.md` §7 の指摘どおり）。

### 5.2 `SceneProfile` 拡張（`models.py:259`）

追加フィールド（すべてデフォルト付きで後方互換。`StrictModel` は未知キーを拒否するので既存JSONはそのまま通る）:

| フィールド | 型/既定 | 意味 |
|---|---|---|
| `placement_mode` | `"fit_longest_edge"` 既定 | `"fit_longest_edge"`（旧動作・既存シーン互換）/ `"metric"`（実寸優先・新規シーンが明示選択）。既定を `metric` にすると既存3シーンが `longest_edge_m` を無視して `uniform_scale=1.0` になりbunnyが0.156mで配置される**破壊的変更**になるため、既定は旧動作 |
| `asset_units_to_m` | float 1.0 | PCD 座標の単位→m 変換係数 |
| `asset_rotation_xyzw` | (0,0,0,1) | 点群固有の向き（アセット座標→表示向き） |
| `pivot_mode` | `"aabb_center"` | `aabb_center` / `aabb_bottom_center` / `explicit` |
| `pivot_asset` | Vector3 (0,0,0) | `explicit` 時のピボット（アセット座標、`units_to_m` 適用前） |
| `uniform_scale` | float 1.0 | 物理サイズ倍率（`metric` モードの主スケール） |
| `depth_gain` | float 1.0 | 奥行きのみの演出スケール（**較正値と分離**、`notes` に明記） |
| `point_radius_m` | float 0.0 | 予約（WebGL点サイズは対象外のため使用しない） |
| `content_response_mode` | `"physical"` | 予約（顔操作モードは対象外） |

### 5.3 行列の形（`scene.py:18` を拡張）

```
M = T(anchor_display_m)
  · R(asset_rotation_xyzw)
  · S(1, 1, depth_gain)
  · S(asset_units_to_m · uniform_scale)        # 等方（深度ゲインは分離）
  · T(-pivot)
```
- `pivot` は `pivot_mode` で決定: `aabb_center` = `(min+max)/2`（旧動作）、
  `aabb_bottom_center` = `((min.x+max.x)/2, min.y, (min.z+max.z)/2)`、`explicit` = `pivot_asset`。
- `fit_longest_edge` モードは旧式 `scale = longest_edge_m / longest_edge` を維持（検証シーンの互換）。
- `metric` モードでは `longest_edge_m` を**使用しない**。`uniform_scale` と `asset_units_to_m` で実寸を出す。
- AABB に単一外れ値があって全体が縮むのを避けるため、fit モードでは百分位範囲（例 1%〜99%）を
  オプション `aabb_quantile`（既定 None=全範囲）で使えるようにする。

### 5.4 renderer.js（WebGL / Canvas2D 両経路）

- `renderer.js:153` `modelMatrixForBounds` → `modelMatrixForScene(scene, bounds)` に置換し、
  四元数→行列、軸別スケール、ピボット、深度ゲインを合成。`this.model`（`renderer.js:420`）は一度だけ構築。
- `renderer.js:505` `zoomedModel()`: 現在は画面原点まわりで拡大。アンカーが原点でない場合に備え、
  `M_zoom = T(anchor) @ S(zoom) @ T(-anchor) @ M_base` にする（`geometry_6dof_review.md` §5.7）。
- `renderer.js:649` Canvas2D 経路: `model[0]`/`model[12..14]` の読み取りをやめ、
  4×4 の `transformPoint4x4`（`renderer.js:175` を拡張）で各点を変換する。Canvas2D と WebGL で同一行列を使い、
  **見え方が乖離しないこと**を E2E で担保（§7.4）。
- `renderer.js:131` の near/far はシーン AABB と眼位置から自動決定する（深度精度確保）。

### 5.5 検証
- `tests/unit/test_scene_profile.py`: 単位換算・回転・ピボット・深度ゲインの行列を逐次検証。
- `tests/e2e/test_browser.py`: `canvas.dataset.modelMin/MaxDisplayM` が、指定ピボット/アンカー/スケールの期待範囲に一致。
- ピボットがアンカーに一致すること、回転追加後も Canvas2D と WebGL の画面交差が一致すること。

---

## 6. 項目5: 通常表示（画面奥）と検証シーン（画面貫通）の分離

### 6.1 現状
- `config/scene_profile.default.json` は `anchor_display_m = (0,0,0)` → bunny が z∈[-0.093, +0.093] で**画面貫通**。
- `renderer.js:viewMode`（`app.js:193` `setViewMode`）は**HUD（画面枠）の表示/非表示だけ**を切り替え、点群配置は同じ。
- 検証用シーン（`scene_profile.depthramp.json` / `scene_profile.synthetic.json`）は存在するが配置は同一。

### 6.2 シーン構成

| モード | プロファイル | 配置 | 目的 |
|---|---|---|---|
| 検証 | 既存 `scene_profile.default.json`（`placement_mode: fit_longest_edge`） | 画面貫通 | 前景/背景の逆向き視差を確認 |
| 通常（没入） | **新規** `config/scene_profile.immersive.json` | 主に画面奥 | 調節・輻輳・運動視差の整合 |

没入シーンの初期値（`geometry_6dof_review.md` §5.6 の目安）:
- `placement_mode: "metric"`、`asset_units_to_m` + `uniform_scale` で実寸を保つ。
- `anchor_display_m.z ≈ -0.11` m、または `pivot_mode: aabb_bottom_center` で底を床へ接地。
- bunny 全体を概ね z∈[-0.20, -0.02] に収める（画面手前への突出を小さく）。
- `back_wall_z_m` 等はシーンごとに明示。

### 6.3 UI/API の切り替え

- 案A（推奨）: `app.js:setViewMode` (`app.js:193`) がモード切替時に
  `/api/profile?scene=<id>` を再取得し、`renderer.load()` を呼び直して点群・行列を差し替える。
  - `api.py` の `/api/profile` に `scene` クエリ（SceneProfile.id 一致 → 切替）を追加。
  - 同一プロファイル内の `anchor_display_m.z` だけ変えるより、**プロファイルを分ける**方が
    「表示上の選択」と「較正値」の分離原則に合う。
- 案B（簡易）: 起動時に `HEADCOUPLED_SCENE` で選択する従来方式のままとし、没入/検証は
  `just serve-ipc ... --scene` で切り替える。UI トグルは付けない。

本計画では案Aを推奨するが、実装量が増える場合は案Bで開始し、UI トグルは後続に回してもよい。

### 6.4 検証
- `tests/e2e/test_browser.py`: 没入シーンで `canvas.dataset.modelMin/MaxDisplayM` の z が全て負
  （画面奥）にあること。検証シーンでは z が正負にまたがること。
- `tests/api/test_api.py`: `/api/profile?scene=...` の切替と未知 id の 404。

---

## 7. 実装順序と依存

| 順 | 作業 | 依存 | 想定工数観点 |
|---|---|---|---|
| 1 | 項目1（較正の実測 + 警告強化 + `app.js` のアスペクト比プロファイル化） | なし（実測作業は要ユーザー協力） | 中（ハードウェア実測が律速） |
| 2 | `lie.py`（単体でテスト完結） | なし | 小 |
| 3 | 項目2（`HeadPoseEstimate` + ゲート + `Se3PoseFilter` + `TrackingState` 拡張） | 2 | 大（フィルタの検証が主） |
| 4 | 項目3（`eyes_from_pose` への差し替え） | 3 | 小 |
| 5 | 項目4（SceneProfile 拡張 + `scene.py` + renderer 両経路） | なし（3 と独立） | 中 |
| 6 | 項目5（immersive シーン + API 切替 + E2E） | 5 | 中 |
| 7 | 総合 E2E + 録画ベースライン再測定 | 1–6 | 中 |

順序上の注意: 項目4・5の「見た目の調整」は項目3の姿勢精度確定**後**に行う（誤差をシーン側で相殺しないため）。

## 8. 追加・変更するテスト一覧

| ファイル | 内容 |
|---|---|
| `tests/unit/test_lie.py`（新規） | exp/log 恒等、⊞/⊖ 可逆性、Ad 合成、θ≈0/π 特異点 |
| `tests/unit/test_filtering.py`（新規） | 定速度追従、静止時低減、外れ値拒否、stale 停止 |
| `tests/unit/test_tracking.py` | `estimate_pose`、reprojection RMS、複数顔選択、IPD 保存、予測方向 |
| `tests/unit/test_scene_profile.py` | 単位/回転/ピボット/深度ゲイン、`metric` vs `fit_longest_edge` |
| `tests/unit/test_profiles.py` | `user_confirmed_mount_synthetic_intrinsics` 警告 |
| `tests/api/test_api.py` | `/api/profile?scene=` 切替、未知 id |
| `tests/e2e/test_browser.py` | 没入/検証シーンの z 範囲、Canvas2D と WebGL の行列一致 |

## 9. 計測・受け入れ基準

- **項目1**: `camera_to_display_matrix` に実測が入り `provenance` が `measured`（または部分実測の明示）。
  `api.py` の警告がプレースホルダ残りをUIに出していること。
- **項目2・3**: 録画ベースラインで、平滑化なしに比べ静止時眼位置ジッターが低下し、
  頭部移動時の視差方向が不変（符号反転しない）であること。外れ値・顔切替で視点が跳ばないこと。
  予測 horizon は 0.1 s 上限でクランプされ、stale 時は保持されること。
- **項目4・5**: 没入シーンで全点が z<0、検証シーンで z が正負にまたがる。ピボットがアンカーと一致し、
  回転・単位を変えても WebGL と Canvas2D が同じ画面交差を返すこと。
- **未達の扱い**: 閾値（reprojection RMS 上限、速度上限、予測上限）は緩めない。
  測れない項目は `performance_results.md` 方式で未達/未検証として記録する。

## 付録: 論文 (arXiv:1812.01537) の主要式と本実装の対応

- **指数写像 SO(3)**: `exp(θa) = I + sinθ [a]× + (1−cosθ) [a]×²`（§5.1）
- **対数写像 SO(3)**: `log(R) = θ/(2sinθ) (R − Rᵀ)∨`、`θ = acos((tr R − 1)/2)`（§5.2）
- **se(3) 指数**: `T = [exp(φ), Vρ; 0, 1]`、`V = I + (1−cosθ)/θ² [φ]× + (θ−sinθ)/θ³ [φ]×²`（§6.1）
- **右ヤコビアン**: `J_r(θa) = I − (1−cosθ)/θ² [a]× + (θ−sinθ)/θ [a]×²`（§7.4）
- **⊞/⊖**: `X ⊞ u = X exp(u)`、`Y ⊖ X = log(X⁻¹Y)`（§4、右摂動規約）
- **EKF の流儀**: 予測は流れ場で状態を更新し、更新はタンジェント空間の誤差（⊖）でカルマンゲインを掛け、
  指数写像でマニフォールドへ戻す（論文「Example: EKF」、v9 で Ex.5 を修正）。
- 実装の参考は論文付属の C++ テンプレートライブラリ `manif`（`SE3d`, `SO3d` の `exp/log`, `boxplus/boxminus`, `adj`）。

本計画では Python での実装のため、`lie.py` に上記の最小限の関数群を持ち、`scipy.spatial.transform.Rotation`
の `from_matrix/to_quat` を四元数変換に使用する（scipy は既存依存）。

---

## 10. 自己レビューと改善事項（2026-08-19 追記）

上記 §1–9 をセルフレビューした結果、以下を見出した。重大度順。

### A. 設計の誤り（修正済み）

1. **`SceneProfile.placement_mode` の既定値が破壊的だった** — 当初 `metric` を既定にしたが、
   既存3シーン（default/depthramp/synthetic）は `longest_edge_m` 依存。`metric` 既定だと
   `uniform_scale=1.0`・`asset_units_to_m=1.0` になり、bunny が 0.156 m で配置される。
   → 既定を `fit_longest_edge`（旧動作）へ修正（§5.2 表に反映済み）。`metric` は新規シーンが明示選択。

### B. 統合の落とし穴（実装前に設計を確定すること）

2. **予測 horizon と時計同期が未解決** — サーバー側予測は「pose 時刻 → ブラウザ受信/描画時刻」
   の片道遅延分だけ進める必要がある。しかし `docs/performance_results.md` の時計不確かさ **5.5 ms
   (>2 ms)** が未解決で、producer Unix 時刻とブラウザ時刻の差を 2 ms 未満で確定できない。
   → 第1段は「既知の一定遅延」（capture+inference+IPC+WS の実測中央値、録画ベースラインから取得）
   を固定 horizon とし、時計同期が解決してから動的 horizon へ移行する。**horizon の動的推定は項目6（表示時刻補間）と等価なので、項目1–5では固定 horizon に留める。**

3. **フィルタ保持と stale 検出の協調** — `tracking_valid=false` で `provider.sample` が
   最後の正常姿勢を返し続けると、`runtime.py:83` `_run` は成功扱いとなり
   `_publish_stale` (`runtime.py:114`) が発火しない。死んだ producer が「完全に静止した観察者」
   として描かれ続ける（まさに ONBOARDING §2 が禁止する状態）。
   → 連続 N 回（既定 5）の invalid で `provider.sample` が `RuntimeError` を投げるか、
   `runtime.py` で「low-confidence 連続時間」を `stale_after_s` と同等に扱う分岐を追加。

### C. 過剰設計・優先度の見直し

4. **マニフォールド EKF は重すぎる可能性** — 状態次元 12（SE(3)+ℝ⁶）の EKF は
   プロセス/観測ノイズ共分散の同定が難しく、30 Hz・~33 ms レイテンシでは過剰。
   → **第1段は「manifold EMA + 速度予測」**（論文の ⊞/⊖/exp/log は使うが EKF は組まない）。
   `T_smooth = T_prev ⊞ (α (T_raw ⊖ T_prev))`、`v = EMA(log(T_prev⁻¹T_raw)/dt)`、
   `T_pred = T_smooth ⊞ (v · horizon)`。EKF 化は第2段（オプション）。
5. **複数顔の継続選択** — 個人機では複数顔は稀。→ 第1段は「スコア最大＋前回予測との距離ゲート」
   のみ。顔 ID 維持は第2段。

### D. 設計の抜け（第1段に含める）

6. **適応ゲイン** — `correct` のゲインを `reprojection_rms_px` で重み付け（RMS 低→信頼大）。
7. **速度の平滑化** — PnP 差分の角速度はノイジー。`v` も EMA/LPF（damping γ だけでは足りない）。
8. **reprojection 閾値の測定先行** — 既定 4 px は根拠不足。→ 録画ベースラインで RMS 分布を
   採取してから閾値を決める（`just benchmark-recorded` の raw に含まれる）。
9. **NFR: フィルタ時間予算** — per frame < 0.5 ms（`cv2.projectPoints` 12点含めて実測）。
10. **renderer の tracking_valid 連動** — `renderer.js:131` の `Math.max(eye[2],0.2)` clamp 撤去
    だけでなく、`setEye` (`renderer.js:475`) が `tracking_valid=false` を無視して `this.eye` を保持する
    経路を明示。`TrackingState.tracking_valid` を WS ペイロードへ載せ、`app.js:116` から渡す。

### E. 軽微・明確化

11. **`depth_gain` の座標系** — `M=T(anchor)·R·S(1,1,depth_gain)·S(units·uniform)·T(-pivot)`
    において `depth_gain` は R 後（表示座標 z）に作用。これが意図（奥行きを表示系 z で圧縮）。
    一文化して `SceneProfile.notes` に明記。
12. **scene 切替の状態管理** — `scene` は renderer 専用。`/api/profile?scene=` は `SceneProfile`
    だけ差し替え、`RuntimeCoordinator`（追跡）は触らない。renderer は `buildStaticGeometry`
    (`renderer.js:351`) を含めて再構築するので、`renderer.load()` の再呼び出しで完結する。
13. **論文の式番号** — 本文中の節番号・式番号は記憶照合のため、実装時は「右ヤコビアン」
    「随伴」「⊞/⊖」など名称で `manif` 実装と突き合わせる。

### 改善の優先順位（計画をより良くするための追作業）

| 優先 | 件 | 対応 |
|---|---|---|
| **P0**（計画即修正） | A1 | `placement_mode` 既定を `fit_longest_edge` へ（**反映済み**） |
| **P0**（実装前確定） | B2 | 予測 horizon を固定値（実測中央値）に制限、動的 horizon は項目6へ分離 |
| **P0**（実装前確定） | B3 | フィルタ invalid 連続 → stale 発火の協調仕様を `runtime.py` に明文化 |
| **P1**（設計変更） | C4 | 第1段を manifold EMA + 速度予測へ格下げ、EKF を第2段化 |
| **P1**（設計変更） | C5 | 複数顔選択を第2段へ |
| **P1**（設計追加） | D6, D7, D8 | 適応ゲイン・速度平滑化・RMS 計測先行 |
| **P2**（実装中明文化） | D9, D10, D11 | 時間予算・renderer 連動・NFR |
| **P2**（明確化） | E11, E12, E13 | depth_gain 座標系・scene 状態管理・論文参照法 |

### レビュー総評

§1–9 の方針（座標規約維持・較正と演出の分離・SE(3) 表現・右摂動）は妥当。ただ
**第1段のフィルタは EKF ではなく manifold EMA + 速度予測**に格下げ、**予測 horizon は固定**
に制限し、**stale 協調**を先に固める方が、30Hz・未解決時計同期の現実に合う。EKF・動的 horizon
・顔ID維持は、時計同期（5.5 ms → 2 ms 以下）と item 6（表示時刻補間）の解決後に段階導入する。

> **第2段（EKF）実装済み（2026-08-19、`temp/workdoc_Aug19-2026_se3_ekf_stage2.md`）:**
> `right_jacobian_se3`（有限差分検証済み）と `Se3EKF`（状態次元 12、⊞/⊖ 形式、
> F=`[[Ad_{exp(-v·dt)}, dt·J_r(v·dt)^{-1}],[0,I]]`、R を reprojection_rms でスケール）を追加。
> 実測では定常速度追従誤差 0.0000 m（EMA 0.0067 m）・遅延誤差 0.0011 m（EMA 0.0040 m）で
> EKF が優位だったため、`quality_metrics["pose_filter"]` の**既定を `"ekf"` に設定**。
> 明示 `"ema"` で第1段 EMA に戻せる。`just check`（304 passed）・`just test-e2e`（13 passed）成功。