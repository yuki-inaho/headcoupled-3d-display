# 性能計測結果

本書は `temp/workdoc_Aug17-2026_headcoupled_scene_latency.md` の性能計測手順で得た実測値を追記していく記録である。
raw JSON は `.gitignore` 対象の `artifacts/perf/` 配下に保存され、本書には SHA-256・実行コマンド・commit・
主要 percentile・pass/fail 判定のみを追跡対象として記録する。raw JSON 自体は再実行すれば再生成できるため、
リポジトリには含めない。

## 計測ログ

### 手順7: 現行録画ベースライン — 2026-08-17T06:42:42.772858+00:00 (commit `4608c427ba57`)

- **コマンド:** `PYTHONPATH=/home/inaho-omen/Project/headcoupled-3d-display uv run python /home/inaho-omen/Project/headcoupled-3d-display/scripts/benchmark_recorded.py --video recordings/test10.avi --output /home/inaho-omen/Project/headcoupled-3d-display/artifacts/perf/baseline_recorded_raw.json`
- **commit:** `4608c427ba57ffe88c65e5dcb1131f0c7e48dc77`
- **入力:** `/home/inaho-omen/Project/facemesh_tracking/recordings/test10.avi` (1280x720, frame_count=294, warmup=5)
- **provider:** `CUDAExecutionProvider`
- **clock_domain:** `monotonic_ns` (uncertainty 0.000001 ms)
- **raw JSON:** `artifacts/perf/baseline_recorded_raw.json` (SHA-256: `219939e7aacc2fb9a32861e2d67f2fae38791b40f89cda475b95d88629a81b3b`)
- **欠測フレーム数:** 0
- **計測時刻 (created_at):** 2026-08-17T06:42:42.772858+00:00

- **参考桁確認:** detector + landmarks の p50 合計 42.73 ms は 既知の静止画ベンチ mean≈33.16 ms と同桁（比 1.29 倍）。過去値は成功判定の固定基準ではなく参考のみ。

| stage | sample_count | p50 (ms) | p95 (ms) | p99 (ms) |
| :--- | ---: | ---: | ---: | ---: |
| capture_decode | 289 | 8.735 | 10.927 | 17.026 |
| detector | 289 | 29.606 | 39.476 | 50.028 |
| landmarks | 289 | 13.121 | 19.231 | 24.040 |
| packet_build | 289 | 2.820 | 3.893 | 5.675 |
| preview_resize_encode | 289 | 2.965 | 4.466 | 6.577 |

**判定:** PASS — CUDA providerが実行中で、全段のp50/p95/p99が採取・検証された（このステップはベースライン記録であり、閾値ゲートは後続手順で行う）。

### 手順27: detector-refresh interval のスイープ — 2026-08-17

- **コマンド:** `just sweep-refresh`（3.10側）→ `scripts/analyze_refresh_sweep.py`（3.13側、実 intrinsics と個人メッシュを指定）
- **入力:** `recordings/test10.avi`（1280×720、実デコード294フレーム、warmup 5）
- **provider:** `CUDAExecutionProvider`（detector/estimator とも実セッション確認）
- **精度の基準:** `interval=1`（毎フレーム全検出）。**画素ではなくメートル系の眼位置と前方ベクトル**で比較（製品と同じ `HeadPoseEstimator`）
- **raw:** `artifacts/perf/refresh_sweep_raw.json`（SHA-256 `f1de61f374522ccbc2697c6017deba7601533032dd7f44d7c64f1d2d51e9d11b`）、
  `artifacts/perf/refresh_sweep_long_raw.json`（SHA-256 `cee5e27d531b7d2818b7eaa279ad17ee638d8c48160dd1d868741c8d0ed56678`）

| interval | 認識 p50 | 認識 p95 | 眼位置 p95 | 眼位置 max | 角度 p95 | 角度 max | 欠測 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1（基準） | 33.951 ms | 54.882 ms | 0 mm | 0 mm | 0° | 0° | 0 |
| 2 | 29.627 ms | 64.931 ms | **3.539 mm** | 6.883 mm | 1.018° | 1.447° | 0 |
| 3 | 17.345 ms | 55.221 ms | 3.989 mm | 6.883 mm | 1.119° | 1.705° | 0 |
| 5 | 25.410 ms | 83.077 ms | 3.890 mm | 7.030 mm | 1.128° | 1.705° | 0 |
| 8 | 21.173 ms | 65.730 ms | 4.436 mm | 7.147 mm | 1.173° | 1.472° | 0 |
| 10 | 16.635 ms | 48.525 ms | 4.687 mm | 7.026 mm | 1.276° | 2.211° | 0 |
| 15 | 12.121 ms | 36.201 ms | 4.261 mm | 7.026 mm | 1.454° | 1.986° | 0 |
| **20** | **12.860 ms** | **30.453 ms** | 5.133 mm | 8.249 mm | 1.584° | 1.993° | 0 |
| 30 | 16.750 ms | 24.976 ms | 4.481 mm | 7.968 mm | 1.692° | 2.186° | 0 |

閾値: 欠測 0、眼位置 p95 ≤ 5 mm、角度 p95 ≤ 1°、認識 median ≤ 16.7 ms、p95 ≤ 33.3 ms。

**判定: FAIL — 選定なし（`selected_interval: null`、非ゼロ終了）。既定は `interval=1`（毎フレーム全検出）のまま据え置く。**

達成状況を項目別に分けると:

- **欠測 0: 全 interval で達成。**
- **眼位置 p95 ≤ 5 mm: interval 2, 3, 5, 8, 10, 15, 30 で達成**（3.539–4.694 mm）。20 だけ 5.133 mm でわずかに未達。
- **角度 p95 ≤ 1°: どの interval でも未達。** 最良でも interval=2 の 1.018°（上限の1.8%超過）で、interval を伸ばすほど悪化する（1.692° @ 30）。
- **認識 median ≤ 16.7 ms かつ p95 ≤ 33.3 ms: interval=20 でのみ両立**（12.860 / 30.453 ms）。interval=30 は p95 24.976 ms だが median 16.750 ms が 0.05 ms 超過。

したがって**精度と遅延を同時に満たす interval は存在しない**。作業書「閾値を満たす候補がなければ
refresh=1 へ戻して精度を守り、性能未達を記録する。閾値を緩めて成功扱いしない」に従い、閾値は
緩めず、既定を毎フレーム全検出のまま据え置いた。時系列ROIは `--detector-refresh-interval N`
で明示的に選ぶ opt-in 機能として残す。

#### 未達の構造的な理由

認識 p95 は**必ず全検出フレームに支配される**。全検出は約 34–55 ms、landmark 単独は約 12–17 ms
なので、全検出フレームが全体の 5% を超える限り（つまり interval < 20）、p95 は必ず全検出の時間に
なる。p95 ≤ 33.3 ms を満たすには interval ≥ 20 が必要で、その領域では眼位置と角度の誤差が増える。

#### 記録しておくべき所見（閾値の見直しはユーザー判断）

角度 p95 が守っている `head_forward_display` は、**レンダラーが一切消費していない**
（`src/headcoupled_display/static/` に `head_forward` の参照は0件）。非対称視錐台は両眼中点の
位置だけで決まり、`view_matrix` は純平行移動である（頭の回転は「何を見るか」を変えるだけで
窓を回さない）。したがって 1° の角度閾値は、現在の表示経路が使っていない量を判定している。
一方、表示に直接効く眼位置 p95 ≤ 5 mm は interval 2–15 と 30 で満たされている。
**この閾値を見直すかどうかは成功条件の変更にあたるため、本作業では変更していない。**
