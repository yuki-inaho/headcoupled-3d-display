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

### 手順7: 現行録画ベースライン — 2026-08-17T07:38:36.254954+00:00 (commit `f6c4abf341c7`)

- **コマンド:** `just benchmark-recorded`
- **commit:** `f6c4abf341c746e47fa22894eb9eb3ac14c08ac1`
- **入力:** `/home/inaho-omen/Project/facemesh_tracking/recordings/test10.avi` (1280x720, frame_count=294, warmup=5)
- **provider:** `CUDAExecutionProvider`
- **clock_domain:** `monotonic_ns` (uncertainty 0.000001 ms)
- **raw JSON:** `artifacts/perf/baseline_recorded_raw.json` (SHA-256: `9aeb300b3b079fb3d5af4298ff0d1b27298bdcdb317058e55ad29c272b1f1565`)
- **欠測フレーム数:** 0
- **計測時刻 (created_at):** 2026-08-17T07:38:36.254954+00:00

- **参考桁確認:** detector + landmarks の p50 合計 48.84 ms は 既知の静止画ベンチ mean≈33.16 ms と同桁（比 1.47 倍）。過去値は成功判定の固定基準ではなく参考のみ。

| stage | sample_count | p50 (ms) | p95 (ms) | p99 (ms) |
| :--- | ---: | ---: | ---: | ---: |
| capture_decode | 289 | 9.031 | 13.043 | 20.702 |
| detector | 289 | 33.416 | 50.107 | 93.212 |
| landmarks | 289 | 15.421 | 22.855 | 46.683 |
| recognition_total | 289 | 48.963 | 73.632 | 141.260 |
| packet_build | 289 | 3.117 | 4.892 | 9.976 |
| preview_resize_encode | 289 | 3.048 | 4.397 | 6.104 |

**判定:** PASS — CUDA providerが実行中で、全段のp50/p95/p99が採取・検証された（このステップはベースライン記録であり、閾値ゲートは後続手順で行う）。

### 手順19-21 / 成功条件9: ブラウザー側の描画遅延 — 2026-08-17

- **コマンド:** `PYTHONPATH=src .venv/bin/python -m pytest -m e2e tests/e2e -q -k draw_latency`
- **入力:** synthetic source（この半分は姿勢の出所に依存しないため）
- **環境:** headless Chromium + SwiftShader。**物理ディスプレイではない。**
- **raw:** `artifacts/perf/browser_timing.json`

| 指標 | 実測 | 閾値 | 判定 |
| :--- | ---: | ---: | :--- |
| CPU 描画 p50 | 0.100 ms | — | — |
| **CPU 描画 p95** | **0.400 ms** | ≤ 4 ms | **PASS** |
| receive→draw p50 | 23.900 ms | — | — |
| **receive→draw p95** | **40.500 ms** | ≤ 16.7 ms | **FAIL** |
| sequence 逆転 | 0 | 0 | PASS |
| GPU timer query | 利用可 | — | — |
| **描画間隔 p50** | **342.900 ms** | — | — |

**判定: 部分的 FAIL。** CPU 描画時間はレンダラー自身の性質であり、閾値 4 ms に対して
0.400 ms と一桁以上の余裕がある。一方 receive→draw は**コンポジタの周期が下限**になる。
姿勢は「次のフレーム」より早くは表示できない。この環境で計測した描画間隔 p50 は
**342.9 ms（約3 fps）** であり、headless Chromium が requestAnimationFrame を絞っている
ことを示す。つまり receive→draw 40.5 ms はレンダラーの遅さではなく、この環境の
コンポジタ周期に律速された値である。

閾値は緩めない。**成功条件9 は未達のまま残す。**正しい判定には物理ディスプレイ上での
再計測が必要であり、それは実カメラ受け入れと同じく本作業の非ゴールである。
E2E テストは CPU 描画時間だけを assert し、receive→draw は
`scripts/validate_performance.py` が判定できるよう JSON に書き出す。こうすることで、
CI 機で必ず落ちるテストにも、通るまで閾値を緩めた判定にもならない。

### 手順40 / 成功条件10: 録画+実CUDA推論の本番相当E2E — 2026-08-17（再測定・訂正）

- **コマンド:** `just test-e2e-recorded-cuda`
- **経路:** `test10.avi` → **実CUDA FaceMesh推論**（Python 3.10環境）→ 採用IPC（2レーン）→
  12点SQPNP → `/ws/pose` → 非対称投影 WebGL2。**合成入力は一切使用していない。**
- **プロファイル:** `hardware_profile.local.json`（15 cm / 12°）+ tagcal実測 `K,D` + 個人メッシュ
- **producer:** `--source test10.avi --pacing realtime --backend cuda --max-frames 120`
- **raw:** `artifacts/perf/recorded_cuda_e2e.json`

**通ったこと（経路の成立）**

| 指標 | 実測 | 判定 |
| :--- | :--- | :--- |
| production経路の完走 | 録画→実CUDA→2レーンIPC→12点SQPNP→WebGL2 | **成立** |
| プレビュー解像度 | 640 × 360 | **PASS** |
| sequence 逆転 | 0 | **PASS** |
| renderer | WebGL2 | — |
| source | `ipc` | — |
| CPU 描画 p95 | 0.6 〜 1.6 ms（閾値 4 ms） | **PASS** |

**成功条件10（推論完了→WebGL）: この環境では評価できない**

複数回実行した結果は次のとおり大きくばらつく。

| run | median | p95 | 描画間隔 p50 | 時計不確かさ |
| :--- | ---: | ---: | ---: | ---: |
| 1 | 60.674 ms | 87.829 ms | 493.200 ms | 未計測 |
| 2（外部レビュー） | 123.295 ms | 246.013 ms | — | 未計測 |
| 3 | 47.615 ms | 65.393 ms | 373.300 ms | **5.550 ms** |
| 4（時計25サンプル） | 48.761 ms | 119.907 ms | **1008.200 ms** | **5.500 ms** |

**判定: REJECTED（FAILではなく、測定として成立していない）。** 理由は2つ。

1. **時計不確かさが閾値を超える。** 作業書は「clock逆行または2 ms超の時刻不確かさがある
   runを拒否する」と定めている。producer の Unix 時刻からブラウザーの Unix 時刻を引く以上、
   両者のズレを測らなければ差分は検証されていない。`/api/health` の `server_unix_ns` を
   2つのローカル読みで挟むNTP方式で25回測り、最速の交換を採っても
   **不確かさは 5.5 ms** で 2 ms を超える（オフセット自体は -1.7 ms と小さく、同一ホスト
   なので当然だが、往復時間が絞られない）。判定器はこの run を明示的に拒否する。
2. **描画間隔がレイテンシを支配し、しかも安定しない。** 373 ms 〜 1008 ms（約 1〜3 fps）。
   headless Chromium が requestAnimationFrame を絞っており、姿勢は「次のフレーム」より
   早くは表示できない。CPU 描画は 0.6〜1.6 ms なので、レンダラー側の遅さではない。

**閾値は緩めない。成功条件10は未達のまま残す。**正しい判定には、物理ディスプレイに
接続し、コンポジタが実表示のリフレッシュ周期で回る環境での再計測が必要である。それは
実カメラ受け入れと同じく本作業の非ゴールである。

### 判定器の総合結果（`scripts/validate_performance.py`）

`artifacts/perf/verdict.json`。

| 条件 | チェック | 判定 |
| :--- | :--- | :--- |
| 前提 | clock_uncertainty | PASS |
| 成功条件4 | cuda_provider | PASS |
| 成功条件5 | frame_completeness | PASS |
| 成功条件5 | accuracy_vs_full_detect | PASS（既定が全検出のため差は定義上ゼロ。最適化を検証して通したのではない） |
| 成功条件6 | recognition_latency | **FAIL** |
| 成功条件7 | control_transport | PASS |
| 成功条件8 | preview_lane | PASS |
| 成功条件9 | browser_draw | **FAIL** |
| 成功条件10 | inference_to_webgl | **FAIL** |

**総合 FAIL（9件中3件）。** 3件の未達はいずれも数値と原因を上に記録してあり、
閾値の緩和も未計測の pass 扱いも行っていない。

### 手順33-34 訂正: transport 採用判定 — 2026-08-17（再測定・再判定）

当初この文書と `docs/performance_design.md` は「HTTP + バイナリ control packet を採用、
基準を中央値で満たす」と記載していた。**これは誤りだった。**
`scripts/benchmark_transports.py` の `control_p95_le_2ms` は `worst <= threshold`、
すなわち **1 run でも違反したら不合格**と定義されている。中央値で判定するのは、
ハーネス自身が不合格にした候補を合格として報告することであり、採用基準を後から
緩めたことになる。

作業書「外れ値が大きい場合は回数を増やす」に従い全4候補を再度25回測定した
（`artifacts/perf/transport_comparison_idle.json`）。計測時ホストは静止しておらず、
load average 6.86 / 6.74 / 5.99（12コア）、デスクトップセッションと無関係のブラウザーが
CPU を消費していた。

| 候補 | 機械判定 | p95 中央値 | **p95 最悪** | 2 ms 以下 | max_age 中央値 |
| :--- | :--- | ---: | ---: | ---: | ---: |
| json_http | **不合格** | 1.302 ms | 3.209 ms | 22/25 | 0.776 ms |
| binary_http | **不合格** | 1.422 ms | 2.308 ms | 21/25 | 0.768 ms |
| zeromq | **不合格** | 1.092 ms | 3.113 ms | 19/25 | 0.879 ms |
| grpc | **不合格** | 2.363 ms | 5.311 ms | 9/25 | **84.479 ms** |

`selected_candidate: null`。初回のより高負荷な25回版でも同結論（binary_http 最悪 4.233 ms、
18/25）で、**負荷を下げても合格しない**。

**判定: FAIL。DoD-7 は未達。** 判定器 `control_transport` は worst-run 値を必須とし、
この実測に対して fail を返す。実装は HTTP で進めたが、それは暗黙採用ではなく
`docs/performance_design.md` §5.3 に明記した判断であり、**ゲートを通ったという主張はしない**。

### 手順7: 現行録画ベースライン — 2026-08-17T11:28:43.356337+00:00 (commit `040a4c2a78dc`)

- **コマンド:** `just benchmark-recorded (final, after review fixes)`
- **commit:** `040a4c2a78dc7acbea0522b1ecb4bd9ba1641f93`
- **入力:** `/home/inaho-omen/Project/facemesh_tracking/recordings/test10.avi` (1280x720, frame_count=294, warmup=5)
- **provider:** `CUDAExecutionProvider`
- **clock_domain:** `monotonic_ns` (uncertainty 0.000001 ms)
- **raw JSON:** `artifacts/perf/baseline_recorded_raw.json` (SHA-256: `325ead08247677a66ace29f880540af10b6f034ed4771e70e805975d12232321`)
- **欠測フレーム数:** 0
- **計測時刻 (created_at):** 2026-08-17T11:28:43.356337+00:00

- **参考桁確認:** detector + landmarks の p50 合計 59.43 ms は 既知の静止画ベンチ mean≈33.16 ms と同桁（比 1.79 倍）。過去値は成功判定の固定基準ではなく参考のみ。

| stage | sample_count | p50 (ms) | p95 (ms) | p99 (ms) |
| :--- | ---: | ---: | ---: | ---: |
| capture_decode | 289 | 10.617 | 28.808 | 163.487 |
| detector | 289 | 39.963 | 75.098 | 106.205 |
| landmarks | 289 | 19.463 | 37.883 | 54.517 |
| recognition_total | 289 | 61.312 | 106.738 | 147.095 |
| packet_build | 289 | 3.698 | 7.771 | 11.733 |
| preview_resize_encode | 289 | 3.319 | 6.885 | 9.551 |

**判定:** PASS — CUDA providerが実行中で、全段のp50/p95/p99が採取・検証された（このステップはベースライン記録であり、閾値ゲートは後続手順で行う）。

### 最終証跡と総合判定 — 2026-08-17（外部レビュー是正後）

- **判定対象 commit:** `040a4c2a78dc7acbea0522b1ecb4bd9ba1641f93`
- **成果物:** `artifacts/perf/final/`（`.gitignore` 対象。再実行で再生成できる）
- **コマンド:** `just validate-performance --final artifacts/perf/final/recognition.json --baseline artifacts/perf/baseline_recorded.json --missing-faces 0 --browser ... --transport ... --accuracy ... --preview ... --end-to-end ... --output artifacts/perf/final/verdict.json`

| ファイル | SHA-256 |
| :--- | :--- |
| `accuracy.json` | `3e8b780a43e51e75b2541679f69ee205b09ab40653182a350cb06ebafc511c55` |
| `browser_timing.json` | `3ef89a9bcc798d30caa5cb67518a908c5fe8f8d01bafaac03b0f71f7857d0919` |
| `end_to_end.json` | `9ed3fb4380beb5216e64f495ca8aaa13fa45bd718124a383d5f00bef6b5cfabd` |
| `preview.json` | `e0e5cd33e210bb7f68661daf593c0c69344fe1ec3cab73104752b6e63ca1f43c` |
| `recognition.json` | `0e4ee2dc56bd8af5882db848fd93464363ace169858e324cae37845fea5d7238` |
| `recognition_raw.json` | `325ead08247677a66ace29f880540af10b6f034ed4771e70e805975d12232321` |
| `refresh_sweep.json` | `7a8a1432655bbd9ef004090f32a97a75ddbf3e111af7422d6c679cbcac1af629` |
| `refresh_sweep_long.json` | `523c21d00a83f1458f54192f8f3fe846b981cf5ec90389b970dcea4472c4d466` |
| `transport.json` | `b5997ce46716f1e3427e6fcbd2ceb30fa53042412dcae73ec12d605080551223` |
| `transport_comparison.json` | `4424835192f698bf795de81bc2cec5d7840128419021247261456884e84e693d` |
| `verdict.json` | `4baa4d5d313eb0d5588494a941d8bf1515f804e3400feb4b5621f553018e4a99` |

| 条件 | チェック | 判定 |
| :--- | :--- | :--- |
| 前提 | `clock_uncertainty` | PASS |
| 成功条件4 | `cuda_provider` | PASS |
| 成功条件5 | `frame_completeness` | PASS |
| 成功条件5 | `accuracy_vs_full_detect` | PASS |
| 成功条件6 | `recognition_latency` | **FAIL** |
| 成功条件7 | `control_transport` | **FAIL** |
| 成功条件8 | `preview_lane` | PASS |
| 成功条件9 | `browser_draw` | **FAIL** |
| 成功条件10 | `inference_to_webgl` | **FAIL** |

**総合 FAIL（9件中 4件が未達）。**

未達4件の内訳と、それぞれ何が原因かは上の各節に数値つきで記録してある。
**閾値の緩和も、未計測の pass 扱いも行っていない。**

- 成功条件6（認識レイテンシ）: 全検出が構造的に p95 を支配する。時系列ROIは精度と
  遅延を同時に満たす N が存在しなかったため既定にしていない。
- 成功条件7（transport）: 全4候補が worst-run 基準で不合格。実装は HTTP で進めたが
  ゲートを通ったという主張はしない。
- 成功条件9・10（ブラウザー描画 / 推論→WebGL）: headless SwiftShader のコンポジタ
  周期（373〜1008 ms）が支配的で、成功条件10 は時計不確かさ 5.5 ms > 2 ms により
  run 自体が拒否される。物理ディスプレイでの再計測が必要。

