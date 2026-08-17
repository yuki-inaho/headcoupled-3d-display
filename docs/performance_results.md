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
