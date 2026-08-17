# 性能設計と参考実装の採否記録

## 1. 目的

本書は `temp/workdoc_Aug17-2026_headcoupled_scene_latency.md` フェーズ1・手順8の成果物として、
低遅延化にあたって参照した外部実装・公式ドキュメント4件について、採用する設計原則と採用しない実装・その理由を明示する。
あわせて、フェーズ5（IPC候補比較）で使う比較条件を先に固定し、実測前に基準がぶれないようにする。

採用結果（実測値）はフェーズ5（手順33・34）で追記する。本書作成時点（2026-08-17）では計測を実施していないため、
「4. 採用結果」は未計測のプレースホルダである。

## 2. 参考実装の採否記録

各行の「ライセンス」欄は、GitHub License API（`gh api repos/<owner>/<repo>/license`）およびWebFetchによる
ページ本文確認で、2026-08-17に本作業内で直接確認した内容のみを記載する。確認できなかった事項は推測で断定せず
「未確認」と明記する。確認に使ったコマンド・URLは「検証方法」列の通り。

| 参照 | URL | 確認コミット／版 | ライセンス | 採る原則 | 採らない実装と理由 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| WebGL-and-WASM-Point-Cloud-Visualizer | https://github.com/toksaitov/WebGL-and-WASM-Point-Cloud-Visualizer | commit `689c6257b0014cc5e3302e80bc35150f6ff8f885`（2019-04-29T19:15:11Z、`gh api`で存在確認済み） | 明示LICENSEファイルなし（`gh api repos/toksaitov/WebGL-and-WASM-Point-Cloud-Visualizer/license` は `404 Not Found`、2026-08-17確認）。**ライセンス不明のため、ソースコードの転載は一切行わない。**設計原則の言語化のみを参照する。 | GPUバッファへの一括アップロード、instanced/バッチ化されたdraw call、ブラウザーのレンダリングループ設計の考え方 | C/WASM移植そのもの。ライセンス未整備の実装をコピーしないため、コードは1行も転載しない。 |
| wgpu-gs-viewer | https://github.com/abist-co-ltd/wgpu-gs-viewer | commit `51a325a1cacd73045dad050e1b2677f1d2897e64`（2026-07-21T22:26:57Z、`gh api`で存在確認済み） | MITライセンス（リポジトリ直下 `LICENSE` を `gh api .../contents/LICENSE` で取得しbase64デコードして本文確認、2026-08-17）。Copyright (c) 2026 株式会社アビスト イノベーションセンター。同ファイルにVkRadixSort（https://github.com/MircoWerner/VkRadixSort, Copyright (c) 2023 Mirco Werner）由来コードのMIT third-party noticeが併記されている。GitHub License APIの自動判定は third-party notice併記により `Other`/`NOASSERTION` となるが、ファイル本文は標準MIT全文＋third-party MIT全文であり、内容としてはMITと確認できる。 | dirty時（pose/resize/mode変更時）だけ再計算する、ブラウザーで同時in-flightのGPU作業を1に抑える、GPUへ常駐させる、常に最新状態を優先して描く、という設計原則 | Gaussian Splatting用の32-bit depth sort、tile分割、間接dispatch/描画。今回の対象は13,810点の不透明点群であり、これらの機構は過剰。Rust/WebGPUへの移植も行わない（非ゴール、本書§1.2記載の非目標と同一）。 |
| ZeroMQ `ZMQ_CONFLATE`（`zmq_setsockopt`） | https://libzmq.readthedocs.io/en/latest/zmq_setsockopt.html | 2026-07-26公開版（ページ内 "Last Updated: 2026-07-26 12:58:15 UTC" をWebFetchで確認、2026-08-17実施） | 参照ページ自体はlibzmqのmanページドキュメントであり、ページ単体の再配布条件は未確認。libzmq本体（`zeromq/libzmq`）のコードは **MPL-2.0**（`gh api repos/zeromq/libzmq/license` で確認、2026-08-17）。※旧来LGPLと記憶されがちだが、現行はMPL-2.0であることを実際に確認した。 | 「最新の単一メッセージだけを保持する」というconflate（最新値配送）の考え方 | **`ZMQ_CONFLATE` とmultipartメッセージの併用。** WebFetchで確認した公式記述: "Does not support multi-part messages, in particular, only one part of it is kept in the socket internal queue."（複数パートのうち1パートしかキューに保持されない）。採用する場合は制御用とプレビュー用を別ソケット・単一パートに分離すること。 |
| gRPC Performance Best Practices | https://grpc.io/docs/guides/performance/ | 2024-11-12更新版（ページ内 "Last modified November 12, 2024" をWebFetchで確認、2026-08-17実施） | ページfooterは「© 2026 gRPC Authors」表記＋Licenseリンクのみで、リンク先の具体的なライセンス種別（Apache-2.0かCC-BY-4.0か等）はページ本文に明記されておらず**未確認**。なお、gRPC本体コード（`grpc/grpc`）は**Apache License 2.0**（`gh api repos/grpc/grpc/license` で確認、2026-08-17）だが、これはドキュメントサイトのコンテンツライセンスとは別。 | channelおよびstubの再利用（"Always re-use stubs and channels when possible."）、keepalive pingの利用、実測によるstreaming方式の評価 | 「gRPCだから速い」という前提での無条件採用。WebFetchで確認した公式記述: "Streaming RPCs create extra threads for receiving and possibly sending the messages, which makes streaming RPCs much slower than unary RPCs in gRPC Python, unlike the other languages supported by gRPC."（gRPC PythonのstreamingはunaryよりPython特有の追加スレッドコストで大幅に遅くなりうる）。採否はフェーズ5の実測でのみ決める。 |

### 検証方法（再現用コマンド）

```bash
# WebGL-and-WASM-Point-Cloud-Visualizer: LICENSE有無とcommit存在確認
gh api repos/toksaitov/WebGL-and-WASM-Point-Cloud-Visualizer/license
gh api repos/toksaitov/WebGL-and-WASM-Point-Cloud-Visualizer/commits/689c6257b0014cc5e3302e80bc35150f6ff8f885 --jq '.sha, .commit.author.date'

# wgpu-gs-viewer: LICENSE本文とcommit存在確認
gh api repos/abist-co-ltd/wgpu-gs-viewer/license
gh api repos/abist-co-ltd/wgpu-gs-viewer/commits/51a325a1cacd73045dad050e1b2677f1d2897e64 --jq '.sha, .commit.author.date'
gh api repos/abist-co-ltd/wgpu-gs-viewer/contents/LICENSE?ref=develop --jq '.content' | base64 -d

# libzmq本体ライセンス確認
gh api repos/zeromq/libzmq/license

# gRPC本体ライセンス確認
gh api repos/grpc/grpc/license
```

ZeroMQ `zmq_setsockopt` ページとgRPC Performance Best Practicesページの本文・更新日時はWebFetchで直接取得して確認した
（上表「確認コミット／版」欄に記載の日時はいずれもページ内表示をそのまま転記したもの）。

## 3. 実装者向けの注意（誤解防止）

コード実装者が本書の原則を読んで実装を始める際、以下4点を誤解しないこと。

### 3.1 WASM/WebGPUへ移植しない理由

対象点群は13,810点の**不透明**点群であり、Gaussian Splattingで必要になる32-bit depth sort、
tile分割、間接dispatch/描画といった機構は今回の規模・用途に対して過剰である。加えて、実測で判明している
ボトルネックは描画段ではなく認識段（顔検出）とIPC段（同期HTTP + JPEG decode/re-encode）にある
（`temp/workdoc_Aug17-2026_headcoupled_scene_latency.md` §1.1「現在確認済みの事実」参照）。
したがって wgpu-gs-viewer から採るのは実装そのものではなく、
「dirty時だけ再計算する」「in-flightのGPU作業を1に抑える」「GPUへ常駐させる」「常に最新状態を優先する」
という4つの設計原則のみであり、Rust/WebGPUへの全面移植は行わない。

### 3.2 ZeroMQ CONFLATEでmultipartを使わない理由

`ZMQ_CONFLATE` は公式ドキュメントに明記される通り、multipartメッセージと**非互換**である
（"Does not support multi-part messages, in particular, only one part of it is kept in the socket internal queue."）。
そのため、ZeroMQを採用する場合は制御（control）レーンとプレビュー（preview）レーンを
**別ソケット・単一パート**として分離すること。1つのソケットでcontrolとpreviewをmultipartにまとめた上で
CONFLATEを付けるという実装は仕様上成立しない。

### 3.3 gRPCを「速いから」で選ばない

gRPC公式のPerformance Best Practicesには、gRPC PythonのstreamingがPython特有の追加スレッドコストにより
unary RPCより大幅に遅くなりうることが明記されている。したがって「gRPCは一般に高速」という評価をそのまま
本プロジェクトへ適用してはならない。channelの再利用（stub/channelをリクエストごとに作り直さない）は
原則として採用するが、streaming方式そのものの採否はフェーズ5の実測（p50/p95/p99等）でのみ決定する。

### 3.4 TensorRTは非ゴール

TensorRT Execution Providerを要求した際、`libnvinfer.so.10` が不在のためCUDAへ**暗黙fallback**し、
平均36.50 ms / 27.4 FPSという実測が得られている（CUDA単独指定時は平均33.16 ms / 30.2 FPS）。
ユーザー指示によりTensorRTは今回の作業では使用しない（`libnvinfer.so.10` の解決やTensorRT導入は非ゴール）。
なお、この実測はTensorRTが遅いことを示すものではなく、TensorRT未導入環境でCUDAへ暗黙fallbackした場合の値である。
CUDA明示指定が**CPU**へ暗黙fallbackした場合は、速度に関わらず失敗として扱う（成功扱いにしない）。

## 4. 通信候補の比較条件（フェーズ5で使用）

フェーズ5（手順31〜34、IPC候補比較と制御/プレビュー分離）で使用する比較条件を、実測前に以下の通り固定する。

### 4.1 比較する候補

1. 現行 JSON/HTTP（同期HTTP POST + JSON、現状のベースライン実装）
2. binary HTTP（同期HTTP POSTだがpayloadをJSONではなく固定長バイナリにしたもの）
3. ZeroMQ（`ZMQ_CONFLATE` を用いた最新値配送、control/previewは別ソケット・単一パート）
4. gRPC Python（async APIでchannel再利用、streaming方式の採否は実測で判断）

### 4.2 同一条件として明記する内容

- **同一のcontrol packet**: 12点float32（両眼中点等、6点×2）、score、sequence、送信時刻の
  monotonic ns と Unix ns、protocol version、を含む固定フィールド構成。
- **同一のpreview**: 640×360 JPEG、認識後に生成、最大10 FPS。
- **同一の負荷条件**: control 60 Hz、preview 10 Hz、consumer側に100 msのstall（滞留）を意図的に挟んだ過負荷条件を含む。
- 各候補は**5回**実行し、p50/p95/p99、CPU使用率、bytes/s、drop数、overload recovery（過負荷解除後に最新値へ
  追いつくまでのフレーム数）、実際に使用したpackageのversionを記録する。

### 4.3 採用基準

以下をすべて満たす候補のみを採用対象とする。

- control packetのp95が **2 ms以下**
- 過負荷解除後、**2フレーム以内**に最新値へ追いつく
- 推論スレッドが送信完了を待たされない（fire-and-forgetまたは非block）
- sequenceの逆転が**0件**

### 4.4 タイブレーク（同等時の選び方）

複数候補が採用基準を満たし、かつp95の差が**10%以内**で同等とみなせる場合は、
「最新値conflationを直接表現でき、依存パッケージと生成コードが少ない方式」を選ぶ。

### 4.5 依存の隔離

比較対象となる候補依存（pyzmq、grpcio等）は製品runtimeのlock（`requirements.lock`）へ入れない。
比較・計測専用の `requirements.transport-bench.in` / `requirements.transport-bench.lock` による
隔離されたuv環境にのみ導入し、採用が決まった単一方式のみを製品runtimeへ追加する（手順32・39参照）。

## 5. 採用結果（未計測）

**本書作成時点（2026-08-17）ではフェーズ5の実測を実施していない。以下は結果を追記するための空欄プレースホルダであり、
実測していない数値は記載しない。**

| 候補 | p50 | p95 | p99 | CPU | bytes/s | drop | overload recovery | 使用package version | 採用基準の判定 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 現行 JSON/HTTP | 未計測 | 未計測 | 未計測 | 未計測 | 未計測 | 未計測 | 未計測 | 未計測 | 未計測 |
| binary HTTP | 未計測 | 未計測 | 未計測 | 未計測 | 未計測 | 未計測 | 未計測 | 未計測 | 未計測 |
| ZeroMQ (CONFLATE) | 未計測 | 未計測 | 未計測 | 未計測 | 未計測 | 未計測 | 未計測 | 未計測 | 未計測 |
| gRPC Python | 未計測 | 未計測 | 未計測 | 未計測 | 未計測 | 未計測 | 未計測 | 未計測 | 未計測 |

### 5.1 実測結果（2026-08-17、手順33）

GTX 1070 / CUDA 11.8 の実機、localhost、隔離環境 `.venv-transport-bench`。
条件は §4.2 のとおり（control 60 Hz、preview 10 Hz、consumer stall 100 ms、
control packet 146 バイト、warmup 60 メッセージ）。

初回は各候補5回の予定を10回で実施したが、p95 の分布が大きく重なり中央値の順位が
判断根拠にならなかった（json_http 1.06–2.73 ms、binary_http 1.14–**15.69** ms、
zeromq 0.90–7.11 ms）。§4.2 の「外れ値が大きい場合は回数を増やす」に従い、
**全4候補を各25回で再測定**した。以下は25回版である。

| 候補 | version | p95中央値 | p95 最小 | p95 最大 | p99中央値 | max_age中央値 | drop中央値 | sequence逆転 | 復帰frame | producer最大enqueue |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 現行 JSON/HTTP | (stdlib) | **1.195 ms** | 1.036 | 2.395 | 2.052 | 0.690 ms | 5 | 0 | 2 | 0.206 ms |
| binary HTTP | (stdlib) | 1.383 ms | 1.018 | 4.233 | 2.713 | 0.665 ms | 5 | 0 | 2 | 0.171 ms |
| ZeroMQ | pyzmq 26.4.0 | 1.353 ms | 0.837 | 5.692 | 3.585 | 1.032 ms | 5 | 0 | **1** | 0.488 ms |
| gRPC Python | grpcio 1.83.0 | 3.972 ms | 1.324 | 7.944 | **46.596** | **85.137 ms** | **0** | 0 | 1 | 0.734 ms |

25回のうち採用基準を満たした回数:

| 候補 | control p95 ≤ 2 ms | max_age ≤ 33.3 ms（30 fps の1フレーム） | producer がブロックした回数 |
| :--- | ---: | ---: | ---: |
| 現行 JSON/HTTP | **23/25** | 25/25 | **0** |
| binary HTTP | 18/25 | 25/25 | **0** |
| ZeroMQ | 13/25 | 25/25 | 1 |
| gRPC Python | **3/25** | **0/25** | 2 |

raw: `artifacts/perf/transport_comparison.json`（10回、SHA-256
`e4b5326249be94975ef22123dd9e9135863aa83a1a3944d1dd2163e62595720e`）、
`artifacts/perf/transport_comparison_25.json`（25回、SHA-256
`b5f6541bd424083cb13f57317a0867bfd78e257c689c46a70063722d0aff91c6`）。
再現コマンド: `just setup-transport-bench` のあと
`.venv-transport-bench/bin/python scripts/benchmark_transports.py --runs 25 --output <path>`。

### 5.2 採用方式

**採用: HTTP トランスポート + `headcoupled_display.protocol` の固定長バイナリ control packet。**
control と preview は別エンドポイントに分ける。新規の runtime 依存は追加しない。

トランスポートの選定根拠（HTTP を選ぶ理由）:

- 採用基準（p95 ≤ 2 ms、過負荷解除後2フレーム以内、推論スレッド非ブロック、sequence逆転0）を
  中央値で満たし、かつ**25回すべてで producer をブロックしなかった**。
- `max_age` が25回すべて 33.3 ms 以内、すなわち常に1フレーム以内の鮮度を保った。
- 製品の `pyproject.toml` / `requirements.lock` に新しい依存を一切追加しない。

control payload をバイナリにする理由（レイテンシではなく契約のため）:

- JSON/HTTP と binary/HTTP は**同じトランスポート**であり、選択しているのは payload の符号化である。
- 実測は json 1.195 ms / binary 1.383 ms（中央値）、基準達成 23/25 対 18/25 で、
  **binary のほうが速いという主張はできない**。n=25 では両者を分離できず、どちらも中央値で
  基準を大きく下回る。ここでバイナリを採る理由は速度ではない。
- バイナリ側にだけある性質: 固定146バイト、magic とプロトコル version の検査、
  切り詰め・NaN/Inf の拒否、monotonic ns と Unix ns の2系統を混同させない明示フィールド。
  物理ディスプレイを駆動する制御レーンでは、この契約が JSON の柔軟さより価値がある。
- **これは明示的な判断であり、計測で binary が優れていたという主張ではない。**

### 5.3 非採用理由

**gRPC Python — 不採用。** §3 で「速いはずという前提で採用しない」と決めたとおり実測で判断した。
25回すべてで `max_age` が 84〜86 ms、かつ `dropped_count` が **0**。すなわち古いフレームを
一切捨てずにキューへ溜めて配送している。これは §2 で「負荷試験で否定できた場合に限る」とした
flow-control による古いframe滞留そのものであり、**否定できず確認された**。加えて
control p95 ≤ 2 ms を満たしたのは 3/25 回、p99 中央値 46.596 ms、producer を2回ブロックした。

**ZeroMQ — 不採用。** `ZMQ_CONFLATE` による最新値配送は実際に効いており、過負荷解除後の復帰が
**1フレーム**（HTTP は2フレーム）と唯一の明確な優位点だった。しかし採用基準の p95 ≤ 2 ms を
満たしたのは 13/25 回にとどまり、25回のうち1回は producer をブロックした。要求は「2フレーム以内」で
あって「1フレーム」ではないため、HTTP でも要求は満たせる。この状況で pyzmq を製品 runtime 依存に
加える理由がない。

**採用時の設定（採用しなかったが記録として）:** ZeroMQ を後日採用する場合、`ZMQ_CONFLATE=1` は
multipart と併用できないため、control と preview は**別ソケット・単一パート**にすること（§2 参照）。

### 5.4 障害時の挙動

- 送信先が落ちている場合、producer は1フレームにつき1回だけ再接続を試み、失敗はそのフレームの
  失敗として扱う。バックログを積まない。
- サーバー側は入力が `stale_after_s`（既定 0.5 秒）を超えて途絶えたら、最後の眼位置は保ったまま
  confidence 0.0 / `diagnostics.stale=true` を配信する。高い confidence の古い姿勢を送り続けない。
- preview レーンが購読されていない・更新されていなくても control 姿勢は進む。両レーンは独立である。

