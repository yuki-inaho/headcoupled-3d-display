# 入力スナップショット

成果物の解析対象は次の三ファイルです。

| ファイル | SHA-256 | 備考 |
|---|---|---|
| `apriltag-camera-calibrator-main.zip` | `f3a4b0397b037a010bb08bbd92a5adb5f309efadf6b9bd7a48236939d2c947a3` | `components/tagcal/` へ展開 |
| `facemesh_tracking_reconstruction-main.zip` | `1320e719ab4aac75c8a5fcc2cc49f2f7dd1e81aebcbbf2f6c3e7256df072ba88` | `components/facemesh_tracking/` へ展開 |
| `ChatGPT-顔認識_3D_ビュー (3).md` | `12b78dddbe1b10dd34cedba4113d1e2444317dd1e654a0e2ee22f2a348e927f` | 4,227行、全文は成果物へ再収録せず要件を`source_review.md`へ整理 |

会話ファイルを成果物へ複製しない理由は、設計の再現に不要な会話文脈を圧縮ファイルへ
重複収録しないためです。要件、未確定値、採用した座標系、較正方式は
`docs/source_review.md` と `docs/architecture.md` に追跡可能な形で記載しています。

## 本環境での実測に使用した追加入力（2026-08-17）

上の三ファイルは配布物作成時の入力です。ローカルでの受け入れと性能計測では、
リポジトリ外にある次の実データを参照しました。いずれも**リポジトリへは追加していません**。

| パス | 用途 | 備考 |
| :--- | :--- | :--- |
| `/home/inaho-omen/Project/facemesh_tracking/recordings/test10.avi` | 段階別レイテンシ、精度スイープ、録画E2E | 1280×720、**実デコード294フレーム**。ヘッダは 602 frame / 60 fps と申告するが信用しない |
| `/home/inaho-omen/Project/facemesh_tracking/outputs/test10_landmarks.json` | 保存済みlandmarks replay | 同じ録画の478点×294フレーム |
| `/home/inaho-omen/Project/facemesh_tracking/recordings/me/shape.pcd` | 個人用478点メッシュ | **生体情報**。`.gitignore` 対象であり、リポジトリへ入れない |
| `/home/inaho-omen/Project/apriltag-camera-calibrator/artifacts/eval_refine/calibration.json` | カメラ内部パラメータ `K, D` | 1280×720、RMS 1.0429 px。録画時 focus 332 / 較正時 focus 256 の不一致が残る |

これらは `docs/performance_results.md` の各計測エントリから参照されます。個人用メッシュは
生体情報なので、成果物にも計測結果にも実体を含めず、パスと用途だけを記録します。

## Stanford Bunny の出典・ライセンス（2026-08-17 追記）

`src/headcoupled_display/static/assets/bunny.pcd` はこれまで `scripts/generate_bunny.py`
が生成する合成点群（13,810点、楕円体サンプリングによる「bunny風」形状）でしたが、
本物の Stanford Bunny（35,947頂点、`bun_zipper` 再構成）に差し替えました。変換は
`scripts/import_stanford_bunny.py` が行い、この節はその入力データの出所と利用条件の
記録です。

| 項目 | 値 |
| :--- | :--- |
| 入力ファイル（リポジトリには追加せず、読み取り専用で参照のみ） | `/home/inaho-omen/open3d_data/extract/BunnyMesh/BunnyMesh.ply` |
| 入力ファイル SHA-256 | `b1acc63bece78444aa2e15bdcc72371a201279b98c6f5d4b74c993d02f0566fe` |
| 頂点数／三角形数 | 35,947 頂点／69,451 三角形 |
| 形式 | ASCII PLY、`comment zipper output`、プロパティ `x y z confidence intensity` |

このファイルは Open3D が配布する "BunnyMesh" サンプルデータセットの一部としてローカルに
存在していたものです。Stanford 3D Scanning Repository
（<http://graphics.stanford.edu/data/3Dscanrep/>、2026-08-17 に直接取得して確認）の
"Stanford Bunny" 項目には次の記載があり、頂点数・三角形数（35947／69451）が完全に一致する
ため、同じ再構成データであると判断しました。

> Source: Stanford University Computer Graphics Laboratory
> Scanner: Cyberware 3030 MS
> Number of scans: 10
> Reconstruction: zipper
> Size of reconstruction: 35947 vertices, 69451 triangles
> Comments: contains 5 holes in the bottom

同ページに記載されている利用条件は次のとおりです（2026-08-17 に当該ページを直接取得し、
本文を検証済みの逐語引用）。

> Please be sure to acknowledge the source of the data and models you take from this
> repository. [...] You are welcome to use the data and models for research purposes.
> You are also welcome to mirror or redistribute them for free. Finally, you may
> publish images made using these models, or the images on this web site, in a
> scholarly article or book - as long as credit is given to the Stanford Computer
> Graphics Laboratory. However, such models or images are not to be used for
> commercial purposes, nor should they appear in a product for sale (with the
> exception of scholarly journals or books), without our permission.

要約すると、出典（Stanford Computer Graphics Laboratory）を明示すれば研究目的での利用・
無償での再配布・複製は許可されており、商用利用や販売製品への組み込みには別途許可が必要、
というライセンスです。本リポジトリでの利用（研究・デモ用の頭部追従3D表示の点群アセット、
非商用）はこの許諾範囲内と判断していますが、これは記載事実からの読み手（本エージェント）
による判断であり、Stanford側からの個別の利用許可を得たものではありません。

**未確認の項目:**
- Open3D が同梱データセットとして再配布するにあたり独自のライセンス表記を追加しているか
  （Open3D 側の配布物にライセンスファイルが同梱されているかは未確認）。
- 上記ローカルファイルの入手元・入手日時（ダウンロード履歴が残っていないため、Stanford
  本家の `bunny.tar.gz` と Open3D 経由のどちらから取得されたコピーかは未確認。ただし
  頂点数・三角形数の完全一致から同一の再構成データであることは確認済み）。
- Stanford 3D Scanning Repository のページに明記された著作権者・発行日（ページ内に
  明示的な年号表記が見当たらず、確認できていない）。
