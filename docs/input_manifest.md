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
