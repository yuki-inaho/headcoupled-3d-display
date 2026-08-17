# facemesh_tracking - task runner
# Usage: just <command>

# 出力先
out := "outputs"

# レシピ一覧
default:
    @just --list

# ===== セットアップ =====

# 依存をインストール
setup:
    uv sync

# 環境診断: GPU / ORT プロバイダ / モデル / カメラを確認
doctor:
    @echo "=== GPU ==="
    @nvidia-smi --query-gpu=name,driver_version,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv || echo "nvidia-smi なし"
    @echo ""
    @echo "=== ONNX Runtime ==="
    @uv run python -c "\
    from facemesh_tracking import preload_cuda_libraries; preload_cuda_libraries(); \
    import onnxruntime as ort; \
    print('version  :', ort.__version__); \
    print('providers:', ort.get_available_providers())"
    @echo ""
    @echo "=== models/ ==="
    @ls -lh models/ 2>/dev/null || echo "models/ はまだ空 (初回実行時に自動取得されます)"
    @echo ""
    @echo "=== camera ==="
    @ls /dev/video* 2>/dev/null || echo "カメラデバイスなし"

# ===== 実行 =====
# 初回推論は Pascal の PTX JIT で数秒かかります (2 枚目以降は通常速度)

# USB カメラをリアルタイム表示 (ESC で終了)
cam DEVICE="0":
    uv run facemesh run --source {{DEVICE}} --show --width 640 --height 480

# 画像を推論して outputs/ に保存 (例: just image ~/Pictures/face.jpg)
image IMG:
    uv run facemesh run --source {{IMG}} --output {{out}}/$(basename {{IMG}} | sed 's/\.[^.]*$//')_mesh.png
    @echo "-> {{out}}/"

# 画像を推論してウィンドウ表示 (任意のキーで閉じる)
show IMG:
    uv run facemesh run --source {{IMG}} --show

# 動画を推論して mp4 に保存 (例: just video ~/Videos/clip.mp4)
video VID:
    uv run facemesh run --source {{VID}} --output {{out}}/mesh.mp4
    @echo "-> {{out}}/mesh.mp4"

# 黒背景に輪郭+点のみ描画 (メッシュの当たり具合を確認したいとき)
inspect IMG:
    uv run facemesh run --source {{IMG}} --mode partial --no-background --output {{out}}/inspect.png
    @echo "-> {{out}}/inspect.png"

# ランドマークを JSON で保存
json SRC:
    uv run facemesh run --source {{SRC}} --save-json {{out}}/landmarks.json
    @echo "-> {{out}}/landmarks.json"

# ===== 計測 =====

# 段階別の速度計測
bench IMG="" ITERS="30":
    uv run facemesh bench {{ if IMG != "" { "--source " + IMG } else { "" } }} --iterations {{ITERS}}

# CPU と CUDA を比較
bench-backends IMG ITERS="20":
    @echo "### cuda ###"
    @uv run facemesh bench --backend cuda --source {{IMG}} --iterations {{ITERS}}
    @echo ""
    @echo "### cpu ###"
    @uv run facemesh bench --backend cpu --source {{IMG}} --iterations {{ITERS}}

# ===== 開発 =====

# テスト
test:
    uv run pytest -q

# lint
lint:
    uv run ruff check .
    uv run ruff format --check .

# フォーマット適用 + 自動修正
fmt:
    uv run ruff format .
    uv run ruff check --fix .

# lint + test をまとめて
check: lint test

# 出力を削除
clean:
    rm -rf {{out}}
