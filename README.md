# yolov3-dlab

Darknetで学習したYOLOv3/YOLOv3-tinyモデルをKria KV260向けに量子化・コンパイルし、KV260に転送するためのツール群です。

## 概要

`deploy_model.py` は以下の4ステップを一括で実行します。

| Step | 処理内容 | 入出力 |
|------|---------|--------|
| 1 | Darknet → Keras 変換 | `.weights` → `.h5` |
| 2 | Vitis AI 量子化 | `.h5` → `quantized_*.h5` |
| 3 | Vitis AI コンパイル | `quantized_*.h5` → `.xmodel` |
| 4 | KV260へ転送 | `.xmodel`, `.prototxt`, `meta.json`, `md5sum.txt` → KV260 |

## ディレクトリ構成

```
yolov3-dlab/
  work/                     # Darknet cfgファイル, prototxtファイル
    backup/                 # Darknet weightsファイル
    yolov3.prototxt         # yolov3用prototxt
    yolov3-tiny.prototxt    # yolov3-tiny用prototxt
  keras_model/              # 変換後のKerasモデル (.h5)
  yolov3_quantized2/        # 量子化済みモデル
  compiled_yolov3/          # コンパイル済みxmodel, meta.json, md5sum.txt
  scripts_tf2/
    deploy_model.py         # メインスクリプト
    convert_model.py        # 変換・量子化・コンパイルのみ (旧スクリプト)
    quantizer_func.py       # 量子化関数
    utils.py                # ユーティリティ
```

## 環境準備

### 1. Vitis AI Docker の起動

```bash
cd ~/Vitis-AI2.5
./docker_run.sh my_vitis-ai-cpu:latest
```

### 2. conda 仮想環境の有効化

```bash
conda activate vitis-ai-tensorflow2
```

### 3. 入力ファイルの配置

- **cfgファイル** を `yolov3-dlab/work/` に配置
- **weightsファイル** を `yolov3-dlab/work/backup/` に配置

## deploy_model.py の使い方

### 基本コマンド

```bash
cd /workspace/yolov3-dlab/scripts_tf2
python deploy_model.py --config_name <cfgファイル名> --weights_name <weightsファイル名>
```

### コマンドライン引数

| 引数 | 必須 | 説明 |
|------|------|------|
| `--config_name` | Yes | `work/` 配下の Darknet cfg ファイル名 |
| `--weights_name` | Yes | `work/backup/` 配下の Darknet weights ファイル名 |
| `--compiled_dir` | No | コンパイル済み xmodel の出力先 (default: `../compiled_yolov3`) |
| `--transfer-only` | No | Step 1-3 をスキップし、既存の xmodel を KV260 に転送のみ行う |

### 実行例: フルパイプライン

```bash
python deploy_model.py --config_name yolov3-tiny-416.cfg --weights_name yolov3-tiny-416_final.weights
```

Step 1-3 が順に実行され、Step 4 でKV260への転送に進みます。

### 実行例: 転送のみ

既にコンパイル済みの xmodel がある場合、`--transfer-only` で転送のみ実行できます。

```bash
python deploy_model.py --transfer-only --config_name yolov3-tiny-416.cfg --weights_name yolov3-tiny-416_final.weights
```

## KV260 への転送 (Step 4)

Step 4 では以下の流れで転送を行います。

### 1. 接続チェック

KV260 (`root@192.168.1.100`) への SSH 接続を確認します。接続できない場合はエラーを表示して停止します。パスワードは最初の1回のみ入力が必要です (SSH ControlMaster による接続共有)。

### 2. モデル名の指定

```
KV260上でのモデル名 [yolov3-tiny-416]: 
```

- **Enter** を押す → デフォルト名 (`yolov3-tiny-416`) を使用
- **別名を入力** → 入力した名前を使用

デフォルト名は weights ファイル名の最後の `_` より前の部分から自動生成されます。
例: `yolov3-tiny-416_final.weights` → `yolov3-tiny-416`

### 3. 転送確認

```
  転送先: root@192.168.1.100:/usr/share/vitis_ai_library/models/yolov3-tiny-416/
  KV260へ転送しますか？ [y/N]: 
```

`y` を入力すると転送を実行します。それ以外はスキップします。

### 4. 転送されるファイル

モデル名が `yolov3-tiny-416` の場合、KV260 の `/usr/share/vitis_ai_library/models/yolov3-tiny-416/` に以下が転送されます。

| ファイル | 説明 |
|---------|------|
| `yolov3-tiny-416.xmodel` | コンパイル済み量子化モデル |
| `yolov3-tiny-416.prototxt` | モデル定義ファイル (tiny の場合 `yolov3-tiny.prototxt` から自動選択) |
| `meta.json` | メタ情報 |
| `md5sum.txt` | チェックサム |
