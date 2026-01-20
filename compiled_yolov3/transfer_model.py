#!/usr/bin/env python3
"""
YOLOv3-tiny モデルファイル転送スクリプト

転送元: ローカルの3ファイル
  1. *.xmodel (任意の名前) → yolov3-tiny-416.xmodel にリネーム
  2. meta.json (そのまま)
  3. md5sum.txt (そのまま)

転送先: root@192.168.1.100:/usr/share/vitis_ai_library/models/yolov3-tiny-416/
"""

import subprocess
import sys
import argparse
from pathlib import Path
from typing import List


# 転送先設定（固定）
REMOTE_HOST = "root@192.168.1.100"
REMOTE_BASE_DIR = "/usr/share/vitis_ai_library/models"
MODEL_NAME = "yolov3-tiny-416"
REMOTE_DIR = f"{REMOTE_BASE_DIR}/{MODEL_NAME}"


def run_command(cmd: List[str], description: str) -> bool:
    """コマンドを実行し、結果を表示"""
    print(f"[実行中] {description}")
    print(f"  コマンド: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"  ✓ 成功")
        return True
    else:
        print(f"  ✗ 失敗: {result.stderr.strip()}")
        return False


def transfer_model(xmodel_path: str) -> bool:
    """モデルファイルを転送"""
    
    # パスの検証
    xmodel = Path(xmodel_path)
    model_dir = xmodel.parent
    meta = model_dir / "meta.json"
    md5 = model_dir / "md5sum.txt"
    
    for f, name in [(xmodel, "xmodel"), (meta, "meta.json"), (md5, "md5sum.txt")]:
        if not f.exists():
            print(f"エラー: {name} が見つかりません: {f}")
            return False
    
    print(f"\n{'='*60}")
    print(f"転送設定")
    print(f"{'='*60}")
    print(f"転送元:")
    print(f"  xmodel:    {xmodel} → {MODEL_NAME}.xmodel")
    print(f"  meta.json: {meta}")
    print(f"  md5sum:    {md5}")
    print(f"転送先: {REMOTE_HOST}:{REMOTE_DIR}/")
    print(f"{'='*60}\n")
    
    # 1. リモートディレクトリ作成
    if not run_command(
        ["ssh", REMOTE_HOST, f"mkdir -p {REMOTE_DIR}"],
        "リモートディレクトリ作成"
    ):
        return False
    
    # 2. xmodel転送（リネームあり）
    if not run_command(
        ["scp", str(xmodel), f"{REMOTE_HOST}:{REMOTE_DIR}/{MODEL_NAME}.xmodel"],
        f"xmodel転送 ({xmodel.name} → {MODEL_NAME}.xmodel)"
    ):
        return False
    
    # 3. meta.json転送
    if not run_command(
        ["scp", str(meta), f"{REMOTE_HOST}:{REMOTE_DIR}/meta.json"],
        "meta.json転送"
    ):
        return False
    
    # 4. md5sum.txt転送
    if not run_command(
        ["scp", str(md5), f"{REMOTE_HOST}:{REMOTE_DIR}/md5sum.txt"],
        "md5sum.txt転送"
    ):
        return False
    
    print(f"\n{'='*60}")
    print("✓ 全ファイルの転送が完了しました")
    print(f"{'='*60}")
    
    # 転送結果確認
    print("\n[確認] リモートディレクトリの内容:")
    subprocess.run(["ssh", REMOTE_HOST, f"ls -la {REMOTE_DIR}/"])
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="YOLOv3-tiny モデルファイルをKria KV260に転送",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  python transfer_model.py yolov3-tiny-416_final.xmodel
  python transfer_model.py ./models/best.xmodel

※ meta.json と md5sum.txt は xmodel と同じディレクトリから自動的に検出されます
        """
    )
    
    parser.add_argument("xmodel", help="xmodelファイルのパス（任意の名前可）")
    
    args = parser.parse_args()
    
    success = transfer_model(args.xmodel)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
