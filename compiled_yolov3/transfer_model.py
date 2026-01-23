#!/usr/bin/env python3
"""
Vitis AI モデルファイル転送スクリプト

転送元: ローカルの3ファイル
  1. *.xmodel (例: yolov3-tiny-416_final.xmodel)
     → 最後の'_'より前の部分をモデル名として使用
     → yolov3-tiny-416.xmodel にリネーム
  2. meta.json (そのまま)
  3. md5sum.txt (そのまま)

転送先: root@192.168.1.100:/usr/share/vitis_ai_library/models/<モデル名>/

例:
  yolov3-tiny-416_final.xmodel → models/yolov3-tiny-416/yolov3-tiny-416.xmodel
  yolov3-512_last.xmodel       → models/yolov3-512/yolov3-512.xmodel
"""

import subprocess
import sys
import argparse
from pathlib import Path
from typing import List


# 転送先設定（固定）
REMOTE_HOST = "root@192.168.1.100"
REMOTE_BASE_DIR = "/usr/share/vitis_ai_library/models"


def extract_model_name(xmodel_filename: str) -> str:
    """xmodelファイル名から最後の'_'より前の部分をモデル名として抽出"""
    # 拡張子を除去
    stem = Path(xmodel_filename).stem  # yolov3-tiny-416_final
    # 最後の'_'で分割して前の部分を取得
    if '_' in stem:
        model_name = stem.rsplit('_', 1)[0]  # yolov3-tiny-416
    else:
        model_name = stem
    return model_name


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
    
    # モデル名を抽出
    model_name = extract_model_name(xmodel.name)
    remote_dir = f"{REMOTE_BASE_DIR}/{model_name}"
    
    for f, name in [(xmodel, "xmodel"), (meta, "meta.json"), (md5, "md5sum.txt")]:
        if not f.exists():
            print(f"エラー: {name} が見つかりません: {f}")
            return False
    
    print(f"\n{'='*60}")
    print(f"転送設定")
    print(f"{'='*60}")
    print(f"モデル名: {model_name}")
    print(f"転送元:")
    print(f"  xmodel:    {xmodel} → {model_name}.xmodel")
    print(f"  meta.json: {meta}")
    print(f"  md5sum:    {md5}")
    print(f"転送先: {REMOTE_HOST}:{remote_dir}/")
    print(f"{'='*60}\n")
    
    # 1. リモートディレクトリ作成
    if not run_command(
        ["ssh", REMOTE_HOST, f"mkdir -p {remote_dir}"],
        "リモートディレクトリ作成"
    ):
        return False
    
    # 2. xmodel転送（リネームあり）
    if not run_command(
        ["scp", str(xmodel), f"{REMOTE_HOST}:{remote_dir}/{model_name}.xmodel"],
        f"xmodel転送 ({xmodel.name} → {model_name}.xmodel)"
    ):
        return False
    
    # 3. meta.json転送
    if not run_command(
        ["scp", str(meta), f"{REMOTE_HOST}:{remote_dir}/meta.json"],
        "meta.json転送"
    ):
        return False
    
    # 4. md5sum.txt転送
    if not run_command(
        ["scp", str(md5), f"{REMOTE_HOST}:{remote_dir}/md5sum.txt"],
        "md5sum.txt転送"
    ):
        return False
    
    print(f"\n{'='*60}")
    print("✓ 全ファイルの転送が完了しました")
    print(f"{'='*60}")
    
    # 転送結果確認
    print("\n[確認] リモートディレクトリの内容:")
    subprocess.run(["ssh", REMOTE_HOST, f"ls -la {remote_dir}/"])
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Vitis AI モデルファイルをKria KV260に転送",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  python transfer_model.py yolov3-tiny-416_final.xmodel
    → models/yolov3-tiny-416/yolov3-tiny-416.xmodel として転送

  python transfer_model.py yolov3-512_last.xmodel
    → models/yolov3-512/yolov3-512.xmodel として転送

※ meta.json と md5sum.txt は xmodel と同じディレクトリから自動的に検出されます
※ モデル名はxmodelファイル名の最後の'_'より前の部分が使われます
        """
    )
    
    parser.add_argument("xmodel", help="xmodelファイルのパス（任意の名前可）")
    
    args = parser.parse_args()
    
    success = transfer_model(args.xmodel)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
