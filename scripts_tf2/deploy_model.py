#!/usr/bin/env python3
"""
Darknet → Keras → 量子化 → コンパイル → KV260転送 一括デプロイスクリプト

使用例:
  python deploy_model.py --config_name yolov3-tiny-416.cfg --weights_name yolov3-tiny-416_final.weights
  python deploy_model.py --config_name yolov3-512.cfg --weights_name yolov3-512_last.weights --compiled_dir /tmp/out

KV260転送時のモデル名:
  デフォルトではweightsファイル名の最後の'_'より前の部分を使用
  (例: yolov3-tiny-416_final.weights → yolov3-tiny-416)
  プロンプトで別名を入力することも可能
"""

import os
import sys
import subprocess
import tempfile
import atexit
from pathlib import Path
from typing import List

from utils import getval_cfg
from quantizer_func import quantizer, CALIB_DATASET

# ---- 転送先設定 ----
REMOTE_HOST = "root@192.168.1.100"
REMOTE_BASE_DIR = "/usr/share/vitis_ai_library/models"

# SSH ControlMaster 用
_ctrl_socket = None

NUMPY_MAGIC = b'\x93NUMPY'


def validate_calib_dataset() -> bool:
    """キャリブレーションデータセット(.npy)がNumPy形式か検証する"""
    if not os.path.exists(CALIB_DATASET):
        print(f"エラー: キャリブレーションデータが見つかりません: {CALIB_DATASET}")
        return False

    with open(CALIB_DATASET, 'rb') as f:
        header = f.read(6)

    if header != NUMPY_MAGIC:
        print(f"エラー: {CALIB_DATASET} は正しいNumPyファイルではありません")
        print(f"  先頭バイト: {header}")
        print(f"  期待値:     {NUMPY_MAGIC}")
        print(f"  Git LFSポインタ等の可能性があります。正しいファイルに置き換えてください。")
        return False

    print(f"  ✓ キャリブレーションデータ検証OK: {CALIB_DATASET}")
    return True


def ssh_control_options() -> List[str]:
    return ["-o", f"ControlPath={_ctrl_socket}",
            "-o", "ControlMaster=auto",
            "-o", "ControlPersist=60"]


def close_ssh_connection():
    if _ctrl_socket is None:
        return
    subprocess.run(
        ["ssh", "-o", f"ControlPath={_ctrl_socket}", "-O", "exit", REMOTE_HOST],
        capture_output=True
    )
    sock_dir = Path(str(_ctrl_socket)).parent
    if sock_dir.exists():
        import shutil
        shutil.rmtree(sock_dir, ignore_errors=True)


def run_ssh_command(cmd: List[str], description: str) -> bool:
    print(f"[実行中] {description}")
    print(f"  コマンド: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"  ✓ 成功")
        return True
    else:
        print(f"  ✗ 失敗: {result.stderr.strip()}")
        return False


def extract_model_name(weights_name: str) -> str:
    """weightsファイル名から最後の'_'より前の部分をモデル名として抽出"""
    stem = Path(weights_name).stem
    if '_' in stem:
        return stem.rsplit('_', 1)[0]
    return stem


def remove_known_host_entry():
    """known_hostsからKV260のホストキーを削除する（再起動時のキー変更対策）"""
    host_ip = REMOTE_HOST.split("@")[-1]
    result = subprocess.run(
        ["ssh-keygen", "-R", host_ip],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        print(f"  known_hostsから {host_ip} のエントリを削除しました")
    else:
        print(f"  known_hostsに {host_ip} のエントリはありませんでした")


def check_and_start_ssh_connection() -> bool:
    """KV260への接続チェックとControlMaster確立を兼ねる（パスワード入力1回）"""
    global _ctrl_socket
    _ctrl_socket = Path(tempfile.mkdtemp()) / "ssh_ctrl_%h_%p_%r"

    print(f"[確認] {REMOTE_HOST} への接続チェック中...")
    remove_known_host_entry()
    cmd = ["ssh", "-o", "ConnectTimeout=5",
           "-o", "StrictHostKeyChecking=accept-new"] + \
          ssh_control_options() + ["-o", "ControlMaster=yes",
          "-N", "-f", REMOTE_HOST]
    result = subprocess.run(cmd)
    if result.returncode == 0:
        print(f"  ✓ {REMOTE_HOST} に接続しました（以降パスワード不要）")
        atexit.register(close_ssh_connection)
        return True
    else:
        print(f"  ✗ {REMOTE_HOST} に接続できません")
        print(f"    KV260の電源やネットワーク、SSH設定を確認してください")
        return False


def select_prototxt(config_name: str) -> Path:
    """cfgファイル名からyolov3用/yolov3-tiny用のprototxtを選択"""
    prototxt_dir = Path("../work")
    if "tiny" in config_name:
        return prototxt_dir / "yolov3-tiny.prototxt"
    else:
        return prototxt_dir / "yolov3.prototxt"


def transfer_model(compiled_dir: str, compiled_name: str, model_name: str, prototxt: Path) -> bool:
    """コンパイル済みモデルをKV260に転送"""
    compiled_dir = Path(compiled_dir)
    xmodel = compiled_dir / (compiled_name + ".xmodel")
    meta = compiled_dir / "meta.json"
    md5 = compiled_dir / "md5sum.txt"

    remote_dir = f"{REMOTE_BASE_DIR}/{model_name}"

    for f, name in [(xmodel, "xmodel"), (meta, "meta.json"), (md5, "md5sum.txt"), (prototxt, "prototxt")]:
        if not f.exists():
            print(f"エラー: {name} が見つかりません: {f}")
            return False

    print(f"\n{'='*60}")
    print(f"転送設定")
    print(f"{'='*60}")
    print(f"モデル名: {model_name}")
    print(f"転送元:")
    print(f"  xmodel:    {xmodel} → {model_name}.xmodel")
    print(f"  prototxt:  {prototxt} → {model_name}.prototxt")
    print(f"  meta.json: {meta}")
    print(f"  md5sum:    {md5}")
    print(f"転送先: {REMOTE_HOST}:{remote_dir}/")
    print(f"{'='*60}\n")

    ctrl = ssh_control_options()

    if not run_ssh_command(
        ["ssh"] + ctrl + [REMOTE_HOST, f"mkdir -p {remote_dir}"],
        "リモートディレクトリ作成"
    ):
        return False

    if not run_ssh_command(
        ["scp"] + ctrl + [str(xmodel), f"{REMOTE_HOST}:{remote_dir}/{model_name}.xmodel"],
        f"xmodel転送 ({xmodel.name} → {model_name}.xmodel)"
    ):
        return False

    if not run_ssh_command(
        ["scp"] + ctrl + [str(prototxt), f"{REMOTE_HOST}:{remote_dir}/{model_name}.prototxt"],
        f"prototxt転送 ({prototxt.name} → {model_name}.prototxt)"
    ):
        return False

    if not run_ssh_command(
        ["scp"] + ctrl + [str(meta), f"{REMOTE_HOST}:{remote_dir}/meta.json"],
        "meta.json転送"
    ):
        return False

    if not run_ssh_command(
        ["scp"] + ctrl + [str(md5), f"{REMOTE_HOST}:{remote_dir}/md5sum.txt"],
        "md5sum.txt転送"
    ):
        return False

    print(f"\n{'='*60}")
    print("✓ 全ファイルの転送が完了しました")
    print(f"{'='*60}")

    print("\n[確認] リモートディレクトリの内容:")
    subprocess.run(["ssh"] + ctrl + [REMOTE_HOST, f"ls -la {remote_dir}/"])

    return True


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Darknet weights → 量子化 → コンパイル → KV260転送',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # フルパイプライン（変換→量子化→コンパイル→転送）
  python deploy_model.py --config_name yolov3-tiny-416.cfg --weights_name yolov3-tiny-416_final.weights

  # 転送のみ（既存のxmodelをKV260へ転送）
  python deploy_model.py --transfer-only --config_name yolov3-tiny-416.cfg --weights_name yolov3-tiny-416_final.weights
        """
    )
    parser.add_argument('--config_name', required=True, help='Darknet cfg file name.')
    parser.add_argument('--weights_name', required=True, help='Darknet weights file name.')
    parser.add_argument('--compiled_dir', default='../compiled_yolov3',
                        help='Output directory for compiled xmodel (default: ../compiled_yolov3).')
    parser.add_argument('--transfer-only', action='store_true',
                        help='Skip conversion/quantization/compile and transfer existing xmodel to KV260.')
    args = parser.parse_args()

    compiled_dir = args.compiled_dir
    compiled_name = os.path.splitext(args.weights_name)[0]

    if not args.transfer_only:
        # ---- デフォルトディレクトリ ----
        config_dir = "../work"
        weights_dir = "../work/backup"
        out_dir = "../keras_model"

        # ==================================================
        # Step 1: Darknet → Keras 変換
        # ==================================================
        print("\n" + "=" * 60)
        print("Step 1: Darknet → Keras 変換")
        print("=" * 60)

        config_path = os.path.join(config_dir, args.config_name)
        if not os.path.exists(config_path):
            print(f'ERROR: {config_path} does NOT exist.')
            exit(1)

        weights_path = os.path.join(weights_dir, args.weights_name)
        if not os.path.exists(weights_path):
            print(f'ERROR: {weights_path} does NOT exist.')
            exit(1)

        h5_name = os.path.splitext(args.weights_name)[0] + ".h5"
        h5_path = os.path.join(out_dir, h5_name)
        if not os.path.exists(out_dir):
            print(f"ERROR: {out_dir} does NOT exist.")
            exit(1)
        if os.path.exists(h5_path):
            os.remove(h5_path)

        keras_converter = "../keras-YOLOv3-model-set/tools/model_converter/convert.py"
        keras_arguments = " ".join([config_path, weights_path, h5_path])
        cmd = " ".join(["python3", keras_converter, keras_arguments])
        print(cmd)
        os.system(cmd)

        if not os.path.exists(h5_path):
            print(f'ERROR: Could not convert into keras_model [{h5_path}].\n')
            exit(1)
        print(f'#### Successfully converted darknet model into keras model {h5_path} ####\n')

        # ==================================================
        # Step 2: 量子化
        # ==================================================
        print("=" * 60)
        print("Step 2: 量子化")
        print("=" * 60)

        if not validate_calib_dataset():
            exit(1)

        width = getval_cfg(config_path, 'width')
        height = getval_cfg(config_path, 'height')

        quantized_dir = "../yolov3_quantized2"
        quantized_name = "quantized_" + h5_name
        quantized_path = os.path.join(quantized_dir, quantized_name)
        if os.path.exists(quantized_path):
            os.remove(quantized_path)

        quantizer(keras_model=h5_path, quantized_path=quantized_path, in_shape=[height, width, 3])
        if not os.path.exists(quantized_path):
            print(f'ERROR: Could not make quantized file [{quantized_path}].\n')
            exit(1)
        print(f'#### Successfully made quantized model {quantized_path} ####\n')

        # ==================================================
        # Step 3: コンパイル
        # ==================================================
        print("=" * 60)
        print("Step 3: コンパイル")
        print("=" * 60)

        arch = "/opt/vitis_ai/compiler/arch/DPUCZDX8G/KV260/arch.json"
        if not os.path.exists(compiled_dir):
            os.makedirs(compiled_dir)

        last_arg = f"{{'mode':'normal','save_kernel':'', 'input_shape':'1,{height},{width},3'}}"
        last_arg = '"' + last_arg + '"'
        compile_args = f"-m {quantized_path} -a {arch} -o {compiled_dir} -n {compiled_name} -e {last_arg}"
        cmd = f"vai_c_tensorflow2 {compile_args}"
        print(cmd)
        os.system(cmd)

    compiled_path = os.path.join(compiled_dir, compiled_name + ".xmodel")
    if not os.path.exists(compiled_path):
        print(f'ERROR: xmodel not found [{compiled_path}].\n')
        exit(1)

    if not args.transfer_only:
        print(f'#### Successfully compiled quantized model {compiled_path} ####\n')

    # ==================================================
    # Step 4: KV260へ転送
    # ==================================================
    print("=" * 60)
    print("KV260へ転送")
    print("=" * 60)

    if not check_and_start_ssh_connection():
        print(f"\nKV260への転送をスキップしました。")
        print(f"コンパイル済みモデル: {compiled_path}")
        exit(1)

    default_model_name = extract_model_name(args.weights_name)
    user_input = input(f"KV260上でのモデル名 [{default_model_name}]: ").strip()
    model_name = user_input if user_input else default_model_name

    print(f"\n  転送先: {REMOTE_HOST}:{REMOTE_BASE_DIR}/{model_name}/")
    confirm = input("  KV260へ転送しますか？ [y/N]: ").strip().lower()
    if confirm != 'y':
        print("転送をスキップしました。")
        print(f"コンパイル済みモデル: {compiled_path}")
        exit(0)

    prototxt = select_prototxt(args.config_name)
    if not transfer_model(compiled_dir, compiled_name, model_name, prototxt):
        exit(1)

    print(f'\n#### デプロイ完了: {REMOTE_HOST}:{REMOTE_BASE_DIR}/{model_name}/ ####\n')


if __name__ == "__main__":
    main()
