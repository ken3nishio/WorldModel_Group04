import sys
import os
import argparse
import subprocess
from datetime import datetime

# プロジェクトルート
PROJECT_ROOT = ""
GENERATE_SCRIPT = os.path.join(PROJECT_ROOT, "evaluation/generate_and_evaluate.py")

def main():
    parser = argparse.ArgumentParser(description="単一動画の生成・評価実験を実行し、整理されたフォルダに出力します。")
    parser.add_argument("")
    parser.add_argument("--image", required=True, help="入力画像の絶対パス")
    parser.add_argument("--beta", type=float, default=0.0, help="Adaptive CFG Beta (Default: 0.0)")
    parser.add_argument("--blur", type=float, default=0.0, help="Temporal Blur Sigma (Default: 0.0)")
    parser.add_argument("--seed", type=int, default=31337, help="乱数シード (Default: 31337)")
    parser.add_argument("--device", default="cuda", help="実行デバイス (cuda / cpu)")
    parser.add_argument("--note", default="default", help="実験名のsuffix（メモ用）")
    
    args = parser.parse_args()

    # 出力ディレクトリ名の構築: timestamp_beta_XX_blur_YY_note
    # generate_and_evaluate.py は output_dir/run_timestamp を作るので、
    # ここでは親ディレクトリを指定し、run_timestamp を期待する形になりますが、
    # わかりやすくするために output_dir 自体を実験名にしたいところです。
    # しかし generate_and_evaluate.py の仕様上、output_dir の中にさらに run_timestamp フォルダを作ります。
    # なので、ここでは experiments/runs を指定し、生成されるフォルダ名はタイムスタンプ任せになります。
    # 
    # 修正: generate_and_evaluate.py は output_dir/run_<timestamp> を生成します。
    # ユーザーが見つけやすいように、シンボリックリンクを貼るか、
    # あるいは generate_and_evaluate.py の出力先ロジックに依存します。
    
    # ここではシンプルに、experiments/runs直下に出力させます。
    output_base = os.path.join(PROJECT_ROOT, "experiments/runs")
    
    cmd = [
        "python3", GENERATE_SCRIPT,
        "--input_image", args.image,
        "--beta", str(args.beta),
        "--blur", str(args.blur),
        "--seed", str(args.seed),
        "--output_dir", output_base,
        "--device", args.device,
    ]

    print(f"\n🚀 実験を開始します")
    print(f"📂 出力ベース: {output_base}")
    print(f"🔧 パラメータ: β={args.beta}, Blur={args.blur}, Device={args.device}")
    print("=" * 60)
    
    try:
        subprocess.run(cmd, check=True)
        print("\n✅ 実験完了！")
        print(f"結果は {output_base} 内の最新のフォルダを確認してください。")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ エラーが発生しました (Exit Code: {e.returncode})")
        sys.exit(e.returncode)

if __name__ == "__main__":
    main()
