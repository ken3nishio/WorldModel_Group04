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
    parser.add_argument("--image", required=True, help="入力画像の絶対パス")
    parser.add_argument("--beta", type=float, default=0.0, help="Adaptive CFG Beta (Default: 0.0)")
    parser.add_argument("--blur", type=float, default=0.0, help="Temporal Blur Sigma (Default: 0.0)")
    parser.add_argument("--length", type=int, default=5, help="動画の長さ（秒）。内部でセクション数に変換されます (Default: 5)")
    parser.add_argument("--seed", type=int, default=31337, help="乱数シード (Default: 31337)")
    parser.add_argument("--device", default="cuda", help="実行デバイス (cuda / cpu)")
    parser.add_argument("--note", default="default", help="実験名のsuffix（メモ用）")
    
    args = parser.parse_args()

    # セクション数の計算 (1 section ≈ 1.1s @ 30fps)
    # length (sec) / 1.1 => sections
    sections = max(1, int(args.length / 1.1 + 0.5))

    # 出力ディレクトリ名の構築: timestamp_beta_XX_blur_YY_note
    output_base = os.path.join(PROJECT_ROOT, "experiments/runs")
    
    cmd = [
        "python", GENERATE_SCRIPT,
        "--input_image", args.image,
        "--beta", str(args.beta),
        "--blur", str(args.blur),
        "--sections", str(sections),
        "--seed", str(args.seed),
        "--length", str(args.length),
        "--output_dir", output_base,
        "--device", args.device,
    ]
    
    if args.prompt:
        cmd.extend(["--prompt", args.prompt])

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
