import os
import sys
import subprocess

# Add src to path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def run_script(script_name, args=[]):
    script_path = os.path.join(BASE_DIR, "src", "evaluation", script_name)
    print(f"\n{'='*60}")
    print(f"🚀 RUNNING: {script_name}")
    print(f"{'='*60}")
    
    cmd = [sys.executable, script_path] + args
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"❌ Error running {script_name}")
    return result.returncode == 0

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Run all evaluation stages",
        epilog=(
            "NOTE (Stage 5 Dataset Expansion): After the handwritten dataset was expanded from 1,000 to 10,000 "
            "samples, the proportional distribution shifted from 33:1 (SVHN:Handwritten) to 3.3:1. "
            "This means the 'Overall' metric under proportional sampling now includes ~8x more handwritten samples "
            "than before. Use --balanced for stable, cross-version comparisons."
        )
    )
    parser.add_argument("--max-samples", type=int, default=100, help="Max samples for each stage")
    parser.add_argument(
        "--balanced",
        action="store_true",
        help=(
            "Use equal/balanced split for categories (recommended for cross-version comparisons). "
            "Ensures a fair 50/50 SVHN vs Handwritten evaluation regardless of dataset folder sizes."
        )
    )
    args = parser.parse_args()

    extra_args = ["--balanced"] if args.balanced else []

    stages = [
        ("eval_global_bbox.py", ["--max-samples", str(args.max_samples)] + extra_args),
        ("eval_sharpening.py", ["--max-samples", str(args.max_samples)] + extra_args),
        ("eval_individual_bbox.py", ["--max-samples", str(args.max_samples)] + extra_args),
        ("eval_digit_recog.py", ["--max-samples", str(args.max_samples)] + extra_args),
        ("eval_pipeline.py", ["--max-samples", str(args.max_samples), "--save-viz", "--analyze-errors"] + extra_args)
    ]
    
    success_count = 0
    for script, script_args in stages:
        if run_script(script, script_args):
            success_count += 1
            
    print(f"\n{'='*60}")
    print(f"✅ EVALUATION COMPLETE: {success_count}/{len(stages)} stages succeeded")
    print(f"{'='*60}")
    print(f"Reports are available in: {os.path.join(BASE_DIR, 'outputs', 'reports')}")
    print(f"\n💡 TIP: Re-run with --balanced for a fair 50/50 SVHN vs Handwritten comparison.")

if __name__ == "__main__":
    main()
