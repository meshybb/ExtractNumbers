import os
import sys
import subprocess

# Add src to path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def run_script(script_name, args=[]):
    script_path = os.path.join(BASE_DIR, "src", "evaluation", script_name)
    print(f"\n{'='*60}")
    print(f"🚀 RUNNING VIDEO EVALUATION: {script_name}")
    print(f"{'='*60}")
    
    cmd = [sys.executable, script_path] + args
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"❌ Error running {script_name}")
    return result.returncode == 0

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Run all video evaluation stages"
    )
    parser.add_argument("--max-samples", type=int, default=10, help="Max video samples to evaluate")
    parser.add_argument(
        "--balanced",
        action="store_true",
        help="Use equal/balanced split for video categories"
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="annotated",
        choices=["annotated", "uniform", "motion_and_blur", "random_1_in_10"],
        help="Frame selection strategy"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(BASE_DIR, "outputs"),
        help="Base directory for outputs"
    )
    args = parser.parse_args()

    extra_args = ["--balanced"] if args.balanced else []
    extra_args += ["--strategy", args.strategy]
    extra_args += ["--output-dir", args.output_dir]

    stages = [
        ("eval_video_global_bbox.py", ["--max-samples", str(args.max_samples)] + extra_args),
        ("eval_video_individual_bbox.py", ["--max-samples", str(args.max_samples)] + extra_args),
        ("eval_video_digit_recog.py", ["--max-samples", str(args.max_samples)] + extra_args),
        ("eval_video_pipeline.py", ["--max-samples", str(args.max_samples), "--save-viz", "--analyze-errors"] + extra_args)
    ]
    
    success_count = 0
    for script, script_args in stages:
        if run_script(script, script_args):
            success_count += 1
            
    print(f"\n{'='*60}")
    print(f"✅ VIDEO EVALUATION COMPLETE: {success_count}/{len(stages)} stages succeeded")
    print(f"{'='*60}")
    print(f"Reports are available in: {os.path.join(args.output_dir, 'reports')}")

if __name__ == "__main__":
    main()
