import subprocess
import sys
import os

TARGET_DATASETS = [       
    "challenge_dataset/balloon_vis/perspective/circle",
    "challenge_dataset/balloon_vis/perspective/flyby",
    "challenge_dataset/balloon_vis/perspective/line",       
    "challenge_dataset/balloon_vis/perspective/semi_circle",
    "challenge_dataset/balloon_vis/perspective/tricky",
    "challenge_dataset/balloon_vis/perspective/zigzag",
    "challenge_dataset/balloon_vis/orthogonal/circle",
    "challenge_dataset/balloon_vis/orthogonal/flyby",
    "challenge_dataset/balloon_vis/orthogonal/line",       
    "challenge_dataset/balloon_vis/orthogonal/semi_circle",
    "challenge_dataset/balloon_vis/orthogonal/tricky",
    "challenge_dataset/balloon_vis/orthogonal/zigzag",
    "challenge_dataset/paper_vis/orthogonal/circle",
    "challenge_dataset/paper_vis/orthogonal/flyby",
    "challenge_dataset/paper_vis/orthogonal/line", ]

BASE_DIR = "/home/gax/NRSfM_dataset"

if __name__ == "__main__":

    for dataset in TARGET_DATASETS:

        result_folder = os.path.join(BASE_DIR, dataset,"results" )
        flag_file = os.path.join(result_folder, "FINISHED_2")

        if os.path.exists(flag_file):
            print(f"⏩ Skipping {dataset}, already finished.")
            continue

        print(f"🚀 Running {dataset}")

        env = os.environ.copy()
        env["DATASET_NAME"] = dataset

        try:
            subprocess.run(
                [sys.executable, "main.py",], #"--resume"
                check=True,
                env=env
            )

            # 成功后写标记
            os.makedirs(result_folder, exist_ok=True)
            open(flag_file, "w").close()

        except subprocess.CalledProcessError:
            print(f"❌ Error in {dataset}, will retry next time.")
            continue