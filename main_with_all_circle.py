import subprocess
import sys
import os

TARGET_DATASETS = [       
    "challenge_dataset/articulated/orthogonal/circle",
    "challenge_dataset/balloon_vis/orthogonal/circle",
    "challenge_dataset/paper_vis/orthogonal/circle",
    "challenge_dataset/stretch/orthogonal/circle",
    "challenge_dataset/tearing_vis/orthogonal/circle",
    "challenge_dataset/articulated/perspective/circle",
    "challenge_dataset/balloon_vis/perspective/circle",
    "challenge_dataset/paper_vis/perspective/circle",
    "challenge_dataset/stretch/perspective/circle",
    "challenge_dataset/tearing_vis/perspective/circle",
    ]

BASE_DIR = "/home/gax/NRSfM_dataset"

if __name__ == "__main__":

    for dataset in TARGET_DATASETS:

        result_folder = os.path.join(BASE_DIR, dataset,"results" )
        flag_file = os.path.join(result_folder, "FINISHED_0.00001")

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