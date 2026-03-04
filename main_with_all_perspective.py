import subprocess
import sys
import os

TARGET_DATASETS = [       
    "challenge_dataset/articulated/perspective/flyby",
        "challenge_dataset/articulated/perspective/line",
        "challenge_dataset/articulated/perspective/semi_circle",
        "challenge_dataset/articulated/perspective/tricky",
        "challenge_dataset/articulated/perspective/zigzag",
        "challenge_dataset/paper_vis/perspective/circle",
        "challenge_dataset/paper_vis/perspective/flyby",
        "challenge_dataset/paper_vis/perspective/line",
        "challenge_dataset/paper_vis/perspective/semi_circle",
        "challenge_dataset/paper_vis/perspective/tricky",
        "challenge_dataset/paper_vis/perspective/zigzag",
        "challenge_dataset/stretch/perspective/circle",
        "challenge_dataset/stretch/perspective/flyby",
        "challenge_dataset/stretch/perspective/line",
        "challenge_dataset/stretch/perspective/semi_circle",
        "challenge_dataset/stretch/perspective/tricky",
        "challenge_dataset/stretch/perspective/zigzag",
        "challenge_dataset/tearing_vis/perspective/circle",
        "challenge_dataset/tearing_vis/perspective/flyby",
        "challenge_dataset/tearing_vis/perspective/line",
        "challenge_dataset/tearing_vis/perspective/semi_circle",
        "challenge_dataset/tearing_vis/perspective/tricky",
        "challenge_dataset/tearing_vis/perspective/zigzag",]

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
                [sys.executable, "main.py","--resume", ], #
                check=True,
                env=env
            )

            # 成功后写标记
            os.makedirs(result_folder, exist_ok=True)
            open(flag_file, "w").close()

        except subprocess.CalledProcessError:
            print(f"❌ Error in {dataset}, will retry next time.")
            continue