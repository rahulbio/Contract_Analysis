import os
import gdown

MODEL_DIR = "resources"
GDRIVE_FOLDER_ID = "1hanu85tSennwbLadJsgEnIxF0ZrvniHL"

def ensure_models():
    """
    Downloads model artifacts from Google Drive
    only if they are not already present.
    """

    required_paths = [
        "resources/deberta-clause-final",
        "resources/clause_centroids.npy",
        "resources/clause_thresholds.npy",
        "resources/clause_applicability.npy",
        "resources/clause_polarity.npy",
    ]

    if all(os.path.exists(p) for p in required_paths):
        print("✅ Models already present. Skipping download.")
        return

    print("⬇️ Downloading models from Google Drive...")

    os.makedirs(MODEL_DIR, exist_ok=True)

    gdown.download_folder(
        id=GDRIVE_FOLDER_ID,
        output=MODEL_DIR,
        quiet=False,
        use_cookies=False
    )

    print("✅ Model download complete.")