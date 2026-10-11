import os
import requests
import hashlib
from tqdm import tqdm

DATA_URL = "https://huggingface.co/datasets/Aiwensile2/Minecraft_QA-pairs_Instruction_Dataset/resolve/main/minecraft_instruction_dataset.json"
TARGET_PATH = os.path.join("data", "raw", "minecraft_instruction_dataset.json")

def download_file():
    if os.path.exists(TARGET_PATH):
        print(f"File already exists at {TARGET_PATH}, size: {os.path.getsize(TARGET_PATH)} bytes")
        return

    print(f"Downloading from {DATA_URL}...")
    headers = {"User-Agent": "MineMentor-Audit/1.0"}
    response = requests.get(DATA_URL, headers=headers, stream=True, timeout=60)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024 * 1024  # 1MB
    sha256 = hashlib.sha256()

    with open(TARGET_PATH, "wb") as f, tqdm(
        total=total_size, unit="B", unit_scale=True, desc="Downloading"
    ) as progress_bar:
        for chunk in response.iter_content(chunk_size=block_size):
            if chunk:
                f.write(chunk)
                sha256.update(chunk)
                progress_bar.update(len(chunk))

    print(f"\nDownload completed: {TARGET_PATH}")
    print(f"File size: {os.path.getsize(TARGET_PATH):,} bytes")
    print(f"SHA-256: {sha256.hexdigest()}")

if __name__ == "__main__":
    download_file()
