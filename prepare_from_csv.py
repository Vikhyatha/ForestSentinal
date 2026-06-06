import os
import csv
import shutil

AUDIO_DIR = r"C:\Users\vigna\OneDrive\Desktop\MiniProject-2\audio"
CSV_PATH = r"C:\Users\vigna\OneDrive\Desktop\MiniProject-2\meta\esc50.csv"   # change if needed
OUT_DIR = r"C:\Users\vigna\OneDrive\Desktop\MiniProject-2\dataset_3class"

BIRD_CATS = {"chirping_birds", "crow", "rooster", "hen"}
ANIMAL_CATS = {"dog", "cat", "cow", "frog", "pig", "insects", "sheep"}
CHAINSAW_CATS = {"chainsaw"}

for cls in ["bird", "animal", "chainsaw"]:
    os.makedirs(os.path.join(OUT_DIR, cls), exist_ok=True)

counts = {"bird": 0, "animal": 0, "chainsaw": 0}

with open(CSV_PATH, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        fn = row["filename"]
        cat = row["category"].strip().lower()

        if cat in BIRD_CATS:
            cls = "bird"
        elif cat in ANIMAL_CATS:
            cls = "animal"
        elif cat in CHAINSAW_CATS:
            cls = "chainsaw"
        else:
            continue

        src = os.path.join(AUDIO_DIR, fn)
        if os.path.exists(src):
            dst = os.path.join(OUT_DIR, cls, fn)
            shutil.copy2(src, dst)
            counts[cls] += 1

print("Done")
print("bird:", counts["bird"])
print("animal:", counts["animal"])
print("chainsaw:", counts["chainsaw"])