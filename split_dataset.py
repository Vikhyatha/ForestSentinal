import os
import random
import shutil

SRC = r"C:\Users\vigna\OneDrive\Desktop\MiniProject-2\dataset_3class"
OUT = r"C:\Users\vigna\OneDrive\Desktop\MiniProject-2\dataset_split"
TEST_RATIO = 0.2
SEED = 42

random.seed(SEED)

classes = ["bird", "animal", "chainsaw"]

for split in ["train", "test"]:
    for cls in classes:
        os.makedirs(os.path.join(OUT, split, cls), exist_ok=True)

for cls in classes:
    cls_dir = os.path.join(SRC, cls)
    files = [f for f in os.listdir(cls_dir) if f.lower().endswith(".wav")]
    random.shuffle(files)

    test_count = int(len(files) * TEST_RATIO)
    test_files = files[:test_count]
    train_files = files[test_count:]

    for f in train_files:
        shutil.copy2(os.path.join(cls_dir, f), os.path.join(OUT, "train", cls, f))

    for f in test_files:
        shutil.copy2(os.path.join(cls_dir, f), os.path.join(OUT, "test", cls, f))

    print(f"{cls} -> train: {len(train_files)}, test: {len(test_files)}")

print("Done: dataset_split created")