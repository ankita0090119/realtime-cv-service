from pathlib import Path
import random
import shutil

SOURCE = Path(r"C:\Users\Ankitaa\Downloads\People Detection.v2i.yolov8")
DEST = Path("dataset")

TRAIN_COUNT = 3000
VAL_COUNT = 100
SEED = 42

random.seed(SEED)


def copy_split(source_split, dest_split, count=None):
    image_dir = SOURCE / source_split / "images"
    label_dir = SOURCE / source_split / "labels"

    images = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.jpeg")) + list(image_dir.glob("*.png"))

    random.shuffle(images)

    if count:
        images = images[:count]

    out_images = DEST / "images" / dest_split
    out_labels = DEST / "labels" / dest_split

    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    copied = 0

    for image in images:
        label = label_dir / f"{image.stem}.txt"

        if not label.exists():
            continue

        shutil.copy2(image, out_images / image.name)
        shutil.copy2(label, out_labels / label.name)

        copied += 1

    print(f"{dest_split}: {copied} images")


def main():
    copy_split("train", "train", TRAIN_COUNT)
    copy_split("valid", "val", VAL_COUNT)

    # Keep the complete original test set untouched.
    copy_split("test", "test", 300)

    print("\nDataset preparation complete.")


if __name__ == "__main__":
    main()