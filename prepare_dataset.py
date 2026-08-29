from pathlib import Path
import shutil


# ============================================================
# 1. PATHS
# ============================================================

# Change this to the location of your original DataSet folder
SOURCE_DIR = Path(r"C:/Users/anujk/Downloads/archive/DataSet")

# Our project dataset directory
PROJECT_DIR = Path("dataset")


# ============================================================
# 2. DATASET CONFIGURATION
# ============================================================

# Original folder name -> our class name
classes = {
    "One": "1_rupee",
    "Two": "2_rupee",
    "Five": "5_rupee",
    "Ten": "10_rupee"
}


# Which physical coins go into train/test?
split = {
    "One": {
        "train": [1, 2, 3, 4, 5, 6],
        "test": [7]
    },

    "Two": {
        "train": [1, 2, 3],
        "test": [4]
    },

    "Five": {
        "train": [1, 2, 3, 4, 5, 6, 7],
        "test": [8, 9]
    },

    "Ten": {
        "train": [1, 2, 3, 4, 5],
        "test": [6]
    }
}


# ============================================================
# 3. CREATE DESTINATION FOLDERS
# ============================================================

for class_name in classes.values():
    (PROJECT_DIR / "train" / class_name).mkdir(
        parents=True,
        exist_ok=True
    )

    (PROJECT_DIR / "test" / class_name).mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# 4. COPY IMAGES
# ============================================================

image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}


for original_class, class_name in classes.items():

    class_source = SOURCE_DIR / original_class

    print(f"\nProcessing {original_class}...")

    for coin_number in split[original_class]["train"]:

        coin_source = class_source / f"Coin {coin_number}"

        # Go through Artificial Light, Low Light, Natural Light
        for light_folder in coin_source.iterdir():

            if not light_folder.is_dir():
                continue

            for image_path in light_folder.iterdir():

                if image_path.suffix.lower() not in image_extensions:
                    continue

                # New unique filename
                new_name = (
                    f"{class_name}_coin{coin_number}_"
                    f"{light_folder.name.replace(' ', '_')}_"
                    f"{image_path.name}"
                )

                destination = (
                    PROJECT_DIR
                    / "train"
                    / class_name
                    / new_name
                )

                shutil.copy2(image_path, destination)

    # --------------------------------------------------------
    # TEST DATA
    # --------------------------------------------------------

    for coin_number in split[original_class]["test"]:

        coin_source = class_source / f"Coin {coin_number}"

        for light_folder in coin_source.iterdir():

            if not light_folder.is_dir():
                continue

            for image_path in light_folder.iterdir():

                if image_path.suffix.lower() not in image_extensions:
                    continue

                new_name = (
                    f"{class_name}_coin{coin_number}_"
                    f"{light_folder.name.replace(' ', '_')}_"
                    f"{image_path.name}"
                )

                destination = (
                    PROJECT_DIR
                    / "test"
                    / class_name
                    / new_name
                )

                shutil.copy2(image_path, destination)


# ============================================================
# 5. COUNT IMAGES
# ============================================================

print("\n" + "=" * 50)
print("DATASET PREPARATION COMPLETE")
print("=" * 50)

for class_name in classes.values():

    train_dir = PROJECT_DIR / "train" / class_name
    test_dir = PROJECT_DIR / "test" / class_name

    train_count = sum(
        1 for file in train_dir.iterdir()
        if file.is_file()
    )

    test_count = sum(
        1 for file in test_dir.iterdir()
        if file.is_file()
    )

    print(
        f"{class_name:10} → "
        f"Train: {train_count:3} | "
        f"Test: {test_count:3}"
    )