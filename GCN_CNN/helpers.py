from sklearn.metrics import classification_report
from typing import List, Tuple, Iterable
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from collections import Counter, defaultdict
import os
import numpy as np

def extract_design(path):
    #../data/asap7/<DESIGN>/data_point_x
    parts = os.path.normpath(path).split(os.sep)
    for i, p in enumerate(parts):
        if p.startswith("data_point_") and i > 0:
            return parts[i - 1]
    return parts[-2]

def design_of(path: str) -> str:
    return extract_design(path)

def summarize_split(train_folders: List[str], test_folders: List[str], val_folders: List[str]) -> None:
    def group_by_design(folders):
        grouped = defaultdict(list)
        for p in folders:
            design = design_of(p)
            data_point = os.path.basename(p)
            grouped[design].append(data_point)
        return {d: sorted(points) for d, points in grouped.items()}

    train_designs = group_by_design(train_folders)
    test_designs  = group_by_design(test_folders)
    val_designs = group_by_design(val_folders)

    
    print("Train designs:", sorted(train_designs.keys()))
    print("Test designs :", sorted(test_designs.keys()))
    print("Val deisgns :", sorted(val_designs.keys()))
    print("Counts----TRAIN:", len(train_folders), "TEST:", len(test_folders), "VAL:", len(val_folders))

    print("\n--- TRAIN details ---")
    for design, points in sorted(train_designs.items()):
        for dp in points:
            print(f"{design}/{dp}")

    print("\n--- TEST details ---")
    for design, points in sorted(test_designs.items()):
        for dp in points:
            print(f"{design}/{dp}")
    
    print("\n--- VAL details ---")
    for design, points in sorted(val_designs.items()):
        for dp in points:
            print(f"{design}/{dp}")

def split_explicit_by_design(
    all_folders: List[str],
    train_designs: Iterable[str],
    test_designs: Iterable[str],
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[List[str], List[str]]:
    train_designs = set(train_designs)
    test_designs  = set(test_designs)
    assert train_designs.isdisjoint(test_designs), "Train/Test design sets must not overlap."

    train_folders = [p for p in all_folders if design_of(p) in train_designs]
    test_folders  = [p for p in all_folders if design_of(p) in test_designs]

    if not train_folders or not test_folders:
        raise RuntimeError("One split is empty. Check your design names or data.")
    if test_size < 1.0:
        rng = np.random.RandomState(random_state)
        idx = np.arange(len(test_folders))
        rng.shuffle(idx)
        k = max(1, int(round(len(test_folders) * test_size)))
        test_folders = [test_folders[i] for i in idx[:k]]
    return train_folders, test_folders



def split_datapoint_random(
    all_folders: List[str],
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[List[str], List[str]]:
    train, test = train_test_split(
        all_folders,
        test_size=test_size,
        random_state=random_state,
        shuffle=True
    )
    return train, test

def split_within_design(all_folders: List[str], design: str, test_size: float = 0.2, random_state: int = 42):
    design_folder = [p for p in all_folders if design_of(p) == design]
    if len(design_folder) < 2:
        raise RuntimeError(f"Need at least 2 data_point folders for {design}, found {len(design_folder)}")
    train_folders, test_folders = train_test_split(
        design_folder, test_size=test_size, random_state=random_state, shuffle=True
    )
    return train_folders, test_folders


