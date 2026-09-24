import h5py
import numpy as np
from pathlib import Path

file_path = str(Path(__file__).resolve().parent / "archive" / "test" / "test_0.h5")

# with h5py.File(file_path, "r") as f:
#
#     print("\n===== DATA INFO =====")
#
#     data = f["data"]
#
#     print("Type:", type(data))
#
#     if isinstance(data, h5py.Dataset):
#         print("Shape:", data.shape)
#         print("Dtype:", data.dtype)
#
#         print("\nFirst 10 rows:")
#         print(data[:10])
#
#     print("\n===== LABELS INFO =====")
#
#     labels = f["labels"]
#
#     print("Type:", type(labels))
#
#     if isinstance(labels, h5py.Dataset):
#         print("Shape:", labels.shape)
#         print("Dtype:", labels.dtype)
#
#         print("\nFirst 10 labels:")
#         print(labels[:10])
#
#     print("\n===== COMPLETE METADATA =====")
#
#     def inspect_item(name, obj):
#
#         if isinstance(obj, h5py.Dataset):
#
#             print(f"\nDATASET: {name}")
#             print("Shape:", obj.shape)
#             print("Dtype:", obj.dtype)
#
#             try:
#                 value = obj[()]
#
#                 # Don't print huge datasets completely
#                 if obj.size <= 20:
#                     print("Value:", value)
#                 else:
#                     print("Sample:", obj[:5])
#
#             except Exception as e:
#                 print("Could not read:", e)
#
#         elif isinstance(obj, h5py.Group):
#
#             print(f"\nGROUP: {name}")
#
#     f["metadata"].visititems(inspect_item)

# def print_group(group, indent=0):
#     """Recursively print datasets, groups, and attributes."""
#
#     prefix = " " * indent
#
#     # Print attributes
#     for key, value in group.attrs.items():
#         print(f"{prefix}ATTRIBUTE: {key} = {value}")
#
#     # Print items
#     for key, item in group.items():
#
#         if isinstance(item, h5py.Dataset):
#
#             print(f"{prefix}DATASET: {key}")
#             print(f"{prefix}  Shape: {item.shape}")
#             print(f"{prefix}  Value: {item[()]}")
#
#             for attr_key, attr_value in item.attrs.items():
#                 print(
#                     f"{prefix}  ATTRIBUTE: "
#                     f"{attr_key} = {attr_value}"
#                 )
#
#         elif isinstance(item, h5py.Group):
#
#             print(f"{prefix}GROUP: {key}")
#
#             print_group(item, indent + 4)
#
#
# with h5py.File(file_path, "r") as f:
#
#     print("\n===== RECEIVER METADATA =====\n")
#
#     receiver = f["metadata/receiver"]
#
#     print_group(receiver)

# to inspect attributes of the group

with h5py.File(file_path, "r") as f:

    receiver = f["metadata/receiver"]

    print("\n===== RECEIVER METADATA =====\n")

    for name, group in receiver.items():

        print(f"\n{'=' * 50}")
        print(f"GROUP: {name}")
        print(f"{'=' * 50}")

        # Print attributes
        if len(group.attrs) > 0:
            print("ATTRIBUTES:")

            for key, value in group.attrs.items():
                print(f"{key}: {value}")
        else:
            print("No attributes found.")

        # Print datasets/items inside the group
        if len(group.keys()) > 0:
            print("\nCONTENTS:")

            for key, item in group.items():
                print(f"{key}: {item}")

        else:
            print("No datasets/items inside this group.")