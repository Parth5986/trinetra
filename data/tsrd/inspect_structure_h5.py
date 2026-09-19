import h5py

file_path = "D:/trinetra/data/tsrd/archive/test/test_0.h5"

with h5py.File(file_path, "r") as f:

    print("\n===== FILE STRUCTURE =====\n")

    def print_structure(name, obj):
        print(name)

    f.visititems(print_structure)

    print("\n===== ROOT KEYS =====\n")
    print(list(f.keys()))