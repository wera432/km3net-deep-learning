import h5py

with h5py.File("aeCC_cleaned.h5", "r") as f:

    def show(name, obj):
        if isinstance(obj, h5py.Dataset):
            print("aeCC_cleaned.h5")
            print(f"{name}")
            print(f"  shape : {obj.shape}")
            print(f"  dtype : {obj.dtype}")
            print()

    f.visititems(show)

with h5py.File("amCC_cleaned.h5", "r") as f:

    def show(name, obj):
        if isinstance(obj, h5py.Dataset):
            print("amCC_cleaned.h5")
            print(f"{name}")
            print(f"  shape : {obj.shape}")
            print(f"  dtype : {obj.dtype}")
            print()

    f.visititems(show)

with h5py.File("atCC_cleaned.h5", "r") as f:

    def show(name, obj):
        if isinstance(obj, h5py.Dataset):
            print("atCC_cleaned.h5")
            print(f"{name}")
            print(f"  shape : {obj.shape}")
            print(f"  dtype : {obj.dtype}")
            print()

    f.visititems(show)

with h5py.File("eCC_cleaned.h5", "r") as f:

    def show(name, obj):
        if isinstance(obj, h5py.Dataset):
            print("eCC_cleaned.h5")
            print(f"{name}")
            print(f"  shape : {obj.shape}")
            print(f"  dtype : {obj.dtype}")
            print()

    f.visititems(show)

with h5py.File("mCC_cleaned.h5", "r") as f:

    def show(name, obj):
        if isinstance(obj, h5py.Dataset):
            print("mCC_cleaned.h5")
            print(f"{name}")
            print(f"  shape : {obj.shape}")
            print(f"  dtype : {obj.dtype}")
            print()

    f.visititems(show)

with h5py.File("mNC_cleaned.h5", "r") as f:

    def show(name, obj):
        if isinstance(obj, h5py.Dataset):
            print("mNC_cleaned.h5")
            print(f"{name}")
            print(f"  shape : {obj.shape}")
            print(f"  dtype : {obj.dtype}")
            print()

    f.visititems(show)

with h5py.File("tCC_cleaned.h5", "r") as f:

    def show(name, obj):
        if isinstance(obj, h5py.Dataset):
            print("tCC_cleaned.h5")
            print(f"{name}")
            print(f"  shape : {obj.shape}")
            print(f"  dtype : {obj.dtype}")
            print()

    f.visititems(show)

with h5py.File("amNC_cleaned.h5", "r") as f:

    def show(name, obj):
        if isinstance(obj, h5py.Dataset):
            print("amNC_cleaned.h5")
            print(f"{name}")
            print(f"  shape : {obj.shape}")
            print(f"  dtype : {obj.dtype}")
            print()

    f.visititems(show)

import pandas as pd

df = pd.read_hdf("aeCC_cleaned.h5")

print(df.columns)

print(df.info())

print(df.describe())