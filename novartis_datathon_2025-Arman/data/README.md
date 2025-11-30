# Data Directory

**This directory is for Google Drive storage only — not tracked in Git.**

All datasets, raw files, and processed data are stored in **Google Drive** at:
```
MyDrive/ml-colab-agentic/data/
```

---

## Structure (Created in Drive)

```
data/
├── raw/                    # Downloaded datasets (CIFAR-10, ImageNet, etc.)
│   └── cifar-10-batches-py/    # Example: CIFAR-10 dataset (~170 MB)
└── processed/              # Preprocessed or augmented datasets
    └── ...                 # Your custom processed data
```

---

## How It Works

### In Colab Notebook

**Section A1** sets up paths:
```python
DATA_DIR = "/content/drive/MyDrive/ml-colab-agentic/data"
```

**Section C1** downloads datasets to Drive:
```python
train_dl, test_dl = prepare_dataloaders_cifar10(
    root=f"{DATA_DIR}/raw",  # Downloads CIFAR-10 here
    batch_size=128,
    num_workers=2,
)
```

### First Run
- CIFAR-10 automatically downloads to `data/raw/` (~160 MB)
- Takes 1-2 minutes depending on connection

### Subsequent Runs
- Dataset is cached in Drive
- Loads in ~10 seconds (no re-download)

---

## Adding Custom Datasets

### Option 1: Upload to Drive

1. Upload your dataset to `MyDrive/ml-colab-agentic/data/raw/my_dataset/`
2. In notebook, point to it:
   ```python
   CFG["data_root"] = f"{DATA_DIR}/raw/my_dataset"
   ```

### Option 2: Download Programmatically

Add a loader in `src/utils.py`:
```python
def prepare_dataloaders_custom(root, batch_size, num_workers):
    # Download or load from Drive
    dataset_path = os.path.join(root, "custom_dataset")
    if not os.path.exists(dataset_path):
        # Download logic here
        download_dataset(dataset_path)
    
    # Load data
    train_dataset = CustomDataset(dataset_path, split="train")
    test_dataset = CustomDataset(dataset_path, split="test")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader
```

---

## Cache Paths (Set in Notebook)

**Section A4b** also sets environment variables for model/tokenizer caching:

```python
os.environ["TORCH_HOME"] = f"{DATA_DIR}/cache/torch"      # PyTorch models
os.environ["HF_HOME"] = f"{DATA_DIR}/cache/hf"            # Hugging Face models
```

This ensures pretrained models (e.g., ResNet, BERT) are cached in Drive instead of re-downloading.

---

## Important Notes

- **Never commit large datasets to Git** — they're excluded via `.gitignore`
- **All data lives in Google Drive** — persists across Colab sessions
- **Colab VM is ephemeral** — storing data there is lost after 12h idle

---

## Example: CIFAR-10

| Property | Value |
|----------|-------|
| **Size** | ~170 MB (raw), ~160 MB (downloaded) |
| **Location** | `data/raw/cifar-10-batches-py/` |
| **Format** | Pickled Python objects (batches) |
| **Classes** | 10 (airplane, car, bird, cat, deer, dog, frog, horse, ship, truck) |
| **Train** | 50,000 images (32×32 RGB) |
| **Test** | 10,000 images (32×32 RGB) |

---

## Disk Usage in Drive

After first run:

```
MyDrive/ml-colab-agentic/
├── data/                       ~170 MB
│   ├── raw/
│   │   └── cifar-10-batches-py/  ~170 MB
│   ├── processed/              (empty)
│   └── cache/                  ~500 MB (if using pretrained models)
├── runs/
│   └── 2025-10-31.../          ~2 MB per run
│       └── checkpoints/        ~1.5 MB (SimpleNet weights)
└── latest/                     (symlink, ~0 MB)
```

**Total:** ~200 MB (CIFAR-10) + ~2 MB per training run

---

## See Also

- [../docs/QUICK_START.md](../docs/QUICK_START.md) — Full setup guide
- [../notebooks/01_train.ipynb](../notebooks/01_train.ipynb) — Training notebook
- [../src/utils.py](../src/utils.py) — Dataset loaders
