import sys

pkgs = [
    'torch',
    'transformers',
    'datasets',
    'scipy',
    'sklearn',
    'matplotlib',
    'pandas',
    'numpy',
    'accelerate',
    'huggingface_hub',
    'tqdm'
]

print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")

missing = []
for p in pkgs:
    try:
        mod = __import__(p)
        ver = getattr(mod, '__version__', 'available')
        print(f"  [+] {p:15s}: {ver}")
    except Exception as e:
        print(f"  [-] {p:15s}: NOT INSTALLED ({e})")
        missing.append(p)

print("\nMissing packages:", missing)
