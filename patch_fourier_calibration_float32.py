from pathlib import Path

src = Path("run_fourier_diffusion_calibration.py")
dst = Path("run_fourier_diffusion_calibration_krobust.py")

text = src.read_text()

old = """    A = sparse.load_npz(args.graph_npz).tocsr()
    if A.shape != (n, n):
"""

new = """    A = sparse.load_npz(args.graph_npz).tocsr()

    if A.dtype != np.float32:
        print(
            f"Casting graph from {A.dtype} to float32 for MKL...",
            flush=True,
        )
        A = A.astype(np.float32, copy=False)

    if A.shape != (n, n):
"""

if old not in text:
    raise RuntimeError(
        "Target code block not found. "
        "The source file may already be patched or has changed."
    )

patched = text.replace(old, new, 1)

dst.write_text(patched)

print("SUCCESS")
print("Source preserved :", src)
print("Patched copy     :", dst)
