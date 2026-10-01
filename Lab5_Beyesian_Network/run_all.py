"""
Single entry point that reproduces every result in report.pdf.
Run with: python run_all.py
"""
import subprocess, sys

scripts = [
    "dataset.py",
    "first_order_model.py",
    "test_normalisation.py",
    "generate.py",
    "second_order_model.py",
    "compare.py",
]
for s in scripts:
    print(f"\n{'='*70}\n>>> Running {s}\n{'='*70}")
    subprocess.run([sys.executable, s], check=True)
