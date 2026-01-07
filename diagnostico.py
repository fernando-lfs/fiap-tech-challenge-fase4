print("1. Testando import sys/os...")
import sys
import os

print("   OK.")

print("2. Testando import pandas...")
import pandas

print("   OK.")

print("3. Testando import yfinance...")
import yfinance

print("   OK.")

print("4. Testando import torch (Suspeito principal)...")
import torch

print("   OK.")

print("5. Testando import src.config...")
sys.path.append(os.getcwd())
from src import config

print("   OK.")
