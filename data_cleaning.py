import csv 
import pandas as pd

df = pd.read_csv("winnie2_log_20261116_150512.csv")
print("read csv")
df = df[df["hand_index"] == 0]
print("invalid hand index removed")
df = df[df["handedness"] == "Right"]
print("invalid hand removed")


df.to_csv("winnie2_filtered.csv", index=False)
print('written to new csv')

