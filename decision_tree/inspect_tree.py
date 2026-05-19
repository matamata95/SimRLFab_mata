from decision_tree.viper import TreePolicy
import numpy as np
import os
import subprocess
import pandas as pd
from sklearn.tree import export_graphviz

TREE_PATH   = "agents/viper_tree_47_states_depth9.joblib"
DOT_PATH    = "agents/viper_tree_47_states_depth9.dot"
OUTPUT_PATH = "agents/viper_tree_47_states_depth9"
FEATURE_MAPPING_PATH = "./log/47 states viper/log_visualization/feature_mapping.csv"

tree = TreePolicy.load(TREE_PATH)
tree.print_info()  # prints depth and n_leaves

# Feature importances
importances = tree.clf.feature_importances_
top = np.argsort(importances)[::-1][:10]
print("\nTop-10 feature importances:")
for rank, i in enumerate(top, 1):
    print(f"  {rank:2d}. feature[{i:3d}] = {importances[i]:.4f}")

fm = pd.read_csv(FEATURE_MAPPING_PATH).sort_values("feature_index")
feature_names = fm["feature_name"].tolist()

# Export full tree to .dot
export_graphviz(
    tree.clf,
    out_file=DOT_PATH,
    feature_names=feature_names,
    filled=True,
    rounded=True,
    impurity=True,
    proportion=False,
    precision=3,
)
print(f"\nDot file saved → {DOT_PATH}")

# Render .dot → PNG and open it
png_path = OUTPUT_PATH + ".png"
result = subprocess.run(
    ["dot", "-Tpng", DOT_PATH, "-o", png_path],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
)
if result.returncode != 0:
    print("dot error:", result.stderr.decode())
else:
    print(f"PNG saved → {png_path}")
