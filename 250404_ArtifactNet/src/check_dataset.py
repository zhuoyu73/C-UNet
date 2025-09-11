import sys
import os
from pathlib import Path
import traceback



# 添加项目根路径到 sys.path
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root / "src"))

from datasets import ArtifactImageSliceDataset

# 路径和文件
root_dir = "/mnt/external/zhuoyu/fully+osci"
data_dir = project_root / "data/v0"
train_dataset = ArtifactImageSliceDataset(root_dir, data_dir / "training.txt")

print(f"✅ Total training samples: {len(train_dataset)}")
'''
# 开始检查
for i in range(len(train_dataset)):
    try:
        _ = train_dataset[i]
    except Exception as e:
        print(f"❌ Error at sample {i}: {e}")
        traceback.print_exc()
        break
'''

start_idx = 56800
end_idx = 56864
step = 1

print(f"Checking samples {start_idx} to {end_idx}...")

for i in range(start_idx, end_idx, step):
    try:
        _ = train_dataset[i]
    except Exception as e:
        print(f"\n❌ Error at sample {i}: {e}")
        traceback.print_exc()
        break

print("✅ Done checking selected range.")

