import re
import matplotlib.pyplot as plt
import os

# ===== 設定輸出路徑 =====
save_dir = "C:\mydata\sf\open\output_images\model_result"        # 資料夾路徑（可自行改）
file_name = "gtcn_6node_5min_0213.png" # 檔名
save_path = os.path.join(save_dir, file_name)

# 如果資料夾不存在就建立
os.makedirs(save_dir, exist_ok=True)

# ===== 訓練紀錄貼這裡 =====
log_text = """
Epoch   1 | Loss: 2.7546 | Acc: 0.1413
Epoch   2 | Loss: 2.0880 | Acc: 0.3480
Epoch   3 | Loss: 1.8118 | Acc: 0.3943
Epoch   4 | Loss: 1.6240 | Acc: 0.4857
Epoch   5 | Loss: 1.4915 | Acc: 0.5131
Epoch   6 | Loss: 1.3740 | Acc: 0.5309
Epoch   7 | Loss: 1.3344 | Acc: 0.5570
Epoch   8 | Loss: 1.2109 | Acc: 0.5772
Epoch   9 | Loss: 1.1076 | Acc: 0.6413
Epoch  10 | Loss: 1.0320 | Acc: 0.6793
Epoch  11 | Loss: 1.0159 | Acc: 0.6686
Epoch  12 | Loss: 0.9406 | Acc: 0.6781
Epoch  13 | Loss: 0.8584 | Acc: 0.7114
Epoch  14 | Loss: 0.8704 | Acc: 0.7447
Epoch  15 | Loss: 0.7544 | Acc: 0.7648
Epoch  16 | Loss: 0.6880 | Acc: 0.7922
Epoch  17 | Loss: 0.6776 | Acc: 0.7957
Epoch  18 | Loss: 0.6978 | Acc: 0.7862
Epoch  19 | Loss: 0.6179 | Acc: 0.8017
Epoch  20 | Loss: 0.5810 | Acc: 0.8266
Epoch  21 | Loss: 0.5325 | Acc: 0.8314
Epoch  22 | Loss: 0.5373 | Acc: 0.8397
Epoch  23 | Loss: 0.5365 | Acc: 0.8385
Epoch  24 | Loss: 0.4902 | Acc: 0.8587
Epoch  25 | Loss: 0.4177 | Acc: 0.8777
Epoch  26 | Loss: 0.3944 | Acc: 0.8824
Epoch  27 | Loss: 0.3945 | Acc: 0.8895
Epoch  28 | Loss: 0.4023 | Acc: 0.8753
Epoch  29 | Loss: 0.3631 | Acc: 0.9014
Epoch  30 | Loss: 0.3789 | Acc: 0.8907
Epoch  31 | Loss: 0.3310 | Acc: 0.9038
Epoch  32 | Loss: 0.3273 | Acc: 0.9002
Epoch  33 | Loss: 0.2860 | Acc: 0.9097
Epoch  34 | Loss: 0.3084 | Acc: 0.8931
Epoch  35 | Loss: 0.3211 | Acc: 0.8990
Epoch  36 | Loss: 0.2927 | Acc: 0.9181
Epoch  37 | Loss: 0.2476 | Acc: 0.9323
Epoch  38 | Loss: 0.2412 | Acc: 0.9311
Epoch  39 | Loss: 0.2489 | Acc: 0.9323
Epoch  40 | Loss: 0.2907 | Acc: 0.9086
Epoch  41 | Loss: 0.2828 | Acc: 0.9157
Epoch  42 | Loss: 0.2528 | Acc: 0.9252
Epoch  43 | Loss: 0.1871 | Acc: 0.9442
Epoch  44 | Loss: 0.2053 | Acc: 0.9406
Epoch  45 | Loss: 0.2320 | Acc: 0.9323
Epoch  46 | Loss: 0.2649 | Acc: 0.9252
Epoch  47 | Loss: 0.2182 | Acc: 0.9359
Epoch  48 | Loss: 0.2478 | Acc: 0.9157
Epoch  49 | Loss: 0.2137 | Acc: 0.9287
Epoch  50 | Loss: 0.1801 | Acc: 0.9454
"""

# ===== 解析資料 =====
pattern = r"Epoch\s+(\d+)\s+\|\s+Loss:\s+([\d.]+)\s+\|\s+Acc:\s+([\d.]+)"

epochs, losses, accs = [], [], []

for e, l, a in re.findall(pattern, log_text):
    epochs.append(int(e))
    losses.append(float(l))
    accs.append(float(a))

# ===== 繪圖 =====
plt.figure(figsize=(8, 5))

plt.plot(epochs, losses, label="Loss")
plt.plot(epochs, accs, label="Accuracy")

plt.xlabel("Epoch")
plt.ylabel("Value")
plt.title("Training Loss & Accuracy")

plt.legend()
plt.grid(True)

# ===== 存檔（關鍵）=====
plt.savefig(save_path, dpi=300, bbox_inches="tight")

# 若不想顯示視窗，可註解下一行
# plt.show()

plt.close()

print(f"圖片已儲存至：{save_path}")