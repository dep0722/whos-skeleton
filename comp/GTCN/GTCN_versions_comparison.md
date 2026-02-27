# GTCN 改進版本對照表

## 📊 三個版本的改進重點

### **版本 1：移除邊特徵 (No Edge Features)**
**檔案**: `gtcn_train_v1_no_edge_features.py`

**改進內容**:
- ❌ 移除膝關節角度計算
- ✅ 簡化模型架構，使用 SimpleGCNLayer
- ✅ 減少計算開銷和潛在噪音

**假設驗證**:
- 邊特徵（膝關節角度）可能引入噪音
- 原始座標序列已包含足夠信息

**預期效果**:
- 訓練更穩定（避免 arccos 的數值問題）
- 可能略微提升準確率（如果邊特徵確實是噪音來源）

---

### **版本 2：加入標準化 (With Normalization)** ⭐ **最關鍵**
**檔案**: `gtcn_train_v2_with_normalization.py`

**改進內容**:
- ✅ 對每個關節的 x, y 座標進行 Z-score 標準化
- ✅ 對邊特徵（膝關節角度）也進行標準化
- ✅ 使用 scipy.interpolate.interp1d 替代 np.interp

**關鍵代碼**:
```python
# 標準化節點特徵
for v in range(X.shape[0]):
    for c in range(X.shape[1]):
        seq = X[v, c, :]
        mean = seq.mean()
        std = seq.std()
        X[v, c, :] = (seq - mean) / (std + 1e-8)

# 標準化邊特徵
edge_attr = (edge_attr - edge_attr.mean(axis=0, keepdims=True)) / \
            (edge_attr.std(axis=0, keepdims=True) + 1e-8)
```

**假設驗證**:
- **原始 GTCN 最大缺陷**: 沒有標準化導致特徵範圍差異大
- 標準化是深度學習的基本操作，缺失會嚴重影響性能

**預期效果**:
- **大幅提升準確率**（預計 +10-20%）
- 訓練收斂更快
- 梯度更穩定

---

### **版本 3：更大 Batch Size (Large Batch)**
**檔案**: `gtcn_train_v3_large_batch.py`

**改進內容**:
- ✅ BATCH_SIZE 從 4 增加到 16（4倍）
- ✅ 包含版本2的所有標準化改進
- ✅ 提升 BatchNorm 的統計估計品質

**假設驗證**:
- Batch size=4 太小，BatchNorm 效果差
- GNN 對 batch size 更敏感（因為有 BatchNorm）

**預期效果**:
- 訓練更穩定（BatchNorm 估計更準確）
- 可能進一步提升 1-3% 準確率
- 每個 epoch 訓練時間略長，但總體更快收斂

---

## 🔬 實驗建議

### **執行順序**:
1. 先跑 **版本2**（標準化版本）→ 預期最大改進
2. 再跑 **版本1**（無邊特徵）→ 驗證邊特徵是否有用
3. 最後跑 **版本3**（大batch）→ 在版本2基礎上進一步優化

### **結果對比指標**:
| 版本 | 改進內容 | 預期準確率 | 訓練穩定性 |
|------|----------|------------|-----------|
| 原始 GTCN | 無標準化, batch=4 | ~60-70% | 差 |
| V1: 無邊特徵 | 移除角度計算 | ~65-75% | 中 |
| V2: 標準化 | Z-score norm | ~75-85% ⭐ | 好 |
| V3: 大batch | V2 + batch=16 | ~78-88% | 很好 |
| LSTM (參考) | 標準化, batch=8 | ~85-95% | 很好 |

---

## 💡 關鍵洞察

### **為什麼標準化如此重要？**

**原始 GTCN 的問題**:
```python
# 沒有標準化，直接輸入到模型
X = np.stack(node_list, axis=0).astype(np.float32)  # 範圍可能是 [0, 1920] 或 [-500, 500]
```

**後果**:
- 不同關節點的數值範圍差異 10-100 倍
- 梯度消失/爆炸
- BatchNorm 無法有效工作
- 優化器難以找到最優解

**修正後**:
```python
# 每個特徵都標準化到 mean=0, std=1
X = (X - X.mean()) / (X.std() + 1e-8)
```

---

## 📝 測試腳本使用

這三個版本的訓練腳本都可以直接運行：

```bash
# 版本1：無邊特徵
python gtcn_train_v1_no_edge_features.py

# 版本2：標準化 (推薦優先測試)
python gtcn_train_v2_with_normalization.py

# 版本3：大batch size
python gtcn_train_v3_large_batch.py
```

**注意事項**:
- 確保資料路徑正確: `D:\Coding\CSV\0211_train`
- 模型會分別保存為:
  - `gtcn_v1_no_edge_features.pth`
  - `gtcn_v2_with_normalization.pth`
  - `gtcn_v3_large_batch.pth`

---

## 🎯 預期結論

如果實驗結果符合預期：

1. **版本2 (標準化) 應該比原始 GTCN 好得多** → 證實標準化的重要性
2. **版本1 (無邊特徵) 可能略好於原始** → 說明邊特徵引入噪音
3. **版本3 (大batch) 應該是最好的** → 標準化 + 大batch 的組合效果

**如果版本2/3 仍不如 LSTM**：
- 說明 GTCN 的架構設計（先空間後時序）本質上不適合步態分析
- 建議嘗試 ST-GCN 或直接使用 LSTM/Transformer
