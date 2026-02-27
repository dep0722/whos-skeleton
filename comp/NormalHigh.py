import pandas as pd
import numpy as np
from dtw import dtw
import os
from itertools import combinations_with_replacement

def get_ankle_data(file_path):
    try:
        df = pd.read_csv(file_path)
        file_name = os.path.basename(file_path).upper()
        col_name = 'LAnkle_y' if file_name.startswith('L') else 'RAnkle_y'
        
        if col_name not in df.columns: return None
        series = df[col_name].values.flatten().astype(float)
        
        mask = series > 0
        if not any(mask) or len(series) < 5: return None
            
        mean_val = series[mask].mean()
        series[~mask] = mean_val
        
        std = np.std(series)
        if std != 0:
            series = (series - np.mean(series)) / std
        return series
    except:
        return None

def compare_folders_with_contrast(path_a, path_b):
    files_a = [f for f in os.listdir(path_a) if f.endswith('.csv')]
    files_b = [f for f in os.listdir(path_b) if f.endswith('.csv')]
    
    contrast_scores = []
    for fa in files_a:
        for fb in files_b:
            if path_a == path_b and fa == fb: continue
            if fa[0].upper() != fb[0].upper(): continue
            
            s1 = get_ankle_data(os.path.join(path_a, fa))
            s2 = get_ankle_data(os.path.join(path_b, fb))
            
            if s1 is not None and s2 is not None:
                d = dtw(s1, s2, keep_internals=False).normalizedDistance
                
                # --- 放大差異的公式：使用更高的懲罰係數 ---
                # 原本是 -0.2，現在改成 -1.0。這會讓距離 d=0.5 從 90分 變成 60分
                # 距離 d=1.0 從 81分 變成 36分 (差異被極度放大)
                s_high_contrast = np.exp(-1.0 * d) 
                contrast_scores.append(s_high_contrast)
                
    return contrast_scores

# --- 設定根目錄 ---
root_path = r"C:\Users\haoti\groundhog\Coding\Vscode\Conda\1217_openpose_test\CSV\1230One"
folders = ['A', 'B', 'C', 'E', 'F'] # 也可以用 os.listdir 自動抓

summary_report = []

print(f"開始【高對比度】全資料夾組合分析 (放大差異模式)...")
print("-" * 60)

for f_name_a, f_name_b in combinations_with_replacement(folders, 2):
    path_a = os.path.join(root_path, f_name_a)
    path_b = os.path.join(root_path, f_name_b)
    
    if not os.path.exists(path_a) or not os.path.exists(path_b): continue
    
    scores = compare_folders_with_contrast(path_a, path_b)
    
    if scores:
        avg = np.mean(scores)
        summary_report.append({
            "Pair": f"{f_name_a} <-> {f_name_b}",
            "Avg": avg,
            "Count": len(scores)
        })
        print(f"已計算: {f_name_a} vs {f_name_b} | 放大相似度: {avg:.2%}")

# --- 最終排序輸出 ---
summary_report.sort(key=lambda x: x['Avg'], reverse=True)

print("\n" + "!"*60)
print(f"{'資料夾對組合':<20} | {'比對數':<6} | {'放大後的相似度 (更嚴格)'}")
print("-" * 60)

for item in summary_report:
    # 這裡的分數會比之前的低很多，因為我們放大了「不夠像」的部分
    print(f"{item['Pair']:<20} | {item['Count']:<6} | {item['Avg']:<10.2%}")
print("!"*60)