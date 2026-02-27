"""
改進的片段過濾策略
使用 Hip_x 的變化率判斷是否為轉彎
"""

import pandas as pd
import numpy as np

def filter_segment_by_hip_stability(df_segment, df_original, start_frame, end_frame, 
                                     max_hip_movement=255,
                                     max_hip_change_rate=20):
    """
    根據 Hip_x 的穩定性過濾片段
    
    參數:
        df_segment: 片段資料
        df_original: 原始完整資料
        start_frame, end_frame: 片段在原始資料中的位置
        max_hip_movement: 整個片段中 Hip 最大移動距離
        max_hip_change_rate: 相鄰幀間 Hip 最大變化率
    
    返回:
        is_valid: 是否為有效片段 (True=直線行走, False=轉彎)
        reason: 過濾原因
    """
    
    segment_data = df_original.iloc[start_frame:end_frame]
    
    # 檢查是否有 MidHip_x
    if 'MidHip_x' not in segment_data.columns:
        return True, "No MidHip data"
    
    hip_x = segment_data['MidHip_x'].values
    
    # 策略1: 檢查整體移動範圍
    hip_range = hip_x.max() - hip_x.min()
    if hip_range > max_hip_movement:
        return False, f"Hip movement too large: {hip_range:.1f} > {max_hip_movement}"
    
    # 策略2: 檢查變化率(相鄰幀的差異)
    hip_changes = np.abs(np.diff(hip_x))
    max_change = hip_changes.max()
    if max_change > max_hip_change_rate:
        return False, f"Hip change rate too high: {max_change:.1f} > {max_hip_change_rate}"
    
    # 策略3: 檢查是否在合理範圍內(中央區域)
    hip_mean = hip_x.mean()
    if hip_mean < 700 or hip_mean > 1300:
        return False, f"Hip center out of range: {hip_mean:.1f}"
    
    # 策略4: 檢查標準差(穩定性)
    hip_std = hip_x.std()
    if hip_std > 30:
        return False, f"Hip too unstable: std={hip_std:.1f}"
    
    return True, "Valid"


def analyze_segments_quality(csv_path, segments_dir):
    """
    分析所有片段的品質
    幫助你調整過濾參數
    """
    import glob
    from pathlib import Path
    
    df_original = pd.read_csv(csv_path)
    segment_files = glob.glob(f"{segments_dir}/*.csv")
    
    results = []
    
    for seg_file in segment_files:
        df_seg = pd.read_csv(seg_file)
        seg_name = Path(seg_file).stem
        
        # 假設有 original_frame 或使用特徵匹配
        if 'original_frame' in df_seg.columns:
            start = int(df_seg['original_frame'].iloc[0])
            end = int(df_seg['original_frame'].iloc[-1]) + 1
        else:
            # 簡單匹配
            start = 0
            end = len(df_seg)
        
        segment_data = df_original.iloc[start:end]
        
        if 'MidHip_x' in segment_data.columns:
            hip_x = segment_data['MidHip_x'].values
            
            info = {
                'name': seg_name,
                'length': len(hip_x),
                'hip_mean': hip_x.mean(),
                'hip_std': hip_x.std(),
                'hip_range': hip_x.max() - hip_x.min(),
                'hip_max_change': np.abs(np.diff(hip_x)).max()
            }
            
            # 測試過濾
            is_valid, reason = filter_segment_by_hip_stability(
                df_seg, df_original, start, end
            )
            info['valid'] = is_valid
            info['reason'] = reason
            
            results.append(info)
    
    # 轉成 DataFrame 方便分析
    df_results = pd.DataFrame(results)
    
    print("="*80)
    print("片段品質分析")
    print("="*80)
    print(f"\n總片段數: {len(df_results)}")
    print(f"有效片段: {df_results['valid'].sum()}")
    print(f"無效片段: {(~df_results['valid']).sum()}")
    
    print("\n統計資訊:")
    print(df_results[['hip_mean', 'hip_std', 'hip_range', 'hip_max_change']].describe())
    
    print("\n無效片段:")
    invalid = df_results[~df_results['valid']]
    for _, row in invalid.iterrows():
        print(f"  {row['name']:20s} | {row['reason']}")
    
    return df_results


# 使用範例
if __name__ == "__main__":
    # ===== 設定路徑 =====
    csv_path = r"C:\mydata\sf\open\output_csv\0128\A\0128_A_3.csv"
    segments_dir = r"C:\mydata\sf\open\output_csv\0128cut\A\A_3"
    output_dir = r"C:\mydata\sf\comp\0214"  # 輸出資料夾
    # ===================
    
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # 分析片段品質,找出最佳過濾參數
    results = analyze_segments_quality(csv_path, segments_dir)
    
    # 儲存分析結果
    csv_output = os.path.join(output_dir, "segment_quality_analysis.csv")
    results.to_csv(csv_output, index=False)
    print(f"\n✓ 分析結果已儲存到 {csv_output}")
    
    # 繪製分布圖
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    results[results['valid']]['hip_mean'].hist(ax=axes[0,0], bins=20, alpha=0.7, label='Valid')
    results[~results['valid']]['hip_mean'].hist(ax=axes[0,0], bins=20, alpha=0.7, label='Invalid')
    axes[0,0].set_xlabel('Hip Mean X')
    axes[0,0].set_title('Hip Center Position')
    axes[0,0].legend()
    
    results[results['valid']]['hip_std'].hist(ax=axes[0,1], bins=20, alpha=0.7, label='Valid')
    results[~results['valid']]['hip_std'].hist(ax=axes[0,1], bins=20, alpha=0.7, label='Invalid')
    axes[0,1].set_xlabel('Hip Std')
    axes[0,1].set_title('Hip Stability')
    axes[0,1].legend()
    
    results[results['valid']]['hip_range'].hist(ax=axes[1,0], bins=20, alpha=0.7, label='Valid')
    results[~results['valid']]['hip_range'].hist(ax=axes[1,0], bins=20, alpha=0.7, label='Invalid')
    axes[1,0].set_xlabel('Hip Range')
    axes[1,0].set_title('Hip Movement Range')
    axes[1,0].legend()
    
    results[results['valid']]['hip_max_change'].hist(ax=axes[1,1], bins=20, alpha=0.7, label='Valid')
    results[~results['valid']]['hip_max_change'].hist(ax=axes[1,1], bins=20, alpha=0.7, label='Invalid')
    axes[1,1].set_xlabel('Hip Max Change')
    axes[1,1].set_title('Hip Change Rate')
    axes[1,1].legend()
    
    plt.tight_layout()
    
    fig_output = os.path.join(output_dir, 'segment_quality_distribution.png')
    plt.savefig(fig_output, dpi=150)
    print(f"✓ 分布圖已儲存到 {fig_output}")