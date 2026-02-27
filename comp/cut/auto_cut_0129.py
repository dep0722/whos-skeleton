"""
改進版步態週期自動檢測
結合多個特徵來識別週期端點
"""

import pandas as pd
import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

class ImprovedGaitDetector:
    """
    改進的步態週期檢測器
    
    檢測邏輯：
    1. 腳踝速度低（幾乎靜止）
    2. Y 座標穩定（腳在地面）
    3. X 座標在特定範圍（面對鏡頭）
    4. 持續時間足夠（不是瞬間停頓）
    """
    
    def __init__(self, df, side='L', params=None):
        self.df = df
        self.side = side
        self.col_x = f"{side}Ankle_x"
        self.col_y = f"{side}Ankle_y"
        
        # 可調參數（對應舊系統參數）
        self.params = params or {
        'speed_threshold': 0.8,        # 對應 FLAT_SLOPE = 0.1
        'min_static_frames': 3,        # 對應 MIN_FLAT_LEN = 5
        'min_cycle_length': 15,        # 基於 RISE_TOTAL = 3.0 的概念
        'smooth_window': 7,            # 對應 FLAT_WINDOW = 5
        'y_stability_threshold': 1.5   # Y 座標穩定性（新增參數）
    }
    
    def compute_features(self):
        """計算所有特徵"""
        x = self.df[self.col_x].values
        y = self.df[self.col_y].values
        n = len(x)
        
        # 1. 速度特徵
        vx = np.gradient(x)
        vy = np.gradient(y)
        speed = np.sqrt(vx**2 + vy**2)
        speed_smooth = uniform_filter1d(speed, size=self.params['smooth_window'])
        
        # 2. Y 座標變化率（判斷腳是否穩定在地面）
        y_change = uniform_filter1d(np.abs(np.gradient(y)), size=self.params['smooth_window'])
        
        # 3. 加速度（判斷運動狀態改變）
        ax = np.gradient(vx)
        ay = np.gradient(vy)
        
        return {
            'x': x,
            'y': y,
            'vx': vx,
            'vy': vy,
            'speed': speed,
            'speed_smooth': speed_smooth,
            'y_change': y_change,
            'ax': ax,
            'ay': ay
        }
    
    def find_static_regions(self, features):
        """
        找出靜止區域（腳著地且幾乎不動）
        
        條件：
        1. 速度低
        2. Y 座標變化小
        3. 持續時間夠長
        """
        speed = features['speed_smooth']
        y_change = features['y_change']
        
        # 判斷是否靜止
        is_static = (speed < self.params['speed_threshold']) & \
                    (y_change < self.params['y_stability_threshold'])
        
        # 找出連續的靜止區段
        regions = []
        start = None
        
        for i, static in enumerate(is_static):
            if static and start is None:
                start = i
            elif not static and start is not None:
                if i - start >= self.params['min_static_frames']:
                    regions.append({
                        'start': start,
                        'end': i,
                        'center': (start + i) // 2,
                        'duration': i - start,
                        'avg_speed': speed[start:i].mean(),
                        'y_pos': features['y'][start:i].mean(),
                        'x_pos': features['x'][start:i].mean()
                    })
                start = None
        
        # 處理最後一段
        if start is not None:
            i = len(is_static)
            if i - start >= self.params['min_static_frames']:
                regions.append({
                    'start': start,
                    'end': i,
                    'center': (start + i) // 2,
                    'duration': i - start,
                    'avg_speed': speed[start:i].mean(),
                    'y_pos': features['y'][start:i].mean(),
                    'x_pos': features['x'][start:i].mean()
                })
        
        return regions
    
    def extract_cycles(self, regions):
        """
        從靜止區域提取步態週期
        
        每兩個連續的靜止區域之間為一個週期
        """
        if len(regions) < 2:
            return []
        
        cycles = []
        for i in range(len(regions) - 1):
            start_region = regions[i]
            end_region = regions[i + 1]
            
            # 使用靜止區域的中心點作為週期邊界
            start_frame = start_region['center']
            end_frame = end_region['center']
            
            cycle_length = end_frame - start_frame
            
            if cycle_length >= self.params['min_cycle_length']:
                cycles.append({
                    'start': start_frame,
                    'end': end_frame,
                    'side': self.side,
                    'length': cycle_length,
                    'start_region': start_region,
                    'end_region': end_region
                })
        
        return cycles
    
    def detect(self):
        """執行完整檢測流程"""
        features = self.compute_features()
        regions = self.find_static_regions(features)
        cycles = self.extract_cycles(regions)
        return cycles, features, regions
    
    def visualize(self, features, regions, cycles):
        """視覺化檢測結果"""
        fig, axes = plt.subplots(4, 1, figsize=(20, 12), sharex=True)
        
        n = len(features['x'])
        frames = np.arange(n)
        
        # 1. X 座標
        axes[0].plot(frames, features['x'], 'b-', linewidth=1, alpha=0.7, label='X position')
        for region in regions:
            axes[0].axvspan(region['start'], region['end'], color='yellow', alpha=0.3)
        axes[0].set_ylabel('X Position')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 2. Y 座標
        axes[1].plot(frames, features['y'], 'g-', linewidth=1, alpha=0.7, label='Y position')
        for region in regions:
            axes[1].axvspan(region['start'], region['end'], color='yellow', alpha=0.3)
        axes[1].set_ylabel('Y Position')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # 3. 速度
        axes[2].plot(frames, features['speed_smooth'], 'r-', linewidth=1, label='Speed (smoothed)')
        axes[2].axhline(self.params['speed_threshold'], color='orange', linestyle='--', label='Threshold')
        for region in regions:
            axes[2].axvspan(region['start'], region['end'], color='yellow', alpha=0.3)
        axes[2].set_ylabel('Speed')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        # 4. 週期標記
        axes[3].plot(frames, features['y'], 'k-', linewidth=1, alpha=0.5)
        for i, cycle in enumerate(cycles):
            color = plt.cm.tab10(i % 10)
            axes[3].axvspan(cycle['start'], cycle['end'], color=color, alpha=0.3, label=f"Cycle {i+1}")
            axes[3].axvline(cycle['start'], color=color, linestyle='--', linewidth=2)
            axes[3].axvline(cycle['end'], color=color, linestyle='--', linewidth=2)
        
        axes[3].set_ylabel('Detected Cycles')
        axes[3].set_xlabel('Frame Index')
        if len(cycles) <= 10:
            axes[3].legend(loc='upper right', ncol=2)
        axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig

def auto_detect_cycles(csv_path, video_path, output_dir, params=None):
    """
    自動檢測步態週期
    
    返回格式與手動切分相同，可以直接用於編輯器
    """
    df = pd.read_csv(csv_path)
    all_cycles = []
    
    for side in ['L', 'R']:
        print(f"\n檢測 {side} 腳步態週期...")
        
        detector = ImprovedGaitDetector(df, side, params)
        cycles, features, regions = detector.detect()
        
        print(f"  找到 {len(regions)} 個靜止區域")
        print(f"  提取 {len(cycles)} 個週期")
        
        # 顯示檢測結果
        if len(cycles) > 0:
            fig = detector.visualize(features, regions, cycles)
            plt.show()
        
        # 格式化為編輯器所需格式
        for i, cycle in enumerate(cycles, 1):
            all_cycles.append({
                'side': side,
                'start': cycle['start'],
                'end': cycle['end'],
                'name': f"{side}_1_{i}",  # 預設為第1次朝向的第i個週期
                'valid': True
            })
    
    # 按時間排序
    all_cycles.sort(key=lambda c: c['start'])
    
    print(f"\n總共找到 {len(all_cycles)} 個週期")
    
    return all_cycles

# =========================
# 整合到主程式
# =========================
if __name__ == "__main__":
    import sys
    
    # 範例使用
    CSV_PATH = r"C:\mydata\sf\open\output_csv\0128\A\0128_A_1.csv"
    VIDEO_PATH = r"C:\mydata\sf\open\walking_video\0128\0128_A_1.mp4"
    OUTPUT_DIR = r"C:\mydata\sf\open\output_csv\0128cut\A\auto\A_1_auto"
    
    # 可調參數（根據你原本的設定調整）
    params = {
        'speed_threshold': 0.8,        # 對應 FLAT_SLOPE = 0.1
        'min_static_frames': 3,        # 對應 MIN_FLAT_LEN = 5
        'min_cycle_length': 15,        # 基於 RISE_TOTAL = 3.0 的概念
        'smooth_window': 7,            # 對應 FLAT_WINDOW = 5
        'y_stability_threshold': 1.5   # Y 座標穩定性（新增參數）
    }
    
    # 自動檢測
    cycles = auto_detect_cycles(CSV_PATH, VIDEO_PATH, OUTPUT_DIR, params)
    
    # 啟動編輯器進行微調（如果需要）
    use_editor = input("\n是否啟動編輯器微調？(y/n): ").strip().lower()
    
    if use_editor == 'y':
        # 動態導入編輯器
        import importlib.util
        spec = importlib.util.spec_from_file_location("editor", "gait_cycle_interactive.py")
        editor_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(editor_module)
        
        df = pd.read_csv(CSV_PATH)
        editor = editor_module.CycleEditor(cycles, df, VIDEO_PATH, OUTPUT_DIR)
        editor.run()