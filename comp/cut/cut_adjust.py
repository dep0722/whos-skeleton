import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.widgets import Button, TextBox, CheckButtons, Slider
import cv2
import os
import threading
import time

# =========================
# 參數(為你的資料量身訂)
# =========================
CSV_PATH = r"C:\mydata\sf\open\output_csv\0128\S\0130_S_2.csv"
VIDEO_PATH = r"C:\mydata\sf\open\walking_video\0130\0130_S_2.mp4"
OUTPUT_DIR = r"C:\mydata\sf\open\output_csv\0128cut\S\S_2"

FLAT_WINDOW = 5
FLAT_SLOPE = 0.1
MIN_FLAT_LEN = 5
RISE_TOTAL = 3.0

NUM_COLORS = 20
cmap = colormaps["tab20"]

# =========================
# 讀資料
# =========================
df = pd.read_csv(CSV_PATH)

ankle_cols = {
    "L": ("LAnkle_x", "LAnkle_y"),
    "R": ("RAnkle_x", "RAnkle_y")
}

# =========================
# 找出所有週期
# =========================
all_cycles = []

for side, (col_x, col_y) in ankle_cols.items():
    x = df[col_x].values
    y = df[col_y].values
    n = len(x)

    # ---------- 找平段 ----------
    is_flat = np.zeros(n, dtype=bool)
    for i in range(n - FLAT_WINDOW):
        slope = (x[i + FLAT_WINDOW] - x[i]) / FLAT_WINDOW
        if abs(slope) < FLAT_SLOPE:
            is_flat[i:i + FLAT_WINDOW] = True

    # ---------- 合併平段 ----------
    flat_segments = []
    start = None
    for i, v in enumerate(is_flat):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start >= MIN_FLAT_LEN:
                flat_segments.append((start, i))
            start = None
    if start is not None and n - start >= MIN_FLAT_LEN:
        flat_segments.append((start, n))

    if len(flat_segments) < 2:
        print(f"{side} 平段不足")
        continue

    # ---------- 平段 + 上升 = 週期 ----------
    for i in range(len(flat_segments) - 1):
        fs, fe = flat_segments[i]
        ns, _ = flat_segments[i + 1]

        if abs(x[ns] - x[fe]) > RISE_TOTAL:
            all_cycles.append({
                'side': side,
                'start': fs,
                'end': ns,
                'name': f"{side}_1_",
                'valid': True
            })

print(f"總共找到 {len(all_cycles)} 個週期")

# =========================
# 每個 side 分開處理，第一次轉彎前不累計 n
# =========================
for side in ["L", "R"]:
    side_cycles = [c for c in all_cycles if c['side'] == side]
    n = 0
    turn_count = 0
    last_walk_dir = None

    for c in side_cycles:
        s, e = c['start'], c['end']
        x0, x1 = df[f"{side}Ankle_x"].iloc[s], df[f"{side}Ankle_x"].iloc[e-1]
        walk_dir = "right" if (x1 - x0) > 0 else "left"

        if last_walk_dir is None:
            n = 1
            c['name'] = f"{side}_{n}_"
            last_walk_dir = walk_dir
            continue

        if walk_dir != last_walk_dir:
            turn_count += 1
            if (side == "L" and turn_count % 2 == 1) or (side == "R" and turn_count % 2 == 0):
                n += 1

        c['name'] = f"{side}_{n}_"
        last_walk_dir = walk_dir

# =========================
# 互動式編輯器
# =========================
class CycleEditor:
    def __init__(self, cycles, df, video_path, output_dir):
        self.cycles = cycles
        self.df = df
        self.video_path = video_path
        self.output_dir = output_dir
        self.current_idx = 0
        self.video_thread = None
        self.video_running = False
        self.video_paused = False
        self.current_frame = 0
        self.frame_start = 0
        self.frame_end = 0
        
        os.makedirs(output_dir, exist_ok=True)
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            print(f"警告: 無法開啟影片 {video_path}")
            self.cap = None
        
        self.setup_ui()
        self.update_display()
        
    def setup_ui(self):
        self.fig = plt.figure(figsize=(16, 10))
        
        # 上方資訊
        ax_info = plt.axes([0.1, 0.85, 0.8, 0.1])
        ax_info.axis('off')
        self.info_text = ax_info.text(0.5, 0.5, '', ha='center', va='center', fontsize=12)
        
        # 中間折線圖
        self.ax_plot = plt.axes([0.1, 0.4, 0.8, 0.4])
        
        # 控制按鈕
        ax_prev = plt.axes([0.1, 0.3, 0.08, 0.05])
        ax_next = plt.axes([0.19, 0.3, 0.08, 0.05])
        ax_save = plt.axes([0.7, 0.3, 0.08, 0.05])
        ax_export = plt.axes([0.79, 0.3, 0.08, 0.05])
        self.btn_prev = Button(ax_prev, 'Previous')
        self.btn_next = Button(ax_next, 'Next')
        self.btn_save = Button(ax_save, 'Save')
        self.btn_export = Button(ax_export, 'Export Current')
        self.btn_prev.on_clicked(self.prev_cycle)
        self.btn_next.on_clicked(self.next_cycle)
        self.btn_save.on_clicked(self.save_adjustment)
        self.btn_export.on_clicked(self.export_current)
        
        # 影片控制
        ax_play = plt.axes([0.35, 0.3, 0.08, 0.05])
        ax_pause = plt.axes([0.44, 0.3, 0.08, 0.05])
        ax_stop = plt.axes([0.53, 0.3, 0.08, 0.05])
        self.btn_play = Button(ax_play, 'Play')
        self.btn_pause = Button(ax_pause, 'Pause')
        self.btn_stop = Button(ax_stop, 'Stop')
        self.btn_play.on_clicked(self.play_video)
        self.btn_pause.on_clicked(self.pause_video)
        self.btn_stop.on_clicked(self.stop_video)
        
        # 影片滑桿
        ax_slider = plt.axes([0.1, 0.25, 0.8, 0.03])
        self.frame_slider = Slider(ax_slider, 'Frame', 0, 100, valinit=0, valstep=1)
        self.frame_slider.on_changed(self.on_frame_change)
        
        # 微調輸入框
        ax_start = plt.axes([0.2, 0.18, 0.15, 0.05])
        ax_end = plt.axes([0.2, 0.11, 0.15, 0.05])
        ax_name = plt.axes([0.2, 0.04, 0.15, 0.05])
        self.tb_start = TextBox(ax_start, 'Start: ')
        self.tb_end = TextBox(ax_end, 'End: ')
        self.tb_name = TextBox(ax_name, 'Name: ')

        ax_jump = plt.axes([0.75, 0.18, 0.1, 0.05])
        self.tb_jump = TextBox(ax_jump, 'Jump:')

        ax_jump_btn = plt.axes([0.86, 0.18, 0.08, 0.05])
        self.btn_jump = Button(ax_jump_btn, 'Go')
        self.btn_jump.on_clicked(self.jump_to_cycle)



        # 新增 n 欄位
        ax_n = plt.axes([0.5, 0.04, 0.1, 0.05])
        self.tb_n = TextBox(ax_n, 'n: ')

        # 有效性
        ax_valid = plt.axes([0.5, 0.11, 0.15, 0.1])
        self.check_valid = CheckButtons(ax_valid, ['Valid'], [True])
        
    def update_display(self):
        if self.current_idx >= len(self.cycles):
            return
        cycle = self.cycles[self.current_idx]
        start, end = cycle['start'], cycle['end']
        self.frame_start, self.frame_end, self.current_frame = start, end, start
        
        # 更新滑桿
        self.frame_slider.valmin = start
        self.frame_slider.valmax = end-1
        self.frame_slider.set_val(start)
        self.frame_slider.ax.set_xlim(start, end-1)
        
        # 更新資訊
        info = f"週期 {self.current_idx+1}/{len(self.cycles)} | {cycle['side']} ankle | Frames: {start}-{end}"
        self.info_text.set_text(info)
        
        # 更新折線圖
        self.ax_plot.clear()
        col_y = f"{cycle['side']}Ankle_y"
        y = self.df[col_y].values[start:end]
        self.ax_plot.plot(range(start, end), y, 'b-', linewidth=2)
        self.ax_plot.axvline(start, color='g', linestyle='--', label='Start')
        self.ax_plot.axvline(end, color='r', linestyle='--', label='End')
        self.ax_plot.set_xlabel('Frame Index')
        self.ax_plot.set_ylabel('Y Position')
        self.ax_plot.set_title(f"{cycle['name']}")
        self.ax_plot.legend()
        self.ax_plot.grid(True)
        
        # 更新輸入框
        self.tb_start.set_val(str(start))
        self.tb_end.set_val(str(end))
        self.tb_name.set_val(cycle['name'])
        # 更新 n
        parts = cycle['name'].split('_')
        self.tb_n.set_val(str(int(parts[1]) if len(parts)>=2 and parts[1].isdigit() else 1))

        # 更新有效性
        current_status = self.check_valid.get_status()[0]
        if current_status != cycle['valid']:
            self.check_valid.set_active(0)
        
        plt.draw()
        self.show_frame(start)
        

    def jump_to_cycle(self, event):
        try:
            f = int(self.tb_jump.text)

            # 目前正在編輯的 side
            current_side = self.cycles[self.current_idx]['side']

            # 只在同 side 搜尋
            for i, c in enumerate(self.cycles):
                if c['side'] != current_side:
                    continue

                if c['start'] <= f < c['end']:
                    self.current_idx = i
                    self.update_display()

                    self.frame_slider.set_val(f)
                    self.show_frame(f)

                    print(f"✓ 跳到 {current_side} cycle {i+1}")
                    return

            print(f"✗ 該 frame 不在 {current_side} 的任何 cycle")

        except:
            print("✗ Jump 輸入錯誤")



    def on_frame_change(self, val):
        frame_idx = int(val)
        self.current_frame = frame_idx
        self.show_frame(frame_idx)
    
    def show_frame(self, frame_idx):
        if self.cap is None:
            return
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        if not ret:
            return
        cv2.putText(frame, f"Frame: {frame_idx}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        window_name = "Video Preview"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 720, 480)
        cv2.imshow(window_name, frame)
        cv2.waitKey(1)
    
    def play_video(self, event):
        self.video_paused = False
        if not self.video_running:
            self.video_running = True
            self.video_thread = threading.Thread(target=self._video_loop, daemon=True)
            self.video_thread.start()
    
    def pause_video(self, event):
        self.video_paused = True
    
    def stop_video(self, event):
        self.video_running = False
        self.video_paused = False
    
    def _video_loop(self):
        while self.video_running:
            if not self.video_paused:
                self.current_frame += 1
                if self.current_frame >= self.frame_end:
                    self.current_frame = self.frame_start
                self.frame_slider.set_val(self.current_frame)
                self.show_frame(self.current_frame)
                time.sleep(0.033)
            else:
                time.sleep(0.1)
    
    def prev_cycle(self, event):
        if self.current_idx > 0:
            self.video_running = False
            self.video_paused = False
            if self.video_thread and self.video_thread.is_alive():
                self.video_thread.join(timeout=0.5)
            self.current_idx -= 1
            self.update_display()
    
    def next_cycle(self, event):
        if self.current_idx < len(self.cycles)-1:
            self.video_running = False
            self.video_paused = False
            if self.video_thread and self.video_thread.is_alive():
                self.video_thread.join(timeout=0.5)
            self.current_idx += 1
            self.update_display()
    
    def save_adjustment(self, event):
        try:
            new_start = int(self.tb_start.text)
            new_end = int(self.tb_end.text)
            is_valid = self.check_valid.get_status()[0]
            new_name = self.tb_name.text.strip()


            # 讀新 n
            try:
                new_n = int(self.tb_n.text)
            except:
                new_n = 1

            cycle = self.cycles[self.current_idx]

            # ===== 解析舊 n =====
            parts = cycle['name'].split('_')
            old_n = int(parts[1]) if len(parts) >= 2 and parts[1].isdigit() else 1

            delta = new_n - old_n
            side = cycle['side']

            # ===== 讀取舊 name 的 k 部分 =====
            old_name = cycle['name']
            k_part = ""

            parts_full = old_name.split('_')
            if len(parts_full) >= 3:
                k_part = parts_full[2]  # 第三段就是 k

            # ===== 更新當前 cycle =====
            cycle['start'] = new_start
            cycle['end'] = new_end
            cycle['valid'] = is_valid

            # 如果使用者有改 name，就用使用者的
            if new_name != "":
                cycle['name'] = new_name
            else:
                if k_part != "":
                    cycle['name'] = f"{side}_{new_n}_{k_part}"
                else:
                    cycle['name'] = f"{side}_{new_n}_"




            # ===== 連鎖更新後面同側 cycle =====
            if delta != 0:
                for j in range(self.current_idx + 1, len(self.cycles)):
                    c = self.cycles[j]

                    if c['side'] != side:
                        continue

                    p = c['name'].split('_')
                    if len(p) >= 2 and p[1].isdigit():
                        n_j = int(p[1])
                        n_j += delta
                        if n_j < 1:
                            n_j = 1
                        pp = c['name'].split('_')
                        k2 = pp[2] if len(pp) >= 3 else ""

                        if k2 != "":
                            c['name'] = f"{side}_{n_j}_{k2}"
                        else:
                            pp = c['name'].split('_')
                            k2 = pp[2] if len(pp) >= 3 else ""

                            if k2:
                                c['name'] = f"{side}_{n_j}_{k2}"
                            else:
                                c['name'] = f"{side}_{n_j}_"



            print("✓ 已儲存調整 (含後續自動更新)")
            print(f"  新名稱: {cycle['name']}")
            print(f"  範圍: {new_start}-{new_end}")
            print(f"  有效: {is_valid}")

            self.video_running = False
            self.video_paused = False
            if self.video_thread and self.video_thread.is_alive():
                self.video_thread.join(timeout=0.5)

            self.update_display()

        except Exception as e:
            print(f"✗ 儲存失敗: {e}")

    
    def export_current(self, event):
        cycle = self.cycles[self.current_idx]
        if not cycle['valid']:
            print(f"✗ 無法輸出: 當前週期標記為無效")
            return
        start, end, name = cycle['start'], cycle['end'], cycle['name']
        segment_df = self.df.iloc[start:end].copy()
        segment_df['frame_index'] = range(len(segment_df))
        output_path = os.path.join(self.output_dir, f"{name}.csv")
        segment_df.to_csv(output_path, index=False)
        print(f"✓ 已輸出: {output_path}")
        print(f"  原始範圍: frame {start}-{end}")
        print(f"  輸出長度: {len(segment_df)} frames (重置為 0-{len(segment_df)-1})")
        if self.current_idx < len(self.cycles)-1:
            self.next_cycle(None)
    
    def run(self):
        plt.show()
        self.video_running = False
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()

# =========================
# 啟動編輯器
# =========================
editor = CycleEditor(all_cycles, df, VIDEO_PATH, OUTPUT_DIR)
editor.run()
