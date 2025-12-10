import json
import os

input_json = r"C:/mydata/sf/openpose/output_json/comb_json/1210_2.json"
output_dir = r"C:/mydata/sf/openpose/output_json/steps/1210_1"
os.makedirs(output_dir, exist_ok=True)

MIN_VISIBLE_RATIO = 0.7
BREAK_BUFFER = 30  # 連續 N 幀才斷段
MIN_SEG_LEN = 100   # 段長不足幀數則合併

with open(input_json, "r") as f:
    frames = json.load(f)

def is_frame_valid(frame):
    kp = frame["keypoints"]
    if not kp:
        return False
    confidences = kp[2::3]
    visible_ratio = sum(1 for c in confidences if c > 0) / len(confidences)
    return visible_ratio >= MIN_VISIBLE_RATIO

def is_turning_point(prev_frame, curr_frame):
    if not prev_frame or not curr_frame:
        return False
    prev_kp = prev_frame["keypoints"]
    curr_kp = curr_frame["keypoints"]
    indices = [(5,2), (12,9)]  # 左右肩, 左右髖
    for left, right in indices:
        prev_diff = prev_kp[left*3] - prev_kp[right*3]
        curr_diff = curr_kp[left*3] - curr_kp[right*3]
        if prev_diff * curr_diff < 0:
            return True
    return False

segments = []
curr_segment = []
break_count = 0

for i, frame in enumerate(frames):
    if curr_segment:
        if not is_frame_valid(frame) or is_turning_point(curr_segment[-1], frame):
            break_count += 1
        else:
            break_count = 0

        if break_count >= BREAK_BUFFER:
            # 真正斷段
            segments.append(curr_segment)
            curr_segment = []
            break_count = 0

    curr_segment.append(frame)

if curr_segment:
    segments.append(curr_segment)

# 合併過短段
merged_segments = []
for seg in segments:
    if merged_segments and len(seg) < MIN_SEG_LEN:
        merged_segments[-1].extend(seg)
    else:
        merged_segments.append(seg)

# 輸出
for idx, seg in enumerate(merged_segments):
    out_file = os.path.join(output_dir, f"step_{idx+1:03d}.json")
    with open(out_file, "w") as f:
        json.dump(seg, f, indent=4)
    print(f"輸出 {len(seg)} 幀 → {out_file}")

print(f"完成！總共分出 {len(merged_segments)} 個步態週期")
