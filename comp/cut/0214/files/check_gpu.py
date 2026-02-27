"""
GPU環境檢測工具 - 完整版
適用於 RTX 4060
"""

print("="*60)
print("🔍 GPU環境檢測工具")
print("="*60)

# 1. 檢查Python版本
import sys
print(f"\n1️⃣ Python版本: {sys.version.split()[0]}")
if sys.version_info < (3, 8):
    print("   ⚠️  Python版本太舊,建議3.8以上")
else:
    print("   ✓ Python版本OK")

# 2. 檢查TensorFlow
print("\n2️⃣ TensorFlow檢測:")
try:
    import tensorflow as tf
    tf_version = tf.__version__
    print(f"   當前版本: {tf_version}")
    
    # 判斷版本
    major_version = int(tf_version.split('.')[0])
    
    if major_version < 2:
        print("   ❌ TensorFlow 1.x - 太舊!")
        print("   → 需要升級到 TensorFlow 2.x")
        need_upgrade = True
    else:
        print(f"   ✓ TensorFlow 2.x")
        need_upgrade = False
        
        # 檢查GPU
        try:
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                print(f"   ✓ 偵測到 {len(gpus)} 個GPU:")
                for i, gpu in enumerate(gpus):
                    print(f"      GPU {i}: {gpu.name}")
            else:
                print("   ⚠️  未偵測到GPU (可能是CPU版)")
        except Exception as e:
            print(f"   ⚠️  GPU檢測錯誤: {e}")
            
except ImportError:
    print("   ❌ TensorFlow未安裝")
    need_upgrade = True

# 3. 檢查NVIDIA GPU
print("\n3️⃣ NVIDIA GPU檢測:")
import subprocess
try:
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5, shell=True)
    if result.returncode == 0:
        print("   ✓ nvidia-smi 可執行")
        # 提取GPU型號
        lines = result.stdout.split('\n')
        for line in lines:
            if 'RTX' in line or 'GTX' in line or 'GeForce' in line:
                # 簡化顯示
                parts = line.split('|')
                if len(parts) > 1:
                    gpu_info = parts[1].strip()
                    print(f"   → 顯示卡: {gpu_info}")
                break
    else:
        print("   ⚠️  nvidia-smi 執行失敗")
except FileNotFoundError:
    print("   ❌ nvidia-smi 未找到 - NVIDIA驅動未安裝")
except Exception as e:
    print(f"   ⚠️  錯誤: {e}")

# 4. 檢查其他相關套件
print("\n4️⃣ 相關套件檢測:")
packages = ['numpy', 'pandas', 'matplotlib', 'scikit-learn']
for pkg in packages:
    try:
        module = __import__(pkg)
        version = getattr(module, '__version__', '未知')
        print(f"   ✓ {pkg}: {version}")
    except ImportError:
        print(f"   ⚠️  {pkg}: 未安裝")

# 5. 生成修復指令
print("\n" + "="*60)
print("📋 修復步驟 (針對 RTX 4060)")
print("="*60)

print("\n請在命令提示字元 (不是Python) 中執行以下指令:")
print("\n--- 步驟 1: 完全移除舊版TensorFlow ---")
print("pip uninstall tensorflow tensorflow-gpu keras -y")

print("\n--- 步驟 2: 安裝新版TensorFlow (支援RTX 4060) ---")
print("pip install tensorflow==2.15.0")

print("\n--- 步驟 3: 驗證安裝 ---")
print('python check_gpu.py')

print("\n" + "="*60)
print("💡 重要說明")
print("="*60)
print("1. RTX 4060 需要 TensorFlow 2.10+ 才能支援")
print("2. TensorFlow 2.15 會自動包含 CUDA 支援,不需手動安裝")
print("3. 確保 NVIDIA 驅動是最新版 (建議 535.x 以上)")
print("4. 如果還是偵測不到GPU,可能需要:")
print("   - 更新顯卡驅動: https://www.nvidia.com/download/index.aspx")
print("   - 重新開機")

print("\n" + "="*60)
input("\n按 Enter 鍵關閉...")
