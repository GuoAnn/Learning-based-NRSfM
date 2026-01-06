import os
import torch
import gc

def apply_fixes():
    """应用所有必要的修复"""
    
    # 1. 清理环境变量
    if 'PYTORCH_CUDA_ALLOC_CONF' in os.environ:
        del os.environ['PYTORCH_CUDA_ALLOC_CONF']
    
    # 2. 清理GPU内存
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        print(f"GPU内存已清理")
        print(f"可用内存: {torch.cuda.get_device_properties(0).total_memory/1024**3:.2f} GB")
    
    # 3. 设置较小的默认批次大小（如果需要）
    if 'BATCH_SIZE' not in os.environ:
        os.environ['BATCH_SIZE'] = '8'
    
    # 4. 启用cudnn基准优化
    torch.backends.cudnn.benchmark = True
    
    print("修复已应用")
    return True

if __name__ == "__main__":
    apply_fixes()
    print("现在可以运行主程序了:")
    print("python main.py")