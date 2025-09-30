import subprocess
import sys
from pathlib import Path
import pandas as pd
from .utils import CHEKPOINTS, PRETRAINED


def infer_seq_optimization(
    in_data: str,  # 模板序列
    output_dir: str,
    mutation_ratio: float = 0.1,
    iterations: int = 20,
    model_weight: str = "trna",
    return_df: bool = False,
) -> pd.DataFrame | None:
    """
    RNA序列优化：使用微调的Uni-RNA模型优化模板序列
    
    Args:
        in_data: 模板序列字符串
        output_dir: 输出目录
        mutation_ratio: 突变比例 (0.01-0.5)
        iterations: 优化迭代次数
        model_weight: 模型权重类型 ("trna" or "5utr")
        return_df: 是否返回DataFrame
    
    Returns:
        优化后的序列DataFrame或None
    """
    # seq_generator.py的路径
    seq_generator_path = Path(__file__).parent / "seq_generator.py"
    
    # 从CHEKPOINTS和PRETRAINED获取模型路径，根据model_weight选择
    model_key = f"{model_weight}_seq_optimization"
    model_path = CHEKPOINTS[model_key]
    base_model_path = PRETRAINED["L16"]
    
    # 构建命令
    cmd = [
        sys.executable,  # python
        str(seq_generator_path),
        "--saved_model_path", str(model_path),
        "--base_model_weights", str(base_model_path),
        "--template", in_data,
        "--mutation_ratio", str(mutation_ratio),
        "--iterations", str(iterations),
        "--output_dir", output_dir,
    ]
    
    # 执行命令
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )
    
    if result.returncode != 0:
        raise RuntimeError(f"Sequence optimization failed: {result.stderr}")
    
    # 查找输出文件
    output_files = list(Path(output_dir).glob("output_sequence*.csv"))
    
    if not output_files:
        raise FileNotFoundError(f"No output file found in {output_dir}")
    
    # 读取最新的结果文件
    latest_file = max(output_files, key=lambda p: p.stat().st_mtime)
    df = pd.read_csv(latest_file)
    
    # 保存为标准格式
    df.to_csv(f"{output_dir}/optimized_sequences.csv", index=False)
    
    if return_df:
        return df
    return None
