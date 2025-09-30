from Bio.SeqIO.FastaIO import FastaIterator
from Bio import SeqIO
import pandas as pd
import numpy as np
import torch
import io
from typing import Union, Optional
from pathlib import Path
from tqdm.auto import tqdm
import joblib
import xgboost as xgb

# DrugRank integrated modules
try:
    from transformers import AutoTokenizer, AutoModel
    from unimol_lite.model import UniMolModel, UniMolConfig
    from unimol_lite.data import smiles2unimolitem
    DRUGRANK_AVAILABLE = True
except ImportError as e:
    print(f"Warning: DrugRank dependencies not available: {e}")
    DRUGRANK_AVAILABLE = False


def read_in_data_drugrank(
    in_data: Union[str, FastaIterator, pd.DataFrame],
    seq_col: str = "seq",
    name_col: str = "name",
) -> pd.DataFrame:
    """读取输入数据并转换为DataFrame格式"""
    if isinstance(in_data, str):
        if in_data.endswith(("fasta", "fa", "fna")):
            out = pd.DataFrame([
                {name_col: i.id, seq_col: str(i.seq)} 
                for i in SeqIO.parse(in_data, "fasta")
            ])
        elif in_data.endswith("csv"):
            out = pd.read_csv(in_data)
        elif in_data.endswith("tsv"):
            out = pd.read_csv(in_data, sep="\t")
        elif in_data.endswith("xlsx"):
            out = pd.read_excel(in_data)
        else:
            raise ValueError("Input file format not supported")
    elif isinstance(in_data, FastaIterator):
        out = pd.DataFrame([
            {name_col: i.id, seq_col: str(i.seq)}
            for i in in_data
        ])
    else:
        out = in_data
    return out


def get_rna_embeddings_local(
    rna_list,
    batch_size: int = 40960,
    model_path: str = None,
    device=None
):
    """提取RNA序列的UniRNA嵌入表征"""
    if not DRUGRANK_AVAILABLE:
        raise ImportError("DrugRank dependencies not available")
    
    if model_path is None:
        # 使用UniRNA App的预训练模型路径
        from .utils import PRETRAINED
        model_path = PRETRAINED["L16"]
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)
    
    # 加载tokenizer和模型
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path).half().to(device)
    model.eval()
    
    # 去重并保留顺序
    unique_rnas = []
    seen = set()
    for rna in rna_list:
        if rna not in seen:
            unique_rnas.append(rna)
            seen.add(rna)
    
    embeddings_dict = {}
    
    # 批量处理
    for i in range(0, len(unique_rnas), batch_size):
        batch_sequences = unique_rnas[i:i + batch_size]
        
        try:
            inputs = tokenizer(
                batch_sequences, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=64
            )
            
            inputs = {key: val.to(device) for key, val in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
            
            batch_embeddings = outputs.pooler_output.float().cpu().numpy()
            
            for seq, embedding in zip(batch_sequences, batch_embeddings):
                embeddings_dict[seq] = embedding
                
        except Exception as e:
            print(f"Error processing RNA batch: {e}")
            continue
    
    return embeddings_dict


def get_unimol_representations_local(
    smiles_list, 
    batch_size=1, 
    model_path=None,
    device=None
):
    """从SMILES列表中提取UniMol表征"""
    if not DRUGRANK_AVAILABLE:
        raise ImportError("DrugRank dependencies not available")
    
    if model_path is None:
        # 使用默认的UniMol模型路径 
        model_path = "/home/dingsz/project/rna_app_review/unirna_app/base_unimol"
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    
    # 加载模型
    import os
    config = UniMolConfig.from_pretrained(model_path)
    model = UniMolModel(config)
    
    # 加载权重文件
    if os.path.exists(f"{model_path}/model.safetensors"):
        from safetensors.torch import load_file
        state_dict = load_file(f"{model_path}/model.safetensors")
        model.load_state_dict(state_dict, strict=False)
    elif os.path.exists(f"{model_path}/pytorch_model.bin"):
        state_dict = torch.load(f"{model_path}/pytorch_model.bin", map_location=device)
        model.load_state_dict(state_dict, strict=False)
    else:
        raise FileNotFoundError(f"No model weights found in {model_path}")
    
    model.to(device)
    model.eval()
    
    embeddings_dict = {}
    
    for smiles in tqdm(smiles_list, desc="Processing SMILES"):
        try:
            items = smiles2unimolitem(smiles)
            
            def move_to_device(obj):
                if isinstance(obj, torch.Tensor):
                    return obj.to(device)
                elif isinstance(obj, dict):
                    return {k: move_to_device(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [move_to_device(v) for v in obj]
                else:
                    return obj
            
            items = move_to_device(items)
            
            with torch.no_grad():
                encoder_output, _ = model(**items)
                pooled_output = encoder_output.mean(dim=1).squeeze().cpu().numpy()
                embeddings_dict[smiles] = pooled_output
                
        except Exception as e:
            print(f"Error processing SMILES '{smiles}': {e}")
            embeddings_dict[smiles] = None
    
    return embeddings_dict


def predict_rankings_local(smiles_df, rna_embeddings_dict, model_path):
    """为每种RNA生成对SMILES的预测排名"""
    # 加载模型
    model_data = joblib.load(model_path)
    model = model_data['model']
    
    # 检查必要的列
    if 'unimol_embedding' not in smiles_df.columns:
        raise ValueError("smiles_df必须包含'unimol_embedding'列")
    
    if 'smiles' not in smiles_df.columns:
        raise ValueError("smiles_df必须包含'smiles'列")
    
    results_df = pd.DataFrame({'SMILES': smiles_df['smiles']})
    
    # 为每种RNA生成预测和排名
    for rna_seq, rna_embedding in tqdm(rna_embeddings_dict.items(), desc="Processing RNAs"):
        X_combined = []
        valid_indices = []
        
        for idx, row in smiles_df.iterrows():
            if row['unimol_embedding'] is not None:
                combined_embedding = np.concatenate([row['unimol_embedding'], rna_embedding])
                X_combined.append(combined_embedding)
                valid_indices.append(idx)
        
        if not X_combined:
            print(f"Warning: No valid SMILES features for RNA '{rna_seq}'")
            results_df[f'rank_{rna_seq}'] = np.nan
            continue
        
        X_combined = np.stack(X_combined)
        dtest = xgb.DMatrix(X_combined)
        
        try:
            predictions = model.predict(dtest)
            rank_series = pd.Series(
                index=valid_indices, 
                data=pd.Series(predictions).rank(ascending=False, method='min').astype(int)
            )
            results_df[f'rank_{rna_seq}'] = rank_series
            
        except Exception as e:
            print(f"Error predicting RNA '{rna_seq}': {e}")
            results_df[f'rank_{rna_seq}'] = np.nan
            continue
    
    return results_df


def infer_drugrank(
    csv_data: Union[str, pd.DataFrame],
    fasta_data: Union[str, FastaIterator, pd.DataFrame],
    output_dir: str,
    model_path: str = "/home/dingsz/project/rna_app_review/DrugRank/models/joint_xgboost_model_2ranks.pkl",
    return_df: bool = False,
) -> Optional[pd.DataFrame]:
    """
    使用DrugRank预测RNA与小分子化合物的结合亲和力排名
    
    Args:
        csv_data: CSV文件路径或DataFrame，必须包含smiles_prepared列
        fasta_data: FASTA文件路径、FastaIterator或DataFrame，包含RNA序列
        output_dir: 输出目录
        model_path: XGBoost模型文件路径
        return_df: 是否返回DataFrame
        
    Returns:
        如果return_df为True，返回包含预测结果的DataFrame
    """
    if not DRUGRANK_AVAILABLE:
        raise ImportError("DrugRank dependencies not available. Please install: xgboost, rdkit, unimol-lite")
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 处理CSV数据
    if isinstance(csv_data, str):
        csv_df = pd.read_csv(csv_data)
    else:
        csv_df = csv_data.copy()
    
    if 'smiles_prepared' not in csv_df.columns:
        raise ValueError("CSV数据必须包含'smiles_prepared'列")
    
    # 处理FASTA数据
    fasta_df = read_in_data_drugrank(fasta_data)
    
    # 提取SMILES特征
    print("Extracting SMILES features...")
    smiles_list = csv_df['smiles_prepared'].tolist()
    unimol_dict = get_unimol_representations_local(smiles_list)
    
    # 提取RNA特征
    print("Extracting RNA features...")
    rna_list = fasta_df['seq'].tolist()
    rna_ids = fasta_df['name'].tolist()
    rna_embeddings_sequence = get_rna_embeddings_local(rna_list)
    
    # 将序列映射回ID
    rna_embeddings_dict = {}
    for rna_id, rna_seq in zip(rna_ids, rna_list):
        if rna_seq in rna_embeddings_sequence:
            rna_embeddings_dict[rna_id] = rna_embeddings_sequence[rna_seq]
    
    # 准备SMILES DataFrame
    smiles_data = []
    for smiles in smiles_list:
        unimol_emb = unimol_dict.get(smiles)
        smiles_data.append({
            'smiles': smiles,
            'unimol_embedding': unimol_emb
        })
    
    smiles_df = pd.DataFrame(smiles_data)
    
    # 进行预测
    print("Predicting rankings...")
    results_df = predict_rankings_local(smiles_df, rna_embeddings_dict, model_path)
    
    # 保存结果
    output_path = f"{output_dir}/result.csv"
    results_df.to_csv(output_path, index=False)
    
    if return_df:
        return results_df