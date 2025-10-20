import sys
sys.path.append("/home/wym/utr_design")

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import logging
from transformers import AutoTokenizer, AutoModel
import unirna_tf
from torch import Tensor
from typing import Optional, List
import torch
import math
import random
import argparse
from torch import nn
import time


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_logging(output_dir):
    os.makedirs(output_dir, exist_ok=True)

    log_file = os.path.join(output_dir, f"mutation_optimization.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    return log_file


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='RNA Sequence Mutation Optimization')

    # 基础参数
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU id to use')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed for reproducibility')
    parser.add_argument('--output_dir', type=str, default='logs/test_gen',
                        help='Directory to save logs: the input commands, init sequences, sequences iterated each time, final sequence and the predicted half-life')
    parser.add_argument('--base_model_weights', type=str,
                        default='checkpoints/pretrained/unirna_L16_E1024_DPRNA500M_STEP400K',
                        help='Path to the UniRNA base model weights')
    parser.add_argument('--saved_model_path', type=str,
                        default='checkpoints/lite_ckpts/best_unirna_reg_model.pth',
                        help='Path to the saved regression model weights')

    # 数据
    parser.add_argument('--template', type=str, default='',
                        help='Template sequence for mutation, eg. AAAAATTTTATTTTTTAATTATGTAAAGTGAATTAGAATGTTGTTTTTTT')
    
    parser.add_argument('--iterations', type=int, default=20,
                        help='Number of mutation iterations')
    parser.add_argument('--selection_mode', type=str, default='soft',
                        choices=['hard', 'soft'],   # 等待后续补充
                        help='Method to select important positions for mutation: hard, soft, etc.')
    parser.add_argument('--importance_type', type=str, default='gradient',
                        choices=['gradient'],   # 等待后续补充
                        help='Method to compute importance scores for mutation: attention, gradient, etc.')
    parser.add_argument('--num_candidates', type=int, default=10,
                        help='Number of candidate sequences generated in each iteration')
    parser.add_argument('--mutation_ratio', type=float, default=0.04,
                        help='Ratio of positions to mutate in each iteration')
    parser.add_argument('--elitism_ratio', type=float, default=0.2,
                        help='Fraction of top sequences to retain from previous generation during elitism selection')
    
    parser.add_argument('--allow_mutate_all_sites', type=bool, default=True,
                        help='Whether to allow mutation at all sites, True means all bp can be mutated, False means only "N" positions in template can be mutated')
    parser.add_argument('--protected_regions', type=str, default='',
                        help="Comma-separated list of mutation range boundaries, e.g., '1,4,9,13' for intervals [1:4] and [9:13]")
    
    parser.add_argument("--csv_path", type=str, default="",
                        help="Path to the input CSV file to train the model, or to give a reference for sequence generation")
    parser.add_argument('--seq_column_name', type=str, default='sequence',
                        help='The name of the colomn as input sequence in csv file')
    parser.add_argument('--label_column_name', type=str, default='norm_eff',
                        help='The name of the colomn as label in csv file')
    
    return parser.parse_args()


class UniRNAWithAdaptiveRegression_linearout(nn.Module):
    '''
    Regression Model with adaptive regression head: B*L*C -> B*C(1024) -> MLP
    '''

    def __init__(self, base_model, output_dim=16):
        super(UniRNAWithAdaptiveRegression_linearout, self).__init__()
        self.base_model = base_model

        self.conv_reg_head = nn.Sequential(
            nn.Conv1d(in_channels=1024, out_channels=256, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.1),

            nn.Conv1d(in_channels=256, out_channels=64, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.1),

            nn.Conv1d(in_channels=64, out_channels=16, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.Dropout(0.1)
        )

        self.final_linear = nn.Sequential(
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(8, 1),
        )

    def forward(self, **inputs):
        # 获取UniRNA的输出
        outputs = self.base_model(**inputs, output_attentions=False)
        # (batch_size, sequence_length, hidden_size)
        last_hidden_state = outputs.last_hidden_state
        # print(last_hidden_state.shape)
        '''
        池化去掉length维度
        '''
        # conv
        last_hidden_state = last_hidden_state.permute(0, 2, 1)
        conv_reg_output = self.conv_reg_head(
            last_hidden_state).permute(0, 2, 1)
        conv_reg_output, _ = torch.max(conv_reg_output, dim=1)
        regression_output = self.final_linear(conv_reg_output)
        return regression_output



def load_model_weights(model, weights_path):
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"Model weights file not found: {weights_path}")
    checkpoint = torch.load(weights_path, map_location='cpu')
    res = model.load_state_dict(checkpoint['model_state_dict'])
    logging.info(f"Model weights loaded from {weights_path} with result: {res}")
    return model


def save_sequences(
    sequences: List[str],
    scores: List[float],
    iteration: int,
    output_csv: str,
    energies: Optional[List[float]] = None,
    structures: Optional[List[str]] = None
):
    """
    保存每次迭代选择的top-k序列及其分数（可选RNAFold能量和结构）。

    Args:
        sequences: 序列列表
        scores: 预测分数列表
        iteration: 当前迭代轮次
        output_csv: 输出CSV文件路径
        energies: 可选，RNAFold能量列表
        structures: 可选，RNA二级结构列表
    """
    data = {
        "Iteration": [f"{iteration}_{j}" for j in range(len(sequences))],
        "Sequence": sequences,
        "Predicted Score": scores,
    }
    if energies is not None:
        data["RNAFold Energy"] = energies
    if structures is not None:
        data["Structure"] = structures

    df = pd.DataFrame(data)
    write_header = not os.path.exists(output_csv)
    df.to_csv(output_csv, mode='w' if write_header else 'a', header=write_header, index=False)


def plot_mean_score_trend(all_scores_per_iteration, output_dir):
    """绘制每一轮的平均得分变化趋势"""
    mean_scores = [sum(scores) / len(scores)
                   for scores in all_scores_per_iteration]
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(mean_scores) + 1),
             mean_scores, marker='o', linestyle='-')
    plt.xlabel('Iteration')
    plt.ylabel('Mean Predicted Score')
    plt.title('Mean Predicted Score Over Iterations')
    plt.grid()
    plt.savefig(os.path.join(output_dir, 'mean_score_trend.png'))
    plt.show()


def plot_score_density_curve(all_scores_per_iteration, reference_scores, output_dir):
    """绘制所有轮次的得分概率密度曲线，并随着迭代次数增加逐渐加深颜色"""
    plt.figure(figsize=(10, 6))
    iterations = len(all_scores_per_iteration)
    cmap = plt.get_cmap('coolwarm')

    for i, scores in enumerate(all_scores_per_iteration):
        color = cmap(i / max(1, iterations - 1))
        # 防止分数全相同导致kdeplot报错
        sns.kdeplot(scores, label=f'Iteration {i+1}', fill=True, color=color)

    if reference_scores is not None:
        sns.kdeplot(reference_scores, label="Original Distribution", fill=True, color='lightgray', linewidth=2)

    plt.xlabel('Predicted Score')
    plt.ylabel('Probability Density')
    plt.title('Score Density Across Iterations')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'mutation_scores_density_curve.png'))


class MaskedGenerator:
    """RNA序列突变生成器"""

    def __init__(
        self, template: str,
        max_length: int = 9999,
        allow_mutate_all_sites: bool = True,
        protected_regions: str = "",
        selection_mode: str = "soft",  # "hard" or "soft"
        softmax_temp: float = 1.0,  # 仅在selection_mode="soft"时有效
        device: str = "cpu"
    ):
        '''
        Args:
            template: 原始模板序列（字符串，T会自动转为U）
            max_length: 序列最大长度
            allow_mutate_all_sites: 是否允许所有位点突变（True=全突变，False=只突变N）
            protected_regions: 需要保护（禁止突变）的区间，格式如 '1,4,9,13'，表示[1:4]和[9:13]
            selection_mode: 突变位点选择模式，"hard"=基于重要性分数选择top位点突变，"soft"=基于分数加权随机选择突变位点
        '''
        assert template, "template sequence must be provided"
        self.template = template.upper().replace("T", "U")[:max_length]
        self.max_length = max_length
        self.allow_mutate_all_sites = allow_mutate_all_sites
        self.selection_mode = selection_mode
        self.softmax_temp = softmax_temp
        assert softmax_temp > 0, "softmax_temp must be positive"

        # 防止生成重复序列（包括选用的和未选用的，只要出现过就不再生成）
        self.all_generated_sequences = set()

        # 初始化mask，True表示该位点禁止突变
        if allow_mutate_all_sites:
            self.mutation_mask = torch.zeros(len(self.template), dtype=torch.bool).to(device)
        else:
            # 只允许N突变，非N禁止突变
            self.mutation_mask = torch.tensor([base != "N" for base in self.template], dtype=torch.bool).to(device)

        # 解析并应用保护区间
        if protected_regions:
            region_indices = list(map(int, protected_regions.split(',')))
            if len(region_indices) % 2 != 0:
                raise ValueError("protected_regions 必须为偶数个数字，表示区间对")
            for start, end in zip(region_indices[::2], region_indices[1::2]):
                # 将负数索引转换为正数索引
                if start < 0:
                    start += len(self.template)
                if end <= 0:
                    end += len(self.template)
                if start < 0 or end > len(self.template) or start >= end:
                    raise ValueError(f"非法保护区间: [{start}, {end})")
                self.mutation_mask[start:end] = True  # 这些区间禁止突变
        print(f"Initialized MaskedGenerator with template length {len(self.template)}. Total masked positions: {self.mutation_mask.sum().item()}")

    def generate_initial_candidates(
        self,
        num_candidates: int,
        tokens: List[str] = ["A", "U", "G", "C"]
    ) -> List[str]:
        """
        生成初始候选序列列表。
        每个候选序列在允许突变的位置随机采样碱基，其余位置保持模板不变。

        Args:
            num_candidates: 生成的候选序列数量
            tokens: 可用的碱基集合

        Returns:
            List[str]: 生成的候选序列列表
        """
        # 包含模板序列在内的候选序列列表
        candidates = [self.template]
        # # 不再随机生成序列，而是直接使用后续的重要性分数指导突变
        # for _ in range(num_candidates-1):
        #     seq = [
        #         self.template[i] if self.mutation_mask[i]  # True表示禁止突变
        #         else random.choice(tokens)
        #         for i in range(len(self.template))
        #     ]
        #     candidates.append("".join(seq))
        return candidates

    def mutate_single(
        self,
        sequence: str,
        mutation_rate: float = 0.05,
        importance_scores: Optional[Tensor] = None,
        tokens: List[str] = ["A", "U", "G", "C"]
    ) -> str:
        """
        生成单个突变序列。
        支持随机突变和基于注意力/梯度的定向突变。

        Args:
            sequence: 原始序列
            mutation_rate: 每轮突变的比例
            importance_scores: 可选，定向突变的打分（如梯度或注意力），高分优先突变
            tokens: 可用的碱基集合

        Returns:
            str: 突变后的新序列
        """
        seq_len = len(sequence)
        # 计算可突变位点
        mutable_positions = [i for i in range(seq_len) if not self.mutation_mask[i]]
        if not mutable_positions:
            # 没有可突变位点，直接返回原序列
            return sequence

        # 计算实际突变位点数量，至少突变1个
        num_mutate = max(1, int(len(mutable_positions) * mutation_rate))

        # 选择突变位点
        if importance_scores is not None:
            # 基于打分（如梯度/注意力）选择突变位点
            # 只考虑可突变位点的打分
            scores_mutable = importance_scores[mutable_positions].cpu().numpy()
            if self.selection_mode == "hard":
                # 选出分数最高的 num_mutate 个位点
                top_indices = np.argpartition(-scores_mutable, num_mutate)[:num_mutate]
                mutate_indices = [mutable_positions[i] for i in top_indices]
            elif self.selection_mode == "soft":
                # softmax采样
                probs = np.exp(scores_mutable / self.softmax_temp)
                probs /= probs.sum()
                mutate_indices = np.random.choice(mutable_positions, size=num_mutate, replace=False, p=probs)
            else:
                # 其他模式待补充
                raise NotImplementedError(f"Unsupported selection_mode: {self.selection_mode}")
        else:
            # 随机选择可突变位点
            mutate_indices = random.sample(mutable_positions, num_mutate)

        # 根据选择的突变位点生成新序列
        new_seq = list(sequence)
        for idx in mutate_indices:
            original_base = new_seq[idx]
            # 保证突变后碱基和原来不同
            choices = [b for b in tokens if b != original_base]
            new_seq[idx] = random.choice(choices) if choices else original_base

        return "".join(new_seq)
    
    def mutate_multiple(
        self,
        sequences: List[str],
        num_mutation: int,
        mutation_ratio: float = 0.1,
        importance_scores_list: Optional[List[Optional[Tensor]]] = None,
        tokens: List[str] = ["A", "U", "G", "C"]
    ) -> List[str]:
        """生成多个突变序列"""
        num_sequences = len(sequences)
        # 先均分，再把余数随机分配
        base_num = num_mutation // num_sequences
        extra = num_mutation % num_sequences
        mutants_per_parent = [base_num + (1 if i < extra else 0) for i in range(num_sequences)]
        # 乱序
        random.shuffle(mutants_per_parent)

        if importance_scores_list is None:
            importance_scores_list = [None] * num_sequences

        mutation_sequences = []
        for seq, scores, n_mut in zip(sequences, importance_scores_list, mutants_per_parent):
            for _ in range(n_mut):
                # 出现重复时，最多尝试20次重新生成（防止死循环）
                retry_cnt = 20
                while retry_cnt > 0:
                    new_seq = self.mutate_single(seq, mutation_ratio, scores, tokens)
                    if new_seq not in self.all_generated_sequences:
                        # 记录已生成的序列，防止重复
                        self.all_generated_sequences.add(new_seq)
                        mutation_sequences.append(new_seq)
                        break
                    retry_cnt -= 1

        return mutation_sequences


def compute_importance_scores(seq, model, tokenizer, importance_type="gradient"):
    """计算序列的importance scores，用于指导突变"""
    # 注意：返回的scores需要与输入序列长度对应（不含cls和sep），且shape为 (1, seq_len)
    
    if importance_type == "gradient":
        gradients = None
        def capture_gradients(module, grad_input, grad_output):
            """捕获模型梯度的hook函数"""
            nonlocal gradients
            gradients = grad_output[0].detach()
        # 注册hook
        hook_handle = model.base_model.encoder.layer[-1].output.register_full_backward_hook(capture_gradients)
        
        # 准备输入
        device = next(model.parameters()).device
        inputs = tokenizer([seq], return_tensors="pt").to(device)
        # 添加了cls和sep token
        assert len(seq) == inputs['input_ids'].shape[1]-2

        # 前向和反向传播
        model.zero_grad()
        outputs = model(**inputs)
        outputs = torch.max(outputs, torch.tensor(0))
        target_score = outputs.sum()
        target_score.backward()

        # 移除hook
        hook_handle.remove()

        # 计算importance scores
        importance_scores = gradients.mean(dim=2)[:, 1:-1]

    else:
        raise ValueError(f"Unknown importance_type: {importance_type}")
    
    return importance_scores


def optimize_sequences(
    initial_sequences: List[str],
    model,
    generator: MaskedGenerator,
    tokenizer,
    reference_scores: Optional[List[float]] = None,
    iterations: int = 10,
    mutation_ratio: float = 0.1,
    num_candidates: int = 10,
    output_dir: Optional[str] = None,
    elitism_ratio: float = 0.2,
    importance_type: str = "gradient"
) -> List[str]:
    """
    RNA序列进化优化主流程。
    每轮对当前序列池进行突变，筛选得分最高的若干序列进入下一轮，最终输出优化后的序列。

    Args:
        initial_sequences: 初始候选序列列表
        model: 预测模型
        generator: MaskedGenerator实例
        tokenizer: 分词器
        reference_scores: 原始分布分数（用于可视化对比）
        iterations: 优化迭代轮数
        mutation_ratio: 每轮突变比例
        num_candidates: 每轮保留的top序列数
        output_dir: 结果保存目录

    Returns:
        List[str]: 优化后的top序列列表
    """
    model.eval()
    current_sequences = initial_sequences
    all_scores_per_iteration = []

    output_csv = os.path.join(output_dir, "output_sequences.csv")
    save_sequences([generator.template], evaluate_sequences([generator.template], model, tokenizer), "Initial", output_csv)

    # 记录上一轮的突变池和得分，用于精英保留
    last_mutated_pool = initial_sequences.copy()
    last_mutated_scores = evaluate_sequences(initial_sequences, model, tokenizer)
    
    # 历史全局top-k
    best_sequences_and_scores = []

    # 记录所有已经不能再突变的序列
    exhausted_sequences = set()

    for iter_idx in range(1, iterations+1):
        # 存储本轮所有突变子代及其得分
        mutated_pool = []
        mutated_scores = []

        # 对每个序列，计算importance scores，以此指导突变
        for seq in current_sequences:
            # 计算importance scores
            importance_scores = compute_importance_scores(seq, model, tokenizer, importance_type)

            # 生成突变子代
            mutated_seqs = generator.mutate_multiple([seq], num_candidates, mutation_ratio, importance_scores)
            mutated_pool.extend(mutated_seqs)

            # 如果该序列未生成任何突变子代，说明所有可能的突变都已生成过，跳过
            if not mutated_seqs:
                exhausted_sequences.add(seq)
                continue
            # 评估突变子代得分
            scores = evaluate_sequences(mutated_seqs, model, tokenizer)
            mutated_scores.extend(scores)

        # 精英保留机制：上一轮的top-m序列也加入本轮候选池
        # m可以自行调整
        if last_mutated_pool and last_mutated_scores:
            half_k = max(1, int(num_candidates * elitism_ratio)) # 至少保留1个
            print(f"Elitism: retaining top {half_k} sequences from previous generation")
            mutated_pool.extend(last_mutated_pool[:half_k])
            mutated_scores.extend(last_mutated_scores[:half_k])

        # 去除已无法突变的序列
        filtered_pool_scores = [(seq, score) for seq, score in zip(mutated_pool, mutated_scores) if seq not in exhausted_sequences]

        # 选出得分最高的top序列
        sorted_seqs_scores = sorted(filtered_pool_scores, key=lambda x: x[1], reverse=True)
        top_sequences = [seq for seq, _ in sorted_seqs_scores[:num_candidates]]
        top_scores = [score for _, score in sorted_seqs_scores[:num_candidates]]

        # 记录本轮的突变池和得分，用于下一轮精英保留
        last_mutated_pool = top_sequences
        last_mutated_scores = top_scores

        # 保存本轮结果
        save_sequences(top_sequences, top_scores, iter_idx, output_csv)
        current_sequences = top_sequences
        all_scores_per_iteration.append(top_scores)
        
        # 更新全局top-k
        # 先加入本轮top-k，保证不重复
        best_seq_set = set(seq for seq, _ in best_sequences_and_scores)
        for seq, score in zip(top_sequences, top_scores):
            if seq not in best_seq_set:
                best_sequences_and_scores.append((seq, score))
                best_seq_set.add(seq)
        # 选择全局top-k
        best_sequences_and_scores = sorted(best_sequences_and_scores, key=lambda x: x[1], reverse=True)[:num_candidates]
        
        # 日志
        logging.info(f"Iteration {iter_idx}/{iterations}, Top mutated sequences:")
        for idx, seq in enumerate(current_sequences):
            logging.info(f"    Sequence {idx+1}: Score {top_scores[idx]:.4f}, {seq}")

    # 可视化分布变化
    plot_score_density_curve(all_scores_per_iteration, reference_scores=reference_scores, output_dir=output_dir)
    plot_mean_score_trend(all_scores_per_iteration, output_dir=output_dir)
    
    # 保存最终的全局top-k
    final_seqs = [seq for seq, _ in best_sequences_and_scores]
    final_scores = [score for _, score in best_sequences_and_scores]
    save_sequences(final_seqs, final_scores, "Final", output_csv)

    return final_seqs


def evaluate_sequences(sequences, model, tokenizer, verbose=False):
    """评估序列的得分（具体含义取决于模型训练时的目标）"""
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        inputs = tokenizer(sequences, return_tensors="pt", padding=True).to(device)
        outputs = model(**inputs)
        scores = outputs[:, 0]
        
        if verbose:
            # 详细打印每个序列及其得分
            for seq, score in zip(sequences, scores):
                logging.info(f"Sequence: {seq}")
                logging.info(f"Predicted score: {score.item():.4f}")

        # 转为 float 列表，自动处理 batch
        return [float(s) for s in scores.cpu()]


def load_scores_from_csv(
    csv_file: str,
    seq_column_name: str,
    label_column_name: str
):
    """
    从CSV文件加载序列和标签

    Args:
        csv_file: CSV文件路径
        seq_column_name: 序列所在列名
        label_column_name: 标签所在列名

    Returns:
        sequences: 序列列表
        log_labels: 标签列表
    """
    df = pd.read_csv(csv_file)
    if seq_column_name not in df.columns or label_column_name not in df.columns:
        raise ValueError(f"CSV文件缺少指定列: {seq_column_name} 或 {label_column_name}")
    sequences = df[seq_column_name].astype(str).tolist()
    labels = df[label_column_name].astype(float).tolist()
    return sequences, labels


def main():
    args = parse_args()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    # 如果没有指定日志保存路径，则使用mode-timestamp的默认路径
    if args.output_dir == '':
        args.output_dir = os.path.join("logs", f"seqGeneration-{time.strftime('%Y%m%d-%H%M')}")
    setup_logging(args.output_dir)

    logging.info(f"Arguments: {args}")
    logging.info(f"Using device: {device}")
    logging.info(f"Random seed set to: {args.seed}")

    # 加载模型和数据
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_weights)
    base_model = AutoModel.from_pretrained(args.base_model_weights)
    model = UniRNAWithAdaptiveRegression_linearout(base_model).to(device)
    
    logging.info(f"Template sequence: {args.template}")
    # 加载训练好的模型权重
    model = load_model_weights(model, args.saved_model_path)

    # 初始化序列生成器
    generator = MaskedGenerator(args.template, allow_mutate_all_sites=args.allow_mutate_all_sites, protected_regions=args.protected_regions, device=device, selection_mode=args.selection_mode)

    # 生成初始候选序列
    initial_sequences = generator.generate_initial_candidates(args.num_candidates)
    logging.info(f"Initial candidate sequences:")
    for idx, seq in enumerate(initial_sequences):
        logging.info(f"    Sequence {idx+1}: {seq}")

    # 读取csv中的参考序列的分数，用于对比生成序列的分数的分布（非必须）
    reference_scores = None
    if args.csv_path:
        _, reference_scores = load_scores_from_csv(args.csv_path, args.seq_column_name, args.label_column_name)

    # 进行序列优化
    mutated_sequences = optimize_sequences(
        initial_sequences,
        model,
        generator,
        tokenizer,
        reference_scores=reference_scores,
        iterations=args.iterations,
        mutation_ratio=args.mutation_ratio,
        num_candidates=args.num_candidates,
        output_dir=args.output_dir,
        elitism_ratio=args.elitism_ratio,
        importance_type=args.importance_type,
    )

    # 评估最终序列
    logging.info("\nFinal sequence evaluation:")
    evaluate_sequences(mutated_sequences, model, tokenizer, verbose=True)

if __name__ == "__main__":
    main()
