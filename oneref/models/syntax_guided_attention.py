"""
Syntax-Guided Attention Module for OneRef
Implementation of Soft Syntax Bias for Structure-aware Finetuning

Author: Assistant
Date: 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import spacy
from typing import Optional, Tuple
from torchscale.component.multihead_attention import MultiheadAttention


class SyntaxDistanceCalculator:
    """计算句法依赖图距离矩阵的工具类"""
    
    def __init__(self, nlp_model=None):
        if nlp_model is None:
            try:
                self.nlp = spacy.load("en_core_web_md")
            except OSError:
                try:
                    self.nlp = spacy.load("en_core_web_sm")
                    print("Warning: Using en_core_web_sm instead of en_core_web_md")
                except OSError:
                    raise ImportError("Please install spaCy model: python -m spacy download en_core_web_md")
        else:
            self.nlp = nlp_model
    
    def get_dependency_distance_matrix(self, text: str, max_len: int = 64) -> torch.Tensor:
        """
        计算句法依赖距离矩阵
        
        Args:
            text: 输入文本
            max_len: 最大序列长度
            
        Returns:
            distance_matrix: 形状为 (max_len, max_len) 的距离矩阵
        """
        doc = self.nlp(text)
        seq_len = min(len(doc), max_len)
        
        # 初始化距离矩阵
        distance_matrix = torch.full((max_len, max_len), float('inf'), dtype=torch.float32)
        
        # 计算每个词到其他词的最短句法距离
        for i in range(seq_len):
            for j in range(seq_len):
                if i == j:
                    distance_matrix[i, j] = 0
                else:
                    # 计算从token i到token j的句法距离
                    distance = self._compute_syntactic_distance(doc[i], doc[j])
                    distance_matrix[i, j] = distance
        
        # 对于超出文本长度的位置，设置为大值
        for i in range(seq_len, max_len):
            for j in range(max_len):
                distance_matrix[i, j] = float('inf')
                distance_matrix[j, i] = float('inf')
        
        return distance_matrix
    
    def _compute_syntactic_distance(self, token1, token2) -> float:
        """
        计算两个token之间的句法距离
        使用依赖树中的最短路径距离
        """
        # 如果是同一个token，距离为0
        if token1.i == token2.i:
            return 0.0
        
        # 获取从token1到根节点的路径
        path1 = self._get_path_to_root(token1)
        # 获取从token2到根节点的路径
        path2 = self._get_path_to_root(token2)
        
        # 找到最近公共祖先
        lca = None
        for node1 in path1:
            if node1 in path2:
                lca = node1
                break
        
        if lca is None:
            # 如果没有公共祖先（理论上不应该发生），返回较大距离
            return 10.0
        
        # 计算距离：从token1到LCA的距离 + 从token2到LCA的距离
        dist1 = path1.index(lca)
        dist2 = path2.index(lca)
        
        return float(dist1 + dist2)
    
    def _get_path_to_root(self, token):
        """获取从token到根节点的路径"""
        path = [token]
        current = token.head
        while current != current:  # 直到根节点
            path.append(current)
            current = current.head
            if len(path) > 20:  # 防止无限循环
                break
        return path


class SyntaxGuidedMultiheadAttention(nn.Module):
    """
    句法引导的多头注意力机制
    Attention = Softmax(QK^T/√d + λ * G)
    其中G是基于句法树生成的图距离矩阵
    """
    
    def __init__(self, args, embed_dim, num_heads, dropout=0.0, 
                 syntax_bias_lambda=0.1, enable_syntax_bias=True):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.syntax_bias_lambda = syntax_bias_lambda
        self.enable_syntax_bias = enable_syntax_bias
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        # 原始的MultiheadAttention组件
        self.base_attention = MultiheadAttention(
            args=args,
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            self_attention=True,
        )
        
        # 句法距离计算器
        self.syntax_calculator = SyntaxDistanceCalculator()
        
        # 用于缓存句法矩阵的字典
        self.syntax_cache = {}
        
    def forward(self, query, key, value, key_padding_mask=None, attn_mask=None, 
                text_inputs=None, incremental_state=None, **kwargs):
        """
        前向传播
        
        Args:
            query, key, value: 形状为 (seq_len, batch_size, embed_dim)
            key_padding_mask: 键的填充掩码
            attn_mask: 注意力掩码
            text_inputs: 原始文本输入，用于计算句法距离
            incremental_state: 增量状态（用于生成）
        """
        
        batch_size = query.size(1)
        seq_len = query.size(0)
        
        # 如果启用句法偏置且有文本输入
        if self.enable_syntax_bias and text_inputs is not None:
            # 计算句法偏置
            syntax_bias = self._compute_syntax_bias(
                text_inputs, seq_len, batch_size, query.device
            )
            
            # 重新计算带有句法偏置的注意力
            attn_output = self._compute_syntax_biased_attention(
                query, key, value, syntax_bias, key_padding_mask, attn_mask
            )
            
            # 返回修改后的注意力和原始权重（用于兼容性）
            return attn_output, None
        
        # 否则使用原始attention
        attn_output, attn_weights = self.base_attention(
            query=query,
            key=key, 
            value=value,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            incremental_state=incremental_state,
            **kwargs
        )
        
        return attn_output, attn_weights
    
    def _compute_syntax_bias(self, text_inputs, seq_len, batch_size, device):
        """计算句法偏置矩阵"""
        syntax_bias_list = []
        
        for text in text_inputs:
            # 使用缓存避免重复计算
            if text in self.syntax_cache:
                distance_matrix = self.syntax_cache[text]
            else:
                distance_matrix = self.syntax_calculator.get_dependency_distance_matrix(
                    text, max_len=seq_len
                )
                self.syntax_cache[text] = distance_matrix
            
            # 将距离转换为偏置（距离越小，偏置越大）
            # 使用负指数函数：bias = -exp(distance/σ)
            sigma = 2.0  # 控制衰减速度
            bias_matrix = -torch.exp(distance_matrix / sigma)
            
            # 对于无穷大距离（填充位置），设置为一个很小的值
            bias_matrix[torch.isinf(bias_matrix)] = -10.0
            
            syntax_bias_list.append(bias_matrix)
        
        # 堆叠成批次
        syntax_bias = torch.stack(syntax_bias_list).to(device)  # (batch_size, seq_len, seq_len)
        
        return syntax_bias
    
    def _compute_syntax_biased_attention(self, query, key, value, syntax_bias, 
                                     key_padding_mask, attn_mask):
        """计算带有句法偏置的注意力"""
        
        batch_size, seq_len, embed_dim = query.shape
        query = query.transpose(0, 1)  # (batch_size, seq_len, embed_dim)
        key = key.transpose(0, 1)
        value = value.transpose(0, 1)
        
        # 重塑为多头
        q = query.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = key.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = value.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 计算原始attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # 添加句法偏置
        syntax_bias_expanded = syntax_bias.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        scores = scores + self.syntax_bias_lambda * syntax_bias_expanded
        
        # 应用注意力掩码
        if attn_mask is not None:
            scores = scores + attn_mask
        
        # 应用键填充掩码
        if key_padding_mask is not None:
            scores = scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf')
            )
        
        # 计算注意力权重
        attn_weights = F.softmax(scores, dim=-1)
        
        # 应用注意力权重到值
        attn_output = torch.matmul(attn_weights, v)
        
        # 重塑输出
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, embed_dim
        )
        
        return attn_output.transpose(0, 1)  # 转回 (seq_len, batch_size, embed_dim)


def create_syntax_guided_attention(args, embed_dim, num_heads, dropout=0.0,
                               syntax_bias_lambda=0.1, enable_syntax_bias=True):
    """创建句法引导的注意力模块"""
    return SyntaxGuidedMultiheadAttention(
        args=args,
        embed_dim=embed_dim,
        num_heads=num_heads,
        dropout=dropout,
        syntax_bias_lambda=syntax_bias_lambda,
        enable_syntax_bias=enable_syntax_bias
    )
