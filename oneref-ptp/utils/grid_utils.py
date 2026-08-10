"""
Grid ID计算工具函数
用于PTP辅助监督任务：将归一化的bbox转换为Grid ID
"""
import torch
import numpy as np


def compute_grid_id(bbox_xywh, grid_size=16):
    """
    将归一化的bbox转换为Grid ID
    
    Args:
        bbox_xywh: 归一化的bbox，格式为 [x, y, w, h] (0-1范围)
                   或 [cx, cy, w, h] (中心点格式)
        grid_size: 网格大小，如16表示16x16=256个网格
    
    Returns:
        grid_id: 整数，范围 [0, grid_size²-1]
    """
    # 转换为numpy数组（如果是tensor）
    if isinstance(bbox_xywh, torch.Tensor):
        bbox_xywh = bbox_xywh.cpu().numpy()
    elif isinstance(bbox_xywh, np.ndarray):
        bbox_xywh = bbox_xywh.copy()
    else:
        bbox_xywh = np.array(bbox_xywh)
    
    # 确保是1D数组
    if bbox_xywh.ndim > 1:
        bbox_xywh = bbox_xywh.flatten()
    
    # 提取中心点坐标
    # OneRef中bbox格式是 [x, y, w, h]（归一化到0-1）
    if len(bbox_xywh) == 4:
        x, y, w, h = bbox_xywh[0], bbox_xywh[1], bbox_xywh[2], bbox_xywh[3]
        # 计算中心点（归一化坐标）
        cx = x + w / 2.0
        cy = y + h / 2.0
    else:
        # 如果已经是中心点格式 [cx, cy, w, h]
        cx, cy = bbox_xywh[0], bbox_xywh[1]
    
    # 限制在 [0, 1] 范围内
    cx = np.clip(cx, 0.0, 1.0)
    cy = np.clip(cy, 0.0, 1.0)
    
    # 计算网格坐标
    # 注意：当cx=1.0时，grid_x应该是grid_size-1，所以使用floor而不是int
    grid_x = int(np.floor(cx * grid_size))
    grid_y = int(np.floor(cy * grid_size))
    
    # 确保在有效范围内 [0, grid_size-1]
    grid_x = min(max(grid_x, 0), grid_size - 1)
    grid_y = min(max(grid_y, 0), grid_size - 1)
    
    # 转换为线性ID: row * grid_size + col
    # 注意：grid_y是行（从上到下），grid_x是列（从左到右）
    grid_id = grid_y * grid_size + grid_x
    
    return grid_id


def compute_grid_id_batch(bboxes_xywh, grid_size=16):
    """
    批量计算Grid ID
    
    Args:
        bboxes_xywh: (B, 4) 归一化的bboxes，格式为 [x, y, w, h]
        grid_size: 网格大小
    
    Returns:
        grid_ids: (B,) 整数tensor
    """
    if isinstance(bboxes_xywh, torch.Tensor):
        bboxes_xywh = bboxes_xywh.cpu().numpy()
    
    batch_size = bboxes_xywh.shape[0]
    grid_ids = []
    
    for i in range(batch_size):
        grid_id = compute_grid_id(bboxes_xywh[i], grid_size)
        grid_ids.append(grid_id)
    
    return torch.tensor(grid_ids, dtype=torch.long)


def test_grid_id_computation():
    """测试Grid ID计算正确性"""
    print("=== 测试Grid ID计算 ===")
    
    # 测试用例1: 中心在图像中心 (0.5, 0.5)
    bbox1 = np.array([0.4, 0.4, 0.2, 0.2])  # 中心在(0.5, 0.5)
    grid_id1 = compute_grid_id(bbox1, grid_size=16)
    expected1 = 8 * 16 + 8  # 第8行第8列 = 136
    print(f"中心bbox [0.4, 0.4, 0.2, 0.2] -> Grid ID: {grid_id1} (期望: {expected1})")
    
    # 测试用例2: 左上角 (0.1, 0.1)
    bbox2 = np.array([0.05, 0.05, 0.1, 0.1])  # 中心在(0.1, 0.1)
    grid_id2 = compute_grid_id(bbox2, grid_size=16)
    expected2 = 1 * 16 + 1  # 第1行第1列 = 17
    print(f"左上角bbox [0.05, 0.05, 0.1, 0.1] -> Grid ID: {grid_id2} (期望: {expected2})")
    
    # 测试用例3: 右下角 (0.9, 0.9)
    bbox3 = np.array([0.85, 0.85, 0.1, 0.1])  # 中心在(0.9, 0.9)
    grid_id3 = compute_grid_id(bbox3, grid_size=16)
    expected3 = 14 * 16 + 14  # 第14行第14列 = 238
    print(f"右下角bbox [0.85, 0.85, 0.1, 0.1] -> Grid ID: {grid_id3} (期望: {expected3})")
    
    # 测试用例4: 边界情况 (cx=1.0)
    bbox4 = np.array([0.95, 0.95, 0.1, 0.1])  # 中心接近(1.0, 1.0)
    grid_id4 = compute_grid_id(bbox4, grid_size=16)
    expected4 = 15 * 16 + 15  # 第15行第15列 = 255
    print(f"边界bbox [0.95, 0.95, 0.1, 0.1] -> Grid ID: {grid_id4} (期望: {expected4})")
    
    print("测试完成！")


if __name__ == "__main__":
    test_grid_id_computation()

