import numpy as np
from collections import defaultdict

def calculate_iou(box1, box2):
    """
    计算两个边界框的IoU（交并比）
    box1, box2: [x1, y1, x2, y2] 格式的边界框
    """
    # 计算交集区域
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # 如果没有交集，返回0
    if x1 >= x2 or y1 >= y2:
        return 0.0
    
    # 计算交集面积
    intersection = (x2 - x1) * (y2 - y1)
    
    # 计算两个框的面积
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # 计算并集面积
    union = area1 + area2 - intersection
    
    # 计算IoU
    iou = intersection / union if union > 0 else 0.0
    return iou

def load_your_data():
    """
    加载你的数据集
    返回格式：
    - all_gts: dict, {image_id: [[x1, y1, x2, y2], ...]}
    - all_preds: dict, {image_id: [[x1, y1, x2, y2, confidence], ...]}
    """
    # 这里是示例数据，你需要根据实际情况替换
    all_gts = {
        "image1": [[100, 100, 200, 200], [300, 300, 400, 400]],
        "image2": [[50, 50, 150, 150]]
    }
    
    all_preds = {
        "image1": [[95, 95, 205, 205, 0.8], [295, 295, 405, 405, 0.9], [500, 500, 600, 600, 0.6]],
        "image2": [[45, 45, 155, 155, 0.7]]
    }
    
    return all_gts, all_preds

def calculate_metrics_at_confidence_threshold(all_gt_boxes, all_pred_boxes, conf_threshold=0.7, iou_threshold=0.5):
    """
    在固定的置信度和IoU阈值下，计算精确率、召回率和F1分数。
    """
    # 1. 过滤掉低置信度的预测
    filtered_preds = defaultdict(list)
    for image_id, preds in all_pred_boxes.items():
        for pred in preds:
            if pred[4] >= conf_threshold:
                filtered_preds[image_id].append(pred)

    num_preds_after_filter = sum(len(p) for p in filtered_preds.values())
    total_num_gts = sum(len(g) for g in all_gt_boxes.values())
    
    # 如果过滤后没有预测或数据集中没有GT，则无法计算
    if num_preds_after_filter == 0 or total_num_gts == 0:
        return 0.0, 0.0, 0.0 # P, R, F1

    # 2. 匹配并计算总TP
    gt_matched_map = {}
    total_tp = 0
    
    for image_id, preds in filtered_preds.items():
        gt_boxes_on_image = all_gt_boxes.get(image_id, [])
        if not gt_boxes_on_image:
            continue
            
        # 为每个图像初始化匹配状态
        if image_id not in gt_matched_map:
            gt_matched_map[image_id] = np.zeros(len(gt_boxes_on_image), dtype=bool)
            
        preds.sort(key=lambda x: x[4], reverse=True) # 内部排序，防止一个高IoU低分的被一个低IoU高分的抢先

        for pred_box in preds:
            best_iou = 0.0
            best_gt_idx = -1
            for j, gt_box in enumerate(gt_boxes_on_image):
                if gt_matched_map[image_id][j]:
                    continue
                iou = calculate_iou(pred_box[:4], gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j
            
            if best_iou >= iou_threshold:
                total_tp += 1
                gt_matched_map[image_id][best_gt_idx] = True

    # 3. 计算指标
    precision = total_tp / num_preds_after_filter if num_preds_after_filter > 0 else 0.0
    recall = total_tp / total_num_gts if total_num_gts > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return precision, recall, f1_score

# --- 在主程序中调用 ---
if __name__ == '__main__':
    # ... (之前的代码) ...
    all_gts, all_preds = load_your_data()
    
    print("\n--- 在固定置信度阈值下的性能 ---")
    conf_thresh = 0.7
    p, r, f1 = calculate_metrics_at_confidence_threshold(all_gts, all_preds, conf_threshold=conf_thresh, iou_threshold=0.5)
    print(f"置信度阈值 = {conf_thresh}, IoU阈值 = 0.5")
    print(f"  - Precision: {p:.4f}")
    print(f"  - Recall:    {r:.4f}")
    print(f"  - F1 Score:  {f1:.4f}")