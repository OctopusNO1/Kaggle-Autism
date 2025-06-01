import os
import shutil
import random
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split

def split_image_dataset(
    input_dir, 
    output_dir, 
    train_ratio=0.7,    # 70%训练, 15%验证, 15%测试
    val_ratio=0.15, 
    test_ratio=0.15,
    random_state=42,
    copy_files=True,
    create_plots=True
):
    """
    切分按文件夹组织的图像数据集为训练集、验证集和测试集
    
    参数:
    input_dir (str): 输入数据集目录路径（按类别组织）
    output_dir (str): 输出目录路径
    train_ratio (float): 训练集比例 (默认0.6)
    val_ratio (float): 验证集比例 (默认0.2)
    test_ratio (float): 测试集比例 (默认0.2)
    random_state (int): 随机种子 (确保可复现性)
    copy_files (bool): 是否复制文件 (True) 还是移动文件 (False)
    create_plots (bool): 是否创建数据分布可视化
    
    返回:
    dict: 包含切分统计信息的字典
    """
    # 验证比例总和为1
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 0.01:
        raise ValueError(f"比例总和应为1.0，当前为: {total_ratio}")
    
    # 创建输出目录
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    test_dir = os.path.join(output_dir, 'test')
    
    for d in [output_dir, train_dir, val_dir, test_dir]:
        os.makedirs(d, exist_ok=True)
    
    # 获取所有类别
    classes = [d for d in os.listdir(input_dir) 
               if os.path.isdir(os.path.join(input_dir, d))]
    
    # 初始化统计信息
    class_stats = {}
    total_images = 0
    class_distribution = {}
    
    # 处理每个类别
    for cls in classes:
        # 创建输出子目录
        for d in [train_dir, val_dir, test_dir]:
            os.makedirs(os.path.join(d, cls), exist_ok=True)
        
        # 获取类别下所有图像
        cls_dir = os.path.join(input_dir, cls)
        images = [f for f in os.listdir(cls_dir) 
                 if os.path.isfile(os.path.join(cls_dir, f))]
        
        # 记录类别分布
        class_distribution[cls] = len(images)
        total_images += len(images)
        
        # 切分数据集
        if len(images) < 10:
            print(f"警告: 类别 '{cls}' 只有 {len(images)} 个样本，切分可能不稳定")
        
        # 首先切分出训练集
        train_val, test = train_test_split(
            images, 
            test_size=test_ratio, 
            random_state=random_state
        )
        
        # 然后从剩余数据中切分出验证集
        train, val = train_test_split(
            train_val, 
            test_size=val_ratio/(1-test_ratio), 
            random_state=random_state
        )
        
        # 记录统计信息
        class_stats[cls] = {
            'total': len(images),
            'train': len(train),
            'val': len(val),
            'test': len(test)
        }
        
        # 复制/移动文件到对应目录
        copy_func = shutil.copy2 if copy_files else shutil.move
        
        # 训练集
        for img in train:
            src = os.path.join(cls_dir, img)
            dst = os.path.join(train_dir, cls, img)
            copy_func(src, dst)
            
        # 验证集
        for img in val:
            src = os.path.join(cls_dir, img)
            dst = os.path.join(val_dir, cls, img)
            copy_func(src, dst)
            
        # 测试集
        for img in test:
            src = os.path.join(cls_dir, img)
            dst = os.path.join(test_dir, cls, img)
            copy_func(src, dst)
    
    # 创建可视化
    if create_plots:
        plot_dataset_distribution(class_stats, output_dir)
    
    # 打印摘要
    print("\n" + "="*50)
    print(f"数据集切分完成! 结果保存在: {output_dir}")
    print(f"总样本数: {total_images}")
    print(f"类别数: {len(classes)}")
    print(f"训练集: {len(list(Path(train_dir).rglob('*.*')))} 张图片")
    print(f"验证集: {len(list(Path(val_dir).rglob('*.*')))} 张图片")
    print(f"测试集: {len(list(Path(test_dir).rglob('*.*')))} 张图片")
    print("="*50 + "\n")
    
    # 打印每个类别的分布
    print(f"{'类别':<20} {'总数':<6} {'训练集':<6} {'验证集':<6} {'测试集':<6}")
    for cls, stats in class_stats.items():
        print(f"{cls:<20} {stats['total']:<6} {stats['train']:<6} {stats['val']:<6} {stats['test']:<6}")
    
    return {
        'total_images': total_images,
        'num_classes': len(classes),
        'class_stats': class_stats,
        'output_dir': output_dir
    }

def plot_dataset_distribution(class_stats, output_dir):
    """可视化数据集分布"""
    # 准备数据
    classes = list(class_stats.keys())
    train_counts = [stats['train'] for stats in class_stats.values()]
    val_counts = [stats['val'] for stats in class_stats.values()]
    test_counts = [stats['test'] for stats in class_stats.values()]
    total_counts = [stats['total'] for stats in class_stats.values()]
    
    # 创建图表
    plt.figure(figsize=(14, 10))
    
    # 1. 类别分布饼图
    plt.subplot(2, 2, 1)
    plt.pie(total_counts, labels=classes, autopct='%1.1f%%', startangle=90)
    plt.title('整体类别分布')
    plt.axis('equal')
    
    # 2. 训练/验证/测试集分布
    plt.subplot(2, 2, 2)
    bar_width = 0.25
    index = np.arange(len(classes))
    
    plt.bar(index, train_counts, bar_width, label='训练集')
    plt.bar(index + bar_width, val_counts, bar_width, label='验证集')
    plt.bar(index + bar_width*2, test_counts, bar_width, label='测试集')
    
    plt.xlabel('类别')
    plt.ylabel('样本数量')
    plt.title('各类别在训练/验证/测试集中的分布')
    plt.xticks(index + bar_width, classes, rotation=45)
    plt.legend()
    plt.tight_layout()
    
    # 3. 各类别内部划分比例
    plt.subplot(2, 1, 2)
    for i, cls in enumerate(classes):
        stats = class_stats[cls]
        total = stats['total']
        plt.bar(i, stats['train']/total, color='blue', label='训练集' if i==0 else "")
        plt.bar(i, stats['val']/total, bottom=stats['train']/total, color='orange', label='验证集' if i==0 else "")
        plt.bar(i, stats['test']/total, bottom=(stats['train']+stats['val'])/total, color='green', label='测试集' if i==0 else "")
    
    plt.xlabel('类别')
    plt.ylabel('比例')
    plt.title('各类别内部划分比例')
    plt.xticks(range(len(classes)), classes, rotation=45)
    plt.ylim(0, 1)
    plt.legend(loc='upper right')
    plt.tight_layout()
    
    # 保存图表
    plot_path = os.path.join(output_dir, 'dataset_distribution.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"数据集分布图已保存至: {plot_path}")

if __name__ == "__main__":
    print("start")
    # ===== 配置参数 =====
    INPUT_DIR = r"C:\Users\zhong\Desktop\AIMed\Kaggle-Autism-master\Kaggle-Autism-master\cleanData\images"   # 替换为您的数据集路径
    OUTPUT_DIR = r"C:\Users\zhong\Desktop\AIMed\Kaggle-Autism-master\Kaggle-Autism-master\cleanData" # 替换为输出路径

    # 切分比例 (总和应为1.0)
    TRAIN_RATIO = 0.6  # 60% 训练集
    VAL_RATIO = 0.2    # 20% 验证集
    TEST_RATIO = 0.2   # 20% 测试集
    
    RANDOM_STATE = 42  # 随机种子 (确保结果可复现)
    COPY_FILES = True  # True=复制文件, False=移动文件
    
    # ===== 执行切分 =====
    result = split_image_dataset(
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO,
        random_state=RANDOM_STATE,
        copy_files=COPY_FILES
    )