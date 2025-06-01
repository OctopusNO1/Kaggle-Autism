import os
import pandas as pd

# 1. 定义数据集根目录和输出文件
dataset_root = r"C:\Users\zhong\Desktop\AIMed\Kaggle-Autism-master\Kaggle-Autism-master\cleanData"  # 替换为你的数据集根目录
output_train_csv = r"C:\Users\zhong\Desktop\AIMed\Kaggle-Autism-master\Kaggle-Autism-master\cleanData\train.csv"
output_val_csv = r"C:\Users\zhong\Desktop\AIMed\Kaggle-Autism-master\Kaggle-Autism-master\cleanData\val.csv"
output_test_csv = r"C:\Users\zhong\Desktop\AIMed\Kaggle-Autism-master\Kaggle-Autism-master\cleanData\test.csv"

# 2. 收集图像路径和标签（按 split 处理）
def collect_data(split_dir):
    image_paths = []
    labels = []
    for label in os.listdir(os.path.join(dataset_root, split_dir)):
        class_dir = os.path.join(dataset_root, split_dir, label)
        if not os.path.isdir(class_dir):  # 跳过非文件夹
            continue
        for image_file in os.listdir(class_dir):
            image_path = os.path.join(split_dir, label, image_file)  # 相对路径
            image_paths.append(image_path)
            labels.append(label)
    return pd.DataFrame({"image": image_paths, "label": labels})

# 3. 生成 CSV 文件
train_df = collect_data("train")
val_df = collect_data("val")
test_df = collect_data("test")

train_df.to_csv(output_train_csv, index=False)
val_df.to_csv(output_val_csv, index=False)
test_df.to_csv(output_test_csv, index=False)

print(f"训练集已保存至 {output_train_csv}")
print(f"验证集已保存至 {output_val_csv}")
print(f"测试集已保存至 {output_test_csv}")