import os

# 配置参数
sequence_length = 60
feature_num = '158+39'

# 训练轮数：可用环境变量 BDC_EPOCHS 覆盖，便于冒烟测试与正式训练一键切换
# 冒烟测试（快速跑通整条链路）：  BDC_EPOCHS=2 sh train.sh
# 正式训练（用默认 50 轮）：      sh train.sh
# Windows 写法： set BDC_EPOCHS=2 && python code/src/train.py
num_epochs = int(os.environ.get('BDC_EPOCHS', 50))

config = {
    'sequence_length': sequence_length,   # 使用过去60个交易日的数据（排序任务可以用稍短的序列）
    'd_model': 256,          # Transformer输入维度
    'nhead': 4,             # 注意力头数量
    'num_layers': 3,        # Transformer层数
    'dim_feedforward': 512, # 前馈网络维度
    'batch_size': 4,        # 排序任务batch_size可以小一些，因为每个batch包含更多股票
    'num_epochs': num_epochs,  # 默认50；可用 BDC_EPOCHS 环境变量覆盖（见上）
    'learning_rate': 1e-5,  # 稍微降低学习率
    'dropout': 0.1,
    'feature_num': feature_num,
    'max_grad_norm': 5.0,

    'pairwise_weight': 1, # 配对损失权重
    'base_weight': 1.0, # 非top-k样本权重
    'top5_weight': 2.0, # top-5样本权重（应大于base_weight）

    'output_dir': f'./model/{sequence_length}_{feature_num}',
    'data_path': './data',
}