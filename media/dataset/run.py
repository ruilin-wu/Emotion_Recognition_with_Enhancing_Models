import matplotlib.pyplot as plt

# 文件路径
path_prefix = "G:\\master\\Year2_S2\\ELEC5305\\projectcode\\Emotion-Classification-Ravdess\\media\\"
files = {
    "Basic_CNN": "val_accuracy_basic_CNN.txt",
    "Improved CNN:": "val_accuracy_Improved_CNN.txt",
    "GRU": "val_accuracy_GRU.txt",
    "LSTM": "val_accuracy_LSTM.txt"
}

# 读取文件内容的函数
def read_accuracy_from_file(filename):
    accuracies = []
    with open(filename, 'r') as file:
        for line in file.readlines():
            line = line.strip()
            if line:  # 检查行是否为空
                try:
                    accuracies.append(float(line))
                except ValueError:
                    print(f"Warning: Invalid value '{line}' in {filename}. Skipping.")
    return accuracies

# 创建一个新的图形
plt.figure(figsize=(10, 6))

# 从文件中读取数据并绘图
for label, file in files.items():
    accuracies = read_accuracy_from_file(path_prefix + file)
    plt.plot(accuracies, label=label)

# 设置图标标题和坐标轴标签，并指定fontsize
plt.title("Validation Accuracy Over Epochs", fontsize=20)
plt.xlabel("Epoch", fontsize=18)
plt.ylabel("Accuracy (%)", fontsize=18)
plt.xticks(fontsize=18)  # 设置x轴刻度的字体大小
plt.yticks(fontsize=18)  # 设置y轴刻度的字体大小
plt.legend(fontsize=18)
plt.grid(True)

# 显示图形
plt.show()
