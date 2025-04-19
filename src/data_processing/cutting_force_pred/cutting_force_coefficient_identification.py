# -*- coding: utf-8 -*-
"""
%%%%%%%%%%%%%%%%%%%%%%%%%
%                         版权声明                           %
%    本代码由李建辉开发     %
%%%%%%%%%%%%%%%%%%%%%%%%%

功能：求解切削力系数，采用线性回归方法
主轴转速为定值6000rpm，通过改变进给速度来实现改变每齿进给量。
"""

import numpy as np
import matplotlib.pyplot as plt

# 加载实验切削力数据(.mat文件)，包含5组不同每齿进给量下，1秒内的3向切削力数据
# 注意：实际使用时需要替换为真实的.mat文件路径
# force_data = loadmat('force.mat')

# 时间参数设置
t1 = 0  # 开始时间(s)
t2 = 1  # 结束时间(s)
r = 0.0001  # 采样频率
t = np.arange(t1, t2 + r, r)  # 时间序列

# 注意方向说明：
# 先将进给方向与参考的X正方向对应，判断测力仪方向与参考方向之间的关系
# 测力仪三向正方向与参考方向对应的要变号，相反的不用变号，因为关注点在刀具，
# 两者是作用力与反作用力之间的关系

# 本例中，由于测力仪摆放原因：
# 参考X正向为测力仪Y负向，参考Y正向为测力仪X负向，参考Z正向为测力仪Z负向
# 因此无需改变符号，只需对调X和Y

# 实际操作说明：
# 实际中应取实际测量信号，求得力的平均值
# 这里为了演示方便，用计算信号加噪声代替实测信号
# 在实际切削力求解中，应取实测信号计算力的均值
# 在切削力模型足够精确时，可以得到较精确的拟合结果
# 然后将拟合后的切削力系数作为该试验条件下的切削力系数

# 定义切削参数
b = 3  # 轴向切削深度(mm)
FT = np.array([0.025, 0.05, 0.075, 0.1, 0.125])  # 每齿进给量数组(mm/tooth)

# 定义刀具参数
Nt = 2  # 刀具齿数

# 假设已经加载了实验数据，这里用示例数据代替
# 每组切削力数据(实际应替换为真实数据)
Fx_300 = np.random.normal(100, 10, len(t))  # 示例数据
Fy_300 = np.random.normal(150, 15, len(t))
Fz_300 = np.random.normal(200, 20, len(t))

Fx_600 = np.random.normal(200, 15, len(t))
Fy_600 = np.random.normal(250, 20, len(t))
Fz_600 = np.random.normal(300, 25, len(t))

Fx_900 = np.random.normal(300, 20, len(t))
Fy_900 = np.random.normal(350, 25, len(t))
Fz_900 = np.random.normal(400, 30, len(t))

Fx_1200 = np.random.normal(400, 25, len(t))
Fy_1200 = np.random.normal(450, 30, len(t))
Fz_1200 = np.random.normal(500, 35, len(t))

Fx_1500 = np.random.normal(500, 30, len(t))
Fy_1500 = np.random.normal(550, 35, len(t))
Fz_1500 = np.random.normal(600, 40, len(t))

# 计算各组实验的平均力值
mean_Fx = np.zeros(len(FT))
mean_Fy = np.zeros(len(FT))
mean_Fz = np.zeros(len(FT))

mean_Fx[0] = np.mean(Fx_300)
mean_Fy[0] = np.mean(Fy_300)
mean_Fz[0] = np.mean(Fz_300)

mean_Fx[1] = np.mean(Fx_600)
mean_Fy[1] = np.mean(Fy_600)
mean_Fz[1] = np.mean(Fz_600)

mean_Fx[2] = np.mean(Fx_900)
mean_Fy[2] = np.mean(Fy_900)
mean_Fz[2] = np.mean(Fz_900)

mean_Fx[3] = np.mean(Fx_1200)
mean_Fy[3] = np.mean(Fy_1200)
mean_Fz[3] = np.mean(Fz_1200)

mean_Fx[4] = np.mean(Fx_1500)
mean_Fy[4] = np.mean(Fy_1500)
mean_Fz[4] = np.mean(Fz_1500)

# 绘制第一组切削力图形(fc=0.025mm)
plt.figure(1, figsize=(13.53 / 2.54, 9 / 2.54), dpi=100)  # 转换为厘米
plt.plot(t, Fx_300, "r", label="Fx")
plt.plot(t, Fy_300, "g", label="Fy")
plt.plot(t, Fz_300, "b", label="Fz")
plt.xlim(0, 0.1)
plt.ylim(-800, 1000)
plt.grid(True)
plt.legend(loc="upper left", prop={"family": "Times New Roman", "size": 14})
plt.title("切削力 (fc=0.025mm)", fontsize=14, fontname="Times New Roman")
plt.xlabel("时间 (s)", fontsize=14, fontname="Times New Roman")
plt.ylabel("力 (N)", fontsize=14, fontname="Times New Roman")
plt.tick_params(labelsize=14)

# 绘制第二组切削力图形(fc=0.05mm)
plt.figure(2, figsize=(13.53 / 2.54, 9 / 2.54), dpi=100)
plt.plot(t, Fx_600, "r", label="Fx")
plt.plot(t, Fy_600, "g", label="Fy")
plt.plot(t, Fz_600, "b", label="Fz")
plt.xlim(0, 0.1)
plt.ylim(-800, 1000)
plt.grid(True)
plt.legend(loc="upper left", prop={"family": "Times New Roman", "size": 14})
plt.title("切削力 (fc=0.05mm)", fontsize=14, fontname="Times New Roman")
plt.xlabel("时间 (s)", fontsize=14, fontname="Times New Roman")
plt.ylabel("力 (N)", fontsize=14, fontname="Times New Roman")
plt.tick_params(labelsize=14)

# 绘制第三组切削力图形(fc=0.075mm)
plt.figure(3, figsize=(13.53 / 2.54, 9 / 2.54), dpi=100)
plt.plot(t, Fx_900, "r", label="Fx")
plt.plot(t, Fy_900, "g", label="Fy")
plt.plot(t, Fz_900, "b", label="Fz")
plt.xlim(0, 0.1)
plt.ylim(-800, 1000)
plt.grid(True)
plt.legend(loc="upper left", prop={"family": "Times New Roman", "size": 14})
plt.title("切削力 (fc=0.075mm)", fontsize=14, fontname="Times New Roman")
plt.xlabel("时间 (s)", fontsize=14, fontname="Times New Roman")
plt.ylabel("力 (N)", fontsize=14, fontname="Times New Roman")
plt.tick_params(labelsize=14)

# 绘制第四组切削力图形(fc=0.1mm)
plt.figure(4, figsize=(13.53 / 2.54, 9 / 2.54), dpi=100)
plt.plot(t, Fx_1200, "r", label="Fx")
plt.plot(t, Fy_1200, "g", label="Fy")
plt.plot(t, Fz_1200, "b", label="Fz")
plt.xlim(0, 0.1)
plt.ylim(-800, 1000)
plt.grid(True)
plt.legend(loc="upper left", prop={"family": "Times New Roman", "size": 14})
plt.title("切削力 (fc=0.1mm)", fontsize=14, fontname="Times New Roman")
plt.xlabel("时间 (s)", fontsize=14, fontname="Times New Roman")
plt.ylabel("力 (N)", fontsize=14, fontname="Times New Roman")
plt.tick_params(labelsize=14)

# 绘制第五组切削力图形(fc=0.125mm)
plt.figure(5, figsize=(13.53 / 2.54, 9 / 2.54), dpi=100)
plt.plot(t, Fx_1500, "r", label="Fx")
plt.plot(t, Fy_1500, "g", label="Fy")
plt.plot(t, Fz_1500, "b", label="Fz")
plt.xlim(0, 0.1)
plt.ylim(-800, 1000)
plt.grid(True)
plt.legend(loc="upper left", prop={"family": "Times New Roman", "size": 14})
plt.title("切削力 (fc=0.125mm)", fontsize=14, fontname="Times New Roman")
plt.xlabel("时间 (s)", fontsize=14, fontname="Times New Roman")
plt.ylabel("力 (N)", fontsize=14, fontname="Times New Roman")
plt.tick_params(labelsize=14)

# 线性回归计算切削力系数
n = len(FT)  # 实验组数

# X方向力系数计算
a1x = (n * np.sum(FT * mean_Fx) - np.sum(FT) * np.sum(mean_Fx)) / (
    n * np.sum(FT**2) - np.sum(FT) ** 2
)
a0x = np.mean(mean_Fx) - a1x * np.mean(FT)

Ktc_fit = 4 * a1x / (Nt * b)  # 切向切削力系数
Kte_fit = np.pi * a0x / (Nt * b)  # 切向刃口力系数

# Y方向力系数计算
a1y = (n * np.sum(FT * mean_Fy) - np.sum(FT) * np.sum(mean_Fy)) / (
    n * np.sum(FT**2) - np.sum(FT) ** 2
)
a0y = np.mean(mean_Fy) - a1y * np.mean(FT)

Krc_fit = -4 * a1y / (Nt * b)  # 径向切削力系数
Kre_fit = -np.pi * a0y / (Nt * b)  # 径向刃口力系数

# Z方向力系数计算
a1z = (n * np.sum(FT * mean_Fz) - np.sum(FT) * np.sum(mean_Fz)) / (
    n * np.sum(FT**2) - np.sum(FT) ** 2
)
a0z = np.mean(mean_Fz) - a1z * np.mean(FT)

Kac_fit = np.pi * a1z / (Nt * b)  # 轴向切削力系数
Kae_fit = 2 * a0z / (Nt * b)  # 轴向刃口力系数

# 打印计算结果
print(f"Ktc_fit = {Ktc_fit:.4f}")
print(f"Kte_fit = {Kte_fit:.4f}")
print(f"Krc_fit = {Krc_fit:.4f}")
print(f"Kre_fit = {Kre_fit:.4f}")
print(f"Kac_fit = {Kac_fit:.4f}")
print(f"Kae_fit = {Kae_fit:.4f}")

# 绘制平均切削力与拟合线
plt.figure(6, figsize=(13.53 / 2.54, 9 / 2.54), dpi=100)
plt.plot(FT, mean_Fx, "ro", label="x方向")
plt.plot(FT, mean_Fy, "gs", label="y方向")
plt.plot(FT, mean_Fz, "b^", label="z方向")
# 绘制拟合线
plt.plot(FT, a0x + a1x * FT, "r")
plt.plot(FT, a0y + a1y * FT, "g")
plt.plot(FT, a0z + a1z * FT, "b")
plt.grid(True)
plt.legend(loc="upper left", prop={"family": "Times New Roman", "size": 14})
plt.title("平均切削力", fontsize=14)
plt.xlabel("每齿进给量 (mm/tooth)", fontsize=14, fontname="Times New Roman")
plt.ylabel("平均力 (N)", fontsize=14, fontname="Times New Roman")
plt.tick_params(labelsize=14)


# 计算决定系数(评价线性回归好坏)
def calculate_r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)


rx2 = calculate_r2(mean_Fx, a0x + a1x * FT)
ry2 = calculate_r2(mean_Fy, a0y + a1y * FT)
rz2 = calculate_r2(mean_Fz, a0z + a1z * FT)

print(f"X方向决定系数 R² = {rx2:.4f}")
print(f"Y方向决定系数 R² = {ry2:.4f}")
print(f"Z方向决定系数 R² = {rz2:.4f}")

plt.show()
