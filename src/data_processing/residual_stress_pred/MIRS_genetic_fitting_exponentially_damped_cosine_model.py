import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random


def plot_fitting_curve(best_individual, stress_data):
    """
    绘制拟合曲线与原始数据的对比图

    参数:
        best_individual: 最优个体，包含四个参数的列表 [p1, p2, p3, p4]
        stress_data: 原始数据，二维数组，第一列为深度，第二列为残余应力
    """
    p1, p2, p3, p4 = best_individual

    # 提取原始数据点
    x_vals = stress_data[:, 0]
    y_vals = stress_data[:, 1]

    # 生成拟合曲线的x值（0到1，步长0.005）
    fitted_x_vals = np.arange(0, 1, 0.005)

    # 计算拟合曲线的y值
    exponent = (p2 / np.sqrt(1 - p2 ** 2)) * p3 * fitted_x_vals
    fitted_y_vals = p1 * np.exp(-exponent) * np.cos(p3 * fitted_x_vals + p4)

    # 绘制原始数据点
    plt.scatter(x_vals, y_vals, label='原始数据')

    # 绘制拟合曲线
    plt.plot(fitted_x_vals, fitted_y_vals, 'r', linewidth=2, label='拟合曲线')

    # 添加标签和图例
    plt.xlabel('深度')
    plt.ylabel('残余应力')
    plt.title('拟合曲线与原始数据对比')
    plt.legend()
    plt.show()


def exponentially_damped_cosine_model(p1, p2, p3, p4, x):
    """
    指数衰减余弦模型

    参数:
        p1: 振幅系数
        p2: 阻尼系数（0.5-0.9）
        p3: 频率系数（0-12）
        p4: 相位系数（0-100）
        x: 输入深度值
    """
    # 计算指数衰减项
    damping_term = (p2 / np.sqrt(1 - p2 ** 2)) * p3 * x
    # 计算余弦项
    cosine_term = np.cos(p3 * x + p4)
    return p1 * np.exp(-damping_term) * cosine_term


def fitness(individual, stress_data):
    """
    计算个体的适应度（均方误差的倒数）

    参数:
        individual: 个体，包含四个参数的列表 [p1, p2, p3, p4]
        stress_data: 原始数据，二维数组，第一列为深度，第二列为残余应力
    """
    p1, p2, p3, p4 = individual
    mse = 0.0

    # 遍历所有数据点计算均方误差
    for i in range(stress_data.shape[0]):
        x = stress_data[i, 0]
        y_true = stress_data[i, 1]
        y_pred = exponentially_damped_cosine_model(p1, p2, p3, p4, x)
        mse += (y_true - y_pred) ** 2

    mse /= stress_data.shape[0]  # 计算平均均方误差

    # 防止除零错误，当mse极小时返回一个大的适应度值
    if mse < 1e-10:
        return 1e10
    return 1.0 / mse


def initialize_population(pop_size, gene_length):
    """
    初始化种群

    参数:
        pop_size: 种群大小
        gene_length: p1的最大值（影响初始范围）

    返回:
        population: 二维数组，每行代表一个个体 [p1, p2, p3, p4]
    """
    population = []
    for _ in range(pop_size):
        p1 = random.uniform(0, gene_length)  # 0~1000
        p2 = random.uniform(0, 1)  # 0~1（后续变异会限制到0.5-0.9）
        p3 = random.uniform(0, 15)  # 0~15（后续变异会限制到0-12）
        p4 = random.uniform(0, 10)  # 0~10（后续变异会限制到0-100）
        population.append([p1, p2, p3, p4])
    return np.array(population)


def select(population, fitness_scores):
    """
    轮盘赌选择个体

    参数:
        population: 当前种群
        fitness_scores: 适应度分数列表

    返回:
        selected_individual: 被选中的个体
    """
    total_fitness = np.sum(fitness_scores)

    # 生成累积概率分布
    probs = fitness_scores / total_fitness
    cumulative_probs = np.cumsum(probs)

    # 生成随机数选择个体
    r = random.random()
    for i in range(len(cumulative_probs)):
        if r <= cumulative_probs[i]:
            return population[i]
    return population[-1]  # 保底返回最后一个个体


def crossover(parent1, parent2):
    """
    交叉操作（算术交叉）

    参数:
        parent1: 父代1
        parent2: 父代2

    返回:
        child1, child2: 两个子代
    """
    alpha = random.random()
    child1 = alpha * parent1 + (1 - alpha) * parent2
    child2 = (1 - alpha) * parent1 + alpha * parent2
    return child1, child2


def mutate(individual, mutation_rate):
    """
    变异操作（高斯变异）

    参数:
        individual: 待变异的个体
        mutation_rate: 变异概率

    返回:
        mutated_individual: 变异后的个体
    """
    mutated = individual.copy()
    for i in range(len(mutated)):
        if random.random() < mutation_rate:
            # 添加高斯噪声
            mutated[i] += random.gauss(0, 0.1)

            # 对每个参数进行范围限制
            if i == 1:  # p2: 0.5-0.9
                mutated[i] = np.clip(mutated[i], 0.5, 0.9)
            elif i == 2:  # p3: 0-12
                mutated[i] = np.clip(mutated[i], 0, 12.0)
            elif i == 3:  # p4: 0-100
                mutated[i] = np.clip(mutated[i], 0, 100.0)
            else:  # p1: 0-1000
                mutated[i] = np.clip(mutated[i], 0, 1000.0)
    return mutated


def genetic_algorithm(pop_size, gene_length, generations, mutation_rate, stress_data):
    """
    遗传算法主函数

    参数:
        pop_size: 种群大小
        gene_length: p1的最大初始值
        generations: 迭代代数
        mutation_rate: 变异率
        stress_data: 原始数据

    返回:
        best_individual: 找到的最优个体
    """
    # 初始化种群
    population = initialize_population(pop_size, gene_length)

    # 记录每代的最佳适应度
    best_fitness_history = []

    # 迭代优化
    for gen in range(generations):
        # 计算适应度
        fitness_scores = np.array([fitness(ind, stress_data) for ind in population])

        # 记录本代最佳个体
        best_idx = np.argmax(fitness_scores)
        current_best = population[best_idx]
        best_fitness = fitness_scores[best_idx]
        best_fitness_history.append(best_fitness)

        # 每100代打印进度
        if gen % 100 == 0:
            print(f"第 {gen} 代: 最佳适应度 = {best_fitness:.4f}")
            print(f"当前最佳个体: {current_best}")

        # 生成新一代种群
        new_population = []
        for _ in range(pop_size // 2):  # 每次生成2个后代
            # 选择父代
            parent1 = select(population, fitness_scores)
            parent2 = select(population, fitness_scores)

            # 交叉
            child1, child2 = crossover(parent1, parent2)

            # 变异
            child1 = mutate(child1, mutation_rate)
            child2 = mutate(child2, mutation_rate)

            new_population.extend([child1, child2])

        # 更新种群（确保种群大小不变）
        population = np.array(new_population)[:pop_size]

    # 最终选择最佳个体
    fitness_scores = np.array([fitness(ind, stress_data) for ind in population])
    best_idx = np.argmax(fitness_scores)
    return population[best_idx]


def main():
    # 读取数据
    file_path = r'E:/MyResearch/Papers/一种大型复杂薄壁零件多工步铣削加工变形有限元预测方法/论文数据及代码分析/端面铣削/VB0/UTF8_ap04ae08fz010.csv'
    stress_data = pd.read_csv(file_path).values  # 转换为numpy数组

    # 设置遗传算法参数
    pop_size = 500  # 种群大小
    gene_length = 1000  # p1的最大初始值
    generations = 200  # 迭代代数
    mutation_rate = 0.01  # 变异率

    # 运行遗传算法
    best_individual = genetic_algorithm(pop_size, gene_length, generations, mutation_rate, stress_data)
    print("最终最佳个体参数:")
    print(
        f"p1={best_individual[0]:.4f}, p2={best_individual[1]:.4f}, p3={best_individual[2]:.4f}, p4={best_individual[3]:.4f}")

    # 绘制结果
    plot_fitting_curve(best_individual, stress_data)


if __name__ == "__main__":
    main()