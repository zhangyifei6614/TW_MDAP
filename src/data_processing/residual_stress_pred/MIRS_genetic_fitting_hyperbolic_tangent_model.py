import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# 绘制拟合曲线与原始数据的对比图
def plot_fitting_curve(best_individual, stress_data):
    """
    绘制拟合曲线与原始数据的对比图。

    参数:
        best_individual (list): 最佳个体，包含两个参数 [p1, p2]。
        stress_data (numpy.ndarray): 原始应力数据，形状为 (n, 2)。
    """
    p1, p2 = best_individual  # 提取最佳个体的参数
    x_vals = stress_data[:, 0]  # 深度数据
    y_vals = stress_data[:, 1]  # 残余应力数据

    # 计算拟合曲线的值
    fitted_x_vals = np.arange(0, 1, 0.005)
    fitted_y_vals = p1 * np.tanh(p2 * fitted_x_vals) - p1

    # 绘制散点图和拟合曲线
    plt.scatter(x_vals, y_vals, label='原始数据', color='blue')
    plt.plot(fitted_x_vals, fitted_y_vals, label='拟合曲线', color='red', linewidth=2)
    plt.xlabel('深度')
    plt.ylabel('残余应力')
    plt.title('拟合曲线 vs 原始数据')
    plt.legend()
    plt.show()


# 定义适应度函数
def fitness(individual, stress_data):
    """
    计算个体的适应度（基于均方误差的倒数）。

    参数:
        individual (list): 个体，包含两个参数 [p1, p2]。
        stress_data (numpy.ndarray): 原始应力数据，形状为 (n, 2)。

    返回:
        float: 适应度值。
    """
    p1, p2 = individual

    def hyperbolic_tangent_model(p1, p2, x):
        """双曲正切模型计算残余应力"""
        return p1 * np.tanh(p2 * x) - p1

    # 计算均方误差
    mean_squared_error = 0
    for i in range(len(stress_data)):
        x, y = stress_data[i]
        predicted_y = hyperbolic_tangent_model(p1, p2, x)
        mean_squared_error += (y - predicted_y) ** 2
    mean_squared_error /= len(stress_data)

    # 返回适应度值（均方误差的倒数）
    return 1 / mean_squared_error if mean_squared_error != 0 else 0


# 初始化种群
def initialize_population(pop_size, gene_length):
    """
    初始化种群。

    参数:
        pop_size (int): 种群大小。
        gene_length (float): 基因的最大实数值。

    返回:
        numpy.ndarray: 初始种群，形状为 (pop_size, 2)。
    """
    population = np.zeros((pop_size, 2))
    for i in range(pop_size):
        population[i, 0] = np.random.rand() * gene_length
        population[i, 1] = np.random.rand() * gene_length / 100
    return population


# 选择函数
def select(population, fitness_scores):
    """
    根据适应度分数选择一个个体。

    参数:
        population (numpy.ndarray): 种群，形状为 (pop_size, 2)。
        fitness_scores (numpy.ndarray): 每个个体的适应度分数。

    返回:
        numpy.ndarray: 被选中的个体。
    """
    total_fitness = np.sum(fitness_scores)
    selection_probs = fitness_scores / total_fitness
    cumulative_probs = np.cumsum(selection_probs)
    r = np.random.rand()
    for i in range(len(cumulative_probs)):
        if r < cumulative_probs[i]:
            return population[i]
    return population[-1]  # 如果未选中任何个体，则返回最后一个个体


# 交叉函数
def crossover(parent1, parent2):
    """
    交叉操作生成两个子代。

    参数:
        parent1 (numpy.ndarray): 父代1。
        parent2 (numpy.ndarray): 父代2。

    返回:
        tuple: 两个子代。
    """
    alpha = np.random.rand()
    child1 = alpha * parent1 + (1 - alpha) * parent2
    child2 = (1 - alpha) * parent1 + alpha * parent2
    return child1, child2


# 变异函数
def mutate(individual, mutation_rate):
    """
    对个体进行变异操作。

    参数:
        individual (numpy.ndarray): 个体。
        mutation_rate (float): 变异率。

    返回:
        numpy.ndarray: 变异后的个体。
    """
    for i in range(len(individual)):
        if np.random.rand() < mutation_rate:
            individual[i] += np.random.randn() * 10  # 高斯变异，标准差为10
            individual[i] = max(1.0, min(1000.0, individual[i]))  # 限制范围在 (1, 1000)
    return individual


# 遗传算法主函数
def genetic_algorithm(pop_size, gene_length, generations, mutation_rate, stress_data):
    """
    遗传算法主函数。

    参数:
        pop_size (int): 种群大小。
        gene_length (float): 基因的最大实数值。
        generations (int): 迭代代数。
        mutation_rate (float): 变异率。
        stress_data (numpy.ndarray): 原始应力数据。

    返回:
        numpy.ndarray: 最佳个体。
    """
    population = initialize_population(pop_size, gene_length)
    for gen in range(generations):
        fitness_scores = np.array([fitness(individual, stress_data) for individual in population])
        new_population = []
        while len(new_population) < pop_size:
            parent1 = select(population, fitness_scores)
            parent2 = select(population, fitness_scores)
            child1, child2 = crossover(parent1, parent2)
            new_population.append(mutate(child1, mutation_rate))
            new_population.append(mutate(child2, mutation_rate))
        population = np.array(new_population[:pop_size])  # 截取到指定大小
    best_index = np.argmax(fitness_scores)
    return population[best_index]


# 主函数
def main():
    """
    主函数：读取数据、运行遗传算法并绘制结果。
    """
    # 读取 CSV 文件并处理数据
    file_path = 'E:/MyResearch/Papers/一种大型复杂薄壁零件多工步铣削加工变形有限元预测方法/论文数据及代码分析/端面铣削/VB0/UTF8_ap12ae08fz008.csv'
    stress_data = pd.read_csv(file_path).values
    stress_data = np.hstack((stress_data[:, 0].reshape(-1, 1), stress_data[:, 2:].reshape(-1, 1)))  # 删除第二列

    # 参数设置
    pop_size = 500  # 种群大小
    gene_length = 1000  # 基因最大值
    generations = 100  # 迭代代数
    mutation_rate = 0.01  # 变异率

    # 运行遗传算法
    best_individual = genetic_algorithm(pop_size, gene_length, generations, mutation_rate, stress_data)
    print(f'最佳个体: {best_individual}')

    # 绘制拟合曲线
    plot_fitting_curve(best_individual, stress_data)


if __name__ == "__main__":
    main()