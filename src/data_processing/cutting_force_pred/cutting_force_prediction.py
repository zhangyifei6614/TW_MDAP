import math
import matplotlib.pyplot as plt


def predict_side_milling_forces(
    phi, D, N, beta_deg, fz, a_p, a_r, Kt, Kr, Ka, dz=0.001
):
    """
    预测侧铣切削力
    :param phi: 当前刀具旋转角度（弧度）
    :param D: 刀具直径（mm）
    :param N: 刀齿数量
    :param beta_deg: 螺旋角（度数）
    :param fz: 每齿进给量（mm/tooth）
    :param a_p: 轴向切深（mm）
    :param a_r: 径向切深（mm）
    :param Kt, Kr, Ka: 切削力系数（N/mm²）
    :param dz: 轴向微元高度（mm）
    :return: Fx, Fy, Fz 切削力分量（N）
    """
    R = D / 2
    beta = math.radians(beta_deg)
    theta_immerse = math.acos(1 - 2 * a_r / D)  #
    theta_entry = math.pi - theta_immerse
    theta_exit = math.pi

    total_Fx = 0.0
    total_Fy = 0.0
    total_Fz = 0.0

    num_elements = int(a_p / dz)
    if num_elements == 0:
        num_elements = 1

    for j in range(N):
        phi_j = phi + j * (2 * math.pi / N)

        for k in range(num_elements):
            z = k * dz + dz / 2  # 微元中心位置
            delta_phi = (z * math.tan(beta)) / R
            current_phi = (phi_j - delta_phi) % (2 * math.pi)

            if theta_entry <= current_phi <= theta_exit:
                h = fz * math.sin(current_phi)

                dFt = Kt * h * dz
                dFr = Kr * h * dz
                dFa = Ka * h * dz

                Fx = -dFt * math.sin(current_phi) - dFr * math.cos(current_phi)
                Fy = dFt * math.cos(current_phi) - dFr * math.sin(current_phi)
                Fz = dFa

                total_Fx += Fx
                total_Fy += Fy
                total_Fz += Fz

    return total_Fx, total_Fy, total_Fz


# 示例用法
if __name__ == "__main__":
    # 侧铣示例
    S = 1210.0  # 切削速度 (r/min)

    D = 10.0  # 刀具直径 (mm)
    N = 4  # 刀齿数
    beta_deg = 30  # 螺旋角 (度)
    fz = 0.08  # 每齿进给量 (mm/tooth)
    a_p = 1.0  # 轴向切深 (mm)
    a_r = 0.3  # 径向切深 (mm)
    Kt = 2000.0  # 切向力系数 (N/mm²)
    Kr = 5000.0  # 径向力系数 (N/mm²)
    Ka = 200.0  # 轴向力系数 (N/mm²)

    import numpy as np

    t = np.arange(0.005, 0.015, 0.00005)  # 时间序列 (s)

    phi = 2 * math.pi * S * t / 60.0  # 刀具旋转角度 (弧度)

    Fx = np.zeros_like(t)
    Fy = np.zeros_like(t)
    Fz = np.zeros_like(t)

    for i in range(len(t)):
        Fx[i], Fy[i], Fz[i] = predict_side_milling_forces(
            phi[i], D, N, beta_deg, fz, a_p, a_r, Kt, Kr, Ka
        )

    plt.figure(figsize=(10, 6))
    plt.plot(t, Fx, label="Fx (切向力)", color="blue")
    plt.plot(t, Fy, label="Fy (径向力)", color="orange")
    plt.plot(t, Fz, label="Fz (轴向力)", color="green")
    plt.title("侧铣切削力预测")
    plt.xlabel("时间 (s)")
    plt.ylabel("切削力 (N)")
    plt.legend()  # 显示图例
    plt.grid()
    plt.show()
