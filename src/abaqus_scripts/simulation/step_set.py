from abaqus import *
from abaqusConstants import *
import numpy as np
import pandas as pd
import json
import re
from itertools import groupby


ROOT_PATH = r"E:\MyResearch\MyWork\multi_step_simulation"
BASEMODEL = "Model-No-Loads"


def extract_and_merge_indices(string_list):
    """
    提取字符串列表中的数字并合并连续数字
    :param string_list: 字符串列表
    :return: 合并后的数字列表
    """
    # 提取数字部分并转换为整数
    numbers = [int(re.search(r"\.(\d+)", s).group(1)) for s in string_list]

    # 对连续数字进行合并
    merged = []
    for k, g in groupby(enumerate(numbers), lambda x: x[0] - x[1]):
        group = list(map(lambda x: x[1], g))
        if len(group) > 1:
            merged.append((group[0], group[-1]))  # 改为存储元组
        else:
            merged.append((group[0], group[0]))  # 单个数字也存储为元组
    return merged


def get_elements_by_indices(elements, index_ranges):
    """
    根据索引范围提取元素
    :param elements: 元素集合
    :param index_ranges: 索引范围
    :return: 提取的元素
    """

    result = elements[0:0]  # 初始化一个空的元素集合

    for start, end in index_ranges:
        result += elements[start - 1 : end]  # 注意Python切片是左闭右开，所以需要+1
    return result


def cut_force_calculate(conc_force_dir, cut_force):

    F_f = cut_force[0]
    F_s = cut_force[1]
    F_n = cut_force[2]

    Dx = conc_force_dir[0]
    Dy = conc_force_dir[1]
    Dz = conc_force_dir[2]

    # 归一化进给方向向量
    D = np.array([Dx, Dy, Dz])
    norm = np.linalg.norm(D)
    if norm == 0:
        raise ValueError("进给方向向量不能为零向量")
    u = D / norm

    # 选择参考向量构造正交基底
    reference = np.array([0, 1, 0])  # 全局Y轴
    if np.allclose(np.cross(u, reference), [0, 0, 0]):
        reference = np.array([0, 0, 1])  # 若共线，改用全局Z轴

    # 计算v'并归一化
    v_prime = np.cross(reference, u)
    v_norm = np.linalg.norm(v_prime)
    if v_norm == 0:
        raise ValueError("无法构造正交基底，参考向量与进给方向共线")
    v = v_prime / v_norm

    # 计算第三个正交向量w
    w = np.cross(u, v)

    # 构建旋转矩阵
    R = np.column_stack((u, v, w))

    # 转换力向量
    F_local = np.array([F_f, F_s, F_n])
    F_global = R @ F_local

    return F_global.tolist()


def step_0_set(model, root_assembly, disactive_section_list):
    """在initial工步下设置step0的内容:删除该工步前已经被彻底消除的网格

    Parameters
    ----------
    root_assembly : root_assembly对象
        abaqus模型的root_assembly对象`
    disactive_section_list : list
        需要被删除的section列表
    """
    step_name = "Step-0"

    model.StaticStep(name=step_name, previous="Initial")

    for section in disactive_section_list:
        region = root_assembly.instances[section].sets["Section"]
        model.ModelChange(
            name=f"Int-{step_name}-disactive-{section}",
            createStepName=step_name,
            region=region,
            activeInStep=False,
            includeStrain=True,
        )


def step_set(
    model,
    prev_step_name,
    step_info,
    act_meshes_inst,
    deact_top_surf_meshes_inst,
    deact_allow_meshes_inst,
    deact_last_surf_meshes_inst,
):
    """
    用于生成各个工步的内容
    """

    # 读取step信息
    step_name = step_info["step_id"]
    act_meshes = step_info["activate_meshes"]
    deact_top_surf_meshes = step_info["deactivate_top_surface_meshes"]
    deact_last_proc_surf_meshes = step_info["deactivate_last_process_surface_meshes"]
    deact_allow_meshes = step_info["deactivate_allowance_meshes"]
    conc_force_dir = step_info["concentrated_force_direction"]
    conc_mesh = step_info["concentrated_mesh"]

    # 新建分析步step
    model.StaticStep(name=step_name, previous=prev_step_name)

    # 记录装配体变量
    root_assembly = model.rootAssembly

    # 激活网格
    # region 建立激活网格set
    if len(act_meshes) == 0:
        pass
    else:
        elements = root_assembly.instances[act_meshes_inst].elements
        element_index_range = extract_and_merge_indices(act_meshes)
        element = get_elements_by_indices(elements, element_index_range)
        root_assembly.Set(
            elements=element,
            name=f"{step_name}_act_meshes",
        )

        # 在step中激活对应网格
        model.ModelChange(
            name=f"act_{step_name}_act_meshes",
            createStepName=step_name,
            region=root_assembly.sets[f"{step_name}_act_meshes"],
            activeInStep=True,
            includeStrain=True,
        )

    # endregion

    # 取消激活网格
    # region 建立取消激活顶层网格set
    if len(deact_top_surf_meshes) == 0:
        pass
    else:
        elements = root_assembly.instances[deact_top_surf_meshes_inst].elements
        element_index_range = extract_and_merge_indices(deact_top_surf_meshes)
        element = get_elements_by_indices(elements, element_index_range)
        root_assembly.Set(
            elements=element,
            name=f"{step_name}_deact_top_surf_meshes",
        )

        # 在step中取消激活顶层网格
        model.ModelChange(
            name=f"deact_{step_name}_deact_top_surf_meshes",
            createStepName=step_name,
            region=root_assembly.sets[f"{step_name}_deact_top_surf_meshes"],
            activeInStep=False,
            includeStrain=False,
        )
    # endregion

    # region 建立取消激活前道工序表层网格set
    if len(deact_last_proc_surf_meshes) == 0:
        pass
    else:
        elements = root_assembly.instances[deact_last_surf_meshes_inst].elements
        element_index_range = extract_and_merge_indices(deact_last_proc_surf_meshes)
        element = get_elements_by_indices(elements, element_index_range)
        root_assembly.Set(
            elements=element,
            name=f"{step_name}_deact_last_proc_surf_meshes",
        )

        # 在step中取消激活顶层网格
        model.ModelChange(
            name=f"deact_{step_name}_deact_last_proc_surf_meshes",
            createStepName=step_name,
            region=root_assembly.sets[f"{step_name}_deact_last_proc_surf_meshes"],
            activeInStep=False,
            includeStrain=False,
        )
    # endregion

    # region 建立取消激活顶层网格set
    if len(deact_allow_meshes) == 0:
        pass
    else:
        elements = root_assembly.instances[deact_allow_meshes_inst].elements
        element_index_range = extract_and_merge_indices(deact_allow_meshes)
        element = get_elements_by_indices(elements, element_index_range)
        root_assembly.Set(
            elements=element,
            name=f"{step_name}_deact_allow_meshes",
        )

        # 在step中取消激活顶层网格
        model.ModelChange(
            name=f"deact_{step_name}_deact_allow_meshes",
            createStepName=step_name,
            region=root_assembly.sets[f"{step_name}_deact_allow_meshes"],
            activeInStep=False,
            includeStrain=False,
        )
    # endregion

    # region 抑制上一工步的集中力
    if prev_step_name != "Step-0":
        try:
            model.loads[f"{prev_step_name}_conc_force"].deactivate(step_name)
        except KeyError:
            pass

    # region 添加此工步的集中力
    conc_force = [100, 200, -50]
    if len([conc_mesh]) == 0:
        pass
    else:
        real_conc_force = cut_force_calculate(conc_force_dir, conc_force)

        elements = root_assembly.instances[act_meshes_inst].elements
        element_index_range = extract_and_merge_indices([conc_mesh])
        element = get_elements_by_indices(elements, element_index_range)
        root_assembly.Set(
            elements=element,
            name=f"{step_name}_conc_mesh",
        )
        region = root_assembly.sets[f"{step_name}_conc_mesh"]
        model.BodyForce(
            name=f"{step_name}_conc_force",
            createStepName=step_name,
            region=region,
            comp1=real_conc_force[0],
            comp2=real_conc_force[1],
            comp3=real_conc_force[2],
        )
    # endregion


def main():

    process_name = "FINISH_1"

    model_name = f"model_{process_name}"

    # # 读取各个工序信息的Excel文件，文件第一行为表头
    # machine_info = pd.read_csv(
    #     ROOT_PATH + "/data/processed/machine_info/刀轨加工参数及对应删改网格.csv"
    # )

    act_meshes_inst = "FinishSurface-1"
    deact_top_surf_meshes_inst = "FinishTopSurface-1"
    deact_allow_meshes_inst = "FinishAllowance-1"
    deact_last_surf_meshes_inst = "Simi2Surface-1"

    # 基于基准模型复制单一工步下的模型
    mdb.Model(name=model_name, objectToCopy=mdb.models[BASEMODEL])
    model = mdb.models[model_name]

    disactive_section_list = [
        "RoughAllowance-1",
        "RoughAllowance-2",
        "RoughAllowance-3",
        "RoughAllowance-4",
        "RoughSurface-1",
        "RoughSurface-2",
        "RoughSurface-3",
        "RoughSurface-4",
        "RoughTopSurface-1",
        "RoughTopSurface-2",
        "RoughTopSurface-3",
        "RoughTopSurface-4",
        "Simi1Allowance-1",
        "Simi1Allowance-2",
        "Simi1Allowance-3",
        "Simi1Allowance-4",
        "Simi1Surface-1",
        "Simi1Surface-2",
        "Simi1Surface-3",
        "Simi1Surface-4",
        "Simi1TopSurface-1",
        "Simi1TopSurface-2",
        "Simi1TopSurface-3",
        "Simi1TopSurface-4",
        "Simi2Allowance-1",
        "Simi2Allowance-2",
        "Simi2Allowance-3",
        "Simi2Allowance-4",
        "Simi2TopSurface-1",
        "Simi2TopSurface-2",
        "Simi2TopSurface-3",
        "Simi2TopSurface-4",
        "FinishSurface-1",
        "FinishSurface-2",
        "FinishSurface-3",
        "FinishSurface-4",
    ]

    root_assembly = model.rootAssembly

    # 在step0中删除前道工序的残余应力层网格
    step_0_set(model, root_assembly, disactive_section_list)

    # 解包指定工序的JSON文件
    with open(ROOT_PATH + rf"\data\processed\step_files\{process_name}.json", "r") as f:
        steps_info = json.load(f)

    previous_step = "Step-0"

    for step_info in steps_info[:60]:

        step_set(
            model,
            previous_step,
            step_info,
            act_meshes_inst,
            deact_top_surf_meshes_inst,
            deact_allow_meshes_inst,
            deact_last_surf_meshes_inst,
        )

        previous_step = step_info["step_id"]


if __name__ == "__main__":
    main()


# a = mdb.models["Model-test"].rootAssembly
# e1 = a.instances["RoughTopSurface-2"].elements
# elements1 = e1[390:391]
# a.Set(elements=elements1, name="Set-3")
# #: The set 'Set-3' has been created (1 element).

# mdb.models["Model-test"].StaticStep(name="Step-1", previous="Initial")
# session.viewports["Viewport: 1"].assemblyDisplay.setValues(step="Step-1")

# a = mdb.models["Model-test"].rootAssembly
# region = a.sets["Set-3"]
# mdb.models["Model-test"].BodyForce(
#     name="Load-1", createStepName="Step-1", region=region, comp3=-100.0
# )

# mdb.models["Model-test"].loads["Load-1"].deactivate("Step-2")

# a = mdb.models['Model-test'].rootAssembly
# region =a.instances['FinishAllowance-1'].sets['Section']
# mdb.models['Model-test'].ModelChange(name='Int-1', createStepName='Step-2',
#     region=region, activeInStep=False, includeStrain=False)
# #: The interaction "Int-1" has been created.

# a = mdb.models['Model-test'].rootAssembly
# region =a.instances['RoughAllowance-3'].sets['Section']
# mdb.models['Model-test'].ModelChange(name='Int-2', createStepName='Step-2',
#     region=region, activeInStep=True, includeStrain=True)
# #: The interaction "Int-2" has been created.
