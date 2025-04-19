import sys

sys.path.append(r"/src/abaqus_scripts/preprocessing/abaqus_operation.py")

from abaqus_operation import *

# 定义本程序运行时处理的Model名称
ModelName = "Model-1"

name_list = [
    "RoughSurface",
    "Simi1Surface",
    "Simi2Surface",
    "FinishSurface",
    "SideSurface",
    "BottomSurface",
    "TopSurface",
    "FinishTopSurface",
    "Simi2TopSurface",
    "Simi1TopSurface",
    "RoughTopSurface",
    "RoughAllowance",
    "Simi1Allowance",
    "Simi2Allowance",
    "FinishAllowance",
    "Workpiece",
]

for i in range(1, 17):
    FromName = "Frame-" + str(i)
    ToName = name_list[i - 1]
    part_rename(FromName, ToName, ModelName)

# 定义本程序运行时处理的Model名称
ModelName = "Model-1"

# 定义part命名序列
PartNameList = [
    "Workpiece",
    "TopSurface",
    "SideSurface",
    "BottomSurface",
    "RoughAllowance",
    "RoughSurface",
    "RoughTopSurface",
    "Simi1Allowance",
    "Simi1Surface",
    "Simi1TopSurface",
    "Simi2Allowance",
    "Simi2Surface",
    "Simi2TopSurface",
    "FinishAllowance",
    "FinishSurface",
    "FinishTopSurface",
]

# 定义各part在装配体中的数量
PartCount = [1, 1, 1, 1, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]

# 循环命名各Instance
for i in range(0, 16):
    part_name = PartNameList[i]
    for j in range(2, PartCount[i] + 1):
        FromName = part_name + "-1-lin-" + str(j) + "-1"
        ToName = part_name + "-" + str(j)
        instance_rename(FromName, ToName, ModelName)


# 定义运行脚本时面向的模型名称
ModelName = "Model-1"

name_list = [
    "RoughSurface",
    "Simi1Surface",
    "Simi2Surface",
    "FinishSurface",
    "SideSurface",
    "BottomSurface",
    "TopSurface",
    "FinishTopSurface",
    "Simi2TopSurface",
    "Simi1TopSurface",
    "RoughTopSurface",
    "RoughAllowance",
    "Simi1Allowance",
    "Simi2Allowance",
    "FinishAllowance",
    "Workpiece",
]


# 将所有Part的属性赋为GH4169
for i in range(1, 17):
    PartName = name_list[i - 1]
    RegionName = "Section"
    SectionName = "GH4169"
    set_section(PartName, RegionName, SectionName, ModelName)


# 定义脚本运行目标模型名称
ModelName = "Model-1-Test1"


# 定义part命名序列
PartNameList = [
    "Workpiece",
    "TopSurface",
    "SideSurface",
    "BottomSurface",
    "FinishAllowance",
    "FinishSurface",
    "FinishTopSurface",
    "Simi2Allowance",
    "Simi2Surface",
    "Simi2TopSurface",
    "Simi1Allowance",
    "Simi1Surface",
    "Simi1TopSurface",
    "RoughAllowance",
    "RoughSurface",
    "RoughTopSurface",
]

# 定义各part在装配体中的数量
PartCount = [1, 1, 1, 1, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]

for i in range(1, 4):
    for j in range(1, PartCount[i] + 1):
        MainInstanceName = PartNameList[0] + "-1"
        SecondaryInstanceName = PartNameList[i] + "-" + str(j)
        tie_set(MainInstanceName, SecondaryInstanceName, ModelName)

for i in range(4, 6):
    for j in range(1, PartCount[i] + 1):
        MainInstanceName = PartNameList[0] + "-1"
        SecondaryInstanceName = PartNameList[i] + "-" + str(j)
        tie_set(MainInstanceName, SecondaryInstanceName, ModelName)

for i in range(6, 9):
    for j in range(1, PartCount[i] + 1):
        MainInstanceName = PartNameList[4] + "-" + str(j)
        SecondaryInstanceName = PartNameList[i] + "-" + str(j)
        tie_set(MainInstanceName, SecondaryInstanceName, ModelName)

for i in range(9, 12):
    for j in range(1, PartCount[i] + 1):
        MainInstanceName = PartNameList[7] + "-" + str(j)
        SecondaryInstanceName = PartNameList[i] + "-" + str(j)
        tie_set(MainInstanceName, SecondaryInstanceName, ModelName)

for i in range(12, 15):
    for j in range(1, PartCount[i] + 1):
        MainInstanceName = PartNameList[10] + "-" + str(j)
        SecondaryInstanceName = PartNameList[i] + "-" + str(j)
        tie_set(MainInstanceName, SecondaryInstanceName, ModelName)

for i in range(15, 16):
    for j in range(1, PartCount[i] + 1):
        MainInstanceName = PartNameList[13] + "-" + str(j)
        SecondaryInstanceName = PartNameList[i] + "-" + str(j)
        tie_set(MainInstanceName, SecondaryInstanceName, ModelName)

# region: 重新绑定边界条件
# 定义脚本运行目标模型名称
model_name = "Model-No-Loads"
model = mdb.models[model_name]
root_asm = model.rootAssembly

# 绑定 Workpiece-1 的边界条件
# workpiece——TopSurface


# workpiece——BottomSurface

# workpiece——SideSurface

# workpiece——FinishSurface

# workpiece——FinishAllowance


def tie_set(main_instance_name, secondary_instance_name, model_name="Model-1"):
    # 定义指定model
    m = mdb.models[model_name]
    # 读取指定Instance
    a = mdb.models[model_name].rootAssembly
    # 获取指定Instance的所有表面
    s1 = surface_set(main_instance_name, model_name)
    s2 = surface_set(secondary_instance_name, model_name)
    # 为Tie约束命名
    tie_name = main_instance_name + "-C-" + secondary_instance_name
    # 通过Tie约束连接两个Surface
    m.Tie(
        name=tie_name,
        main=a.surfaces[s1],
        secondary=a.surfaces[s2],
        positionToleranceMethod=COMPUTED,
        adjust=OFF,
        tieRotations=ON,
        thickness=ON,
    )
    print(tie_name)
