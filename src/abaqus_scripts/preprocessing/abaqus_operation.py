from abaqus import *
from abaqusConstants import *

def set_section(m1_part_name, region_name, section_name, model_name="Model-1"):
    """
    为指定part赋section属性，首先将整个part划分成一个region，然后为region赋section属性
    :param m1_part_name: part名称
    :param region_name: region名称
    :param section_name: section名称
    :param model_name: 模型名称
    :return: None
    """
    # 读取指定part，并建立region
    p = mdb.models[model_name].parts[m1_part_name]
    c = p.cells
    cells = c[0:1]
    region = p.Set(cells=cells, name=region_name)
    # 赋给指定part指定section
    p.SectionAssignment(
        region=region,
        sectionName=section_name,
        offset=0.0,
        offsetType=MIDDLE_SURFACE,
        offsetField="",
        thicknessAssignment=FROM_SECTION,
    )


def part_rename(from_name: str, to_name: str, model_name: str = "Model-1"):
    """
    重命名part
    :param from_name: 原始的part名称
    :param to_name: 新的part名称
    :param model_name: 模型名称
    :return: None
    """
    mdb.models[model_name].parts.changeKey(fromName=from_name, toName=to_name)


def instance_rename(from_name, to_name, model_name="Model-1"):
    mdb.models[model_name].rootAssembly.instances.changeKey(
        fromName=from_name, toName=to_name
    )


def surface_set(instance_name, model_name="Model-1"):
    # 读取指定Instance
    a = mdb.models[model_name].rootAssembly
    # 获取指定Instance的所有表面
    s = a.instances[instance_name].faces
    # 将所有表面形成一个表面序列
    s1 = 0
    s2 = len(s)
    side1_faces1 = s[s1:s2]
    # 定义Surface名称
    surface_name = instance_name + "-Surface"
    # 将所有表面序列定义为一个Surface
    a.Surface(side1Faces=side1_faces1, name=surface_name)
    return surface_name


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
