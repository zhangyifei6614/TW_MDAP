from abaqus import *
from abaqusConstants import *
import os

# 设置仿真工作目录
ROOTPATH = r"E:\00-Working\TW_GH4169_MDAP\data\simulation"
# 设置仿真文件名
SIMULATION_NAME = r"\Frame_2"


# 设置abaqus工作目录
os.chdir(ROOTPATH)

# 设置abaqus为按实际索引号进行索引
session.journalOptions.setValues(replayGeometry=INDEX, recoverGeometry=INDEX)

# 打开模型数据库
openMdb(pathName=ROOTPATH + SIMULATION_NAME)

# 直接进入装配视角
a = mdb.models["Model-No-Loads"].rootAssembly
session.viewports["Viewport: 1"].setValues(displayedObject=a)
