# QArobot
# 1.本地部署需要安装包如下：
streamlit、pandas、geopandas、folium、streamlit_folium、PIL、matplotlib、plotly、numpy、json、openai、langchain_experimental、httpx、threading、queue、wave、io、pyaudio
注意：  httpx版本不宜过高，最新版有bug，实测有效版本0.27.2

# 2.代码运行需要读取三个本地文件，已打包进“支撑文件”文件夹
运行源码前需在以下代码中修改三处文件位置：(位于源码“voiceAI.py”起始段，约43行-53行)

## 2.1地图配置参数（可根据实际情况调整）
shp_path = r"E:\Spyder work\AI\geo\JSExpwy2025.shp"
image_path = r"E:\Spyder work\AI\geo\江苏省OSM_220919152142_L10\OSM_220919152142.png"
### Folium 格式 [[min_lat, min_lon], [max_lat, max_lon]]
image_bounds = [[29.8406439, 115.3125], [35.7465123, 123.3984375]]


def load_coordinates():
    coords_path = r"E:\Spyder work\AI\测试文件\坐标表.xlsx"
    df_coords = pd.read_excel(coords_path)
    return df_coords


# 3.在终端输入：
streamlit run "E:\voiceAI.py" 
即可启动本地服务，引号内地址请自行替换

# 4.
（1）地图可视化模块依赖“断面编号”，可参考打包文件内“输入测试文件”文件夹下“断面2025年01月14日至2025年01月27日.csv”文件（编码为gbk）查看可匹配断面编号；
（2）期望线模块依赖打包文件内“支撑文件”文件夹下“坐标表.xlsx”所提供经纬度坐标，若期望绘图地点名称不在表内可自行补充。
