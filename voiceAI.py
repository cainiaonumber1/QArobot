# -*- coding: utf-8 -*-
"""
Created on Tue Jul  1 09:59:16 2025

@author: 032614
"""

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import geopandas as gpd
import folium
from streamlit_folium import folium_static
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, rgb2hex
from matplotlib.font_manager import FontProperties, fontManager
import os
import plotly.graph_objects as go
import numpy as np
import json
from openai import OpenAI
from langchain_experimental.utilities import PythonREPL
import httpx  # Version: 0.27.2
import io
import base64

# import wave
# import pyaudio
# import threading
# import queue

# import contextily as cx
# from langchain.chains import LLMChain
# from langchain.prompts import PromptTemplate
# from pydub import AudioSegment
# import tempfile
# import time

# streamlit run "E:\Spyder work\AI\voiceAI.py"
# git add .
# git commit -m "Update files"
# git push


# 设置页面布局
st.set_page_config(page_title="QA Robot", layout="wide")
st.title("🌍 智能数据分析问答机器人 📊")

# 获取当前文件所在目录
base_dir = os.path.dirname(__file__)

# 地图配置参数（可根据实际情况调整）
shp_path = os.path.join(base_dir, "支撑文件", "JSExpwy2025.shp")
image_path = os.path.join(base_dir, "支撑文件", "OSM_220919152142.png")
# Folium 格式 [[min_lat, min_lon], [max_lat, max_lon]]
image_bounds = [[29.8406439, 115.3125], [35.7465123, 123.3984375]]

# 加载中文字体
font_path = os.path.join(base_dir, "simhei.ttf")
if os.path.exists(font_path):
    fontManager.addfont(font_path)
    prop = FontProperties(fname=font_path)
    plt.rcParams['font.sans-serif'] = [prop.get_name()]
else:
    st.warning("⚠️ 字体文件 SimHei.ttf 未找到，中文可能无法显示")

plt.rcParams['axes.unicode_minus'] = False

def load_coordinates():
    coords_path = os.path.join(base_dir, "支撑文件", "坐标表.xlsx")
    df_coords = pd.read_excel(coords_path)
    return df_coords


def init_deepseek():
    # 创建自定义HTTP客户端
    custom_client = httpx.Client(
        proxies=None,  # 显式禁用代理
        trust_env=False,  # 新增：禁止读取环境变量代理配置
        timeout=30.0,
        transport=httpx.HTTPTransport(retries=3)
    )

    return OpenAI(
        api_key="sk-ee72ed73b1bf4a2bbe867660fcfe52b2",
        base_url="https://api.deepseek.com/v1",
        http_client=custom_client  # 使用自定义客户端
    )


# ----------------------------
# 初始化 session_state
# ----------------------------
if 'mode' not in st.session_state:
    st.session_state.mode = "分析问答"  # 默认进入模式
if 'map_ready' not in st.session_state:
    st.session_state.map_ready = False

# ----------------------------
# 加载并缓存路网数据（只加载一次）
# ----------------------------


@st.cache_resource
def load_gdf():
    gdf = gpd.read_file(shp_path)
    if gdf.crs is None or gdf.crs.to_string() != "EPSG:4326":
        gdf = gdf.to_crs(epsg=4326)
    return gdf


def map_values_to_roadno(gdf, df_user, match_field, value_field):
    """
    将用户数据中 match_field 与 ROADNO 匹配的 value_field 值映射到路网数据上。
    如果未匹配到，则填 0。

    参数:
        gdf (GeoDataFrame): 路网数据
        df_user (DataFrame): 用户上传的数据
        match_field (str): 用户数据中与 ROADNO 对应的字段
        value_field (str): 需要映射到地图上的数值字段

    返回:
        GeoDataFrame: 新增 value_field 列的路网数据（含 0 填充）
    """
    # 创建映射字典 {match_field_value: value_field_value}
    mapping = dict(zip(df_user[match_field], df_user[value_field]))

    # 在路网数据中新增一列，根据 ROADNO 查找对应的 value_field 值
    gdf[value_field] = gdf['ROADNO'].map(mapping)

    # 填充空值为 0
    gdf[value_field].fillna(0, inplace=True)

    # # 删除 geometry 为空的行（安全起见）
    # gdf = gdf[gdf.geometry.notnull()]

    return gdf


def render_map(full_gdf, matched_data, value_field):
    """
    渲染地图，支持：
    - 原始完整路网
    - 用户上传的 PNG 底图
    - 匹配数据的动态样式（颜色、线宽）

    参数:
        full_gdf (GeoDataFrame): 完整的原始路网数据
        matched_data (GeoDataFrame): 已匹配到数据的路段
        value_field (str): 要展示的数值字段名
    """

    # 创建地图对象
    center_lat = full_gdf.geometry.centroid.y.mean()
    center_lon = full_gdf.geometry.centroid.x.mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=8, tiles=None)

    # 添加默认底图
    folium.TileLayer('OpenStreetMap', name='OpenStreetMap').add_to(m)

    # 加载并添加 PNG 图像为底图（默认不显示）
    try:
        img = Image.open(image_path).convert("RGBA")
        img_data = np.array(img)
        img_data[:, :, 3] = (np.ones_like(img_data[:, :, 3])
                             * 255 * 0.8).astype(np.uint8)  # 设置透明度

        folium.raster_layers.ImageOverlay(
            image=img_data,
            bounds=image_bounds,
            name="PNG 底图",
            opacity=1.0,
            show=False,  # 默认隐藏
            cross_origin=False,
            zindex=1,
        ).add_to(m)
    except Exception as e:
        st.warning(f"加载底图失败：{e}")

    # 添加原始完整路网（灰色线条）
    folium.GeoJson(
        full_gdf,
        name='原始路网',
        style_function=lambda x: {'color': '#666', 'weight': 1},
        tooltip=folium.GeoJsonTooltip(fields=['ROADNO'], aliases=['路段编号']),
    ).add_to(m)

    # 如果有匹配数据，则绘制高亮路段
    if not matched_data.empty and value_field in matched_data.columns:
        # 获取最大最小值用于映射
        max_value = matched_data[value_field].max()
        min_value = matched_data[value_field].min()
        range_value = max_value - min_value if max_value != min_value else 1

        cmap = plt.cm.get_cmap('RdYlGn_r')  # 使用颜色映射
        norm = Normalize(vmin=min_value, vmax=max_value)  # 归一化函数

        def get_color(value):
            if value == 0:
                return "#999999"  # 灰色表示无数据
            else:
                # 获取归一化后的颜色，并转换为HEX格式
                color_rgb = cmap(norm(value))[:3]
                return rgb2hex(color_rgb)

        def get_width(value):
            if value == 0:
                return 1
            return 1 + (value - min_value) / range_value * 15  # 线宽范围

        def style_function(feature):
            value = feature['properties'][value_field]
            return {
                'color': get_color(value),
                'weight': get_width(value),
                'opacity': 0.9
            }

        # 添加高亮路段图层
        folium.GeoJson(
            matched_data,
            name=f'高亮路段 ({value_field})',
            style_function=style_function,
            tooltip=folium.GeoJsonTooltip(
                fields=['ROADNO', value_field],
                aliases=['路段编号', f'{value_field}'],
                localize=True
            )
        ).add_to(m)

    # 添加图层控制
    folium.LayerControl().add_to(m)

    return m


# 语音识别函数
def transcribe_audio(audio_bytes):
    try:
        # 创建OpenAI客户端（使用官方API）
        openai_client = OpenAI(
            api_key="sk-hHLAlXBdnlefUZNrbr9V7okyNBVjNLc7oMzHUGfAqsW4T2Wv",
            base_url="https://api.ipacs.top/v1"
        )
        # 创建语音识别请求 - 使用BytesIO对象
        with io.BytesIO(audio_bytes) as audio_file:
            audio_file.name = "audio.wav"
            # 调用API并获取结构化响应
            response = openai_client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )
            # 调试输出 - 查看响应结构
            # st.write("API响应类型:", type(response))
            # st.write("API响应内容:", response)

            # 提取转录文本 - 检查不同可能的响应结构
            if hasattr(response, 'text'):
                return response.text
            elif hasattr(response, 'result') and hasattr(response.result, 'text'):
                return response.result.text
            elif hasattr(response, 'transcriptions') and len(response.transcriptions) > 0:
                return response.transcriptions[0].text
            else:
                st.error(f"无法解析API响应: {response}")
                return None
    except Exception as e:
        st.error(f"语音识别失败: {str(e)}")
        return None


# 侧边栏设置
with st.sidebar:
    st.header("⚙️ 配置")
    uploaded_file = st.file_uploader(
        "上传 Excel、CSV 或 TXT 文件", type=["xlsx", "csv", "txt"])

    # 初始化session_state
    if 'preview_data' not in st.session_state:
        st.session_state.preview_data = None
    if 'full_data' not in st.session_state:
        st.session_state.full_data = None
    if 'audio_bytes' not in st.session_state:
        st.session_state.audio_bytes = None
    if 'audio_b64' not in st.session_state:
        st.session_state.audio_b64 = None  # 新增：用于存储浏览器录音数据
    if 'transcribed_text' not in st.session_state:
        st.session_state.transcribed_text = ""

    # 文件处理逻辑
    if uploaded_file is not None:
        try:
            file_type = uploaded_file.name.split('.')[-1].lower()

            delimiter = ","
            encoding = "utf-8"

            # 只有 CSV/TXT 才显示解析设置
            if file_type in ['csv', 'txt']:
                st.subheader("📄 文件解析设置")

                # 分隔符：下拉菜单 + 自定义输入
                delimiter_options = {
                    ',': ', (逗号)',
                    '\t': '\\t (制表符)',
                    ';': '; (分号)',
                    ' ': '空格',
                    '|': '| (竖线)'
                }
                delimiter_choice = st.selectbox(
                    "请选择字段分隔符",
                    options=list(delimiter_options.keys()),
                    format_func=lambda x: delimiter_options[x],
                    index=0,
                    key="delimiter_select"
                )
                delimiter_custom = st.text_input(
                    "或自定义分隔符", value=delimiter_choice, key="delimiter_input")
                delimiter = delimiter_custom  # 用户输入优先级高于下拉选择

                # 编码格式：下拉菜单 + 自定义输入
                encoding_options = ["utf-8", "gbk",
                                    "latin1", "utf-8-sig", "cp1252"]
                encoding_choice = st.selectbox(
                    "请选择文件编码格式",
                    options=encoding_options,
                    index=0,
                    key="encoding_select"
                )
                encoding_custom = st.text_input(
                    "或自定义编码格式", value=encoding_choice, key="encoding_input")
                encoding = encoding_custom.strip() or encoding_choice  # 若为空则回退到选择项

            # 根据文件类型加载数据
            if file_type == 'xlsx':
                df = pd.read_excel(uploaded_file)
            elif file_type == 'csv':
                df = pd.read_csv(
                    uploaded_file, delimiter=delimiter, encoding=encoding)
            elif file_type == 'txt':
                df = pd.read_csv(
                    uploaded_file, delimiter=delimiter, encoding=encoding)
            else:
                raise ValueError(f"不支持的文件类型: .{file_type}")

            # 存储前5行数据到session state
            st.session_state.preview_data = df.head(10)
            st.session_state.full_data = df  # 存储完整数据
            # 显示成功提示
            st.toast("✅ 文件已成功加载", icon="✅")

        except Exception as e:
            st.error(f"❌ 文件读取失败: {str(e)}")
            # 清除所有数据状态
            st.session_state.preview_data = None
            st.session_state.full_data = None
    else:
        # 当文件被删除时清除所有数据
        st.session_state.preview_data = None
        st.session_state.full_data = None

    # 显示预览或错误信息
    if st.session_state.preview_data is not None:
        st.subheader("📋 数据预览（前10行）")
        st.dataframe(
            st.session_state.preview_data,
            use_container_width=True,
            height=220  # 控制预览高度
        )
        st.caption(f"总行数: {len(st.session_state.full_data)} 行")  # 显示完整数据的总行数

    else:
        st.subheader("📋 数据预览")
        st.error("🚫 数据预览失败")  # 错误提示
        st.caption("请先上传有效数据文件")  # 补充说明


# 自定义Python执行环境
class SafePythonREPL(PythonREPL):
    def __init__(self):
        super().__init__()
        self.globals = {
            "pd": __import__('pandas'),
            "np": __import__('numpy'),
            "plt": __import__('matplotlib.pyplot'),
            "sns": __import__('seaborn'),
            "st": __import__('streamlit'),
        }
        self.locals = {}

    # 安全执行代码
    def run(self, code):
        try:
            exec(code, self.globals, self.locals)
        except Exception as e:
            return f"执行错误: {str(e)}"


# 问题处理模块
def process_question(df, question, client):
    # 第一步：生成查询代码
    prompt = f"""基于数据表结构：{df.columns.tolist()}
    生成Python代码解决：{question}
    要求：
    1. 只输出代码 
    2. 不要Markdown标记 
    3. 如果使用了函数定义，必须显式调用该函数，并且使用变量 'df' 作为输入数据
    4. 最终结果必须保存到名为 'result' 的变量中
    5. 所有操作都必须基于变量 'df' 进行，不要创建新的数据集
    6. 使用 pd.DataFrame 进行操作
    7. 如果需要绘图，请在全局作用域绘制图表，并将图表保存为 'fig' 变量
    8. 不要在函数内部绘制图表
    9. 绘图时必须设置中文字体（如 SimHei），确保中文正常显示
    10. 不要添加额外解释"""

    # 调用Deepseek API
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=500
    )

    # 提取并清洗代码
    raw_code = response.choices[0].message.content
    clean_code = raw_code.replace("```python", "").replace("```", "").strip()

    # 第二步：执行生成的代码
    try:
        repl = SafePythonREPL()  # 创建执行环境实例
        # 时间列转换
        df_repl = df.copy()
        for col in df_repl.select_dtypes(include=['datetime', 'datetimetz']).columns:
            df_repl[col] = df_repl[col].dt.strftime('%Y-%m-%d %H:%M:%S')
        # 初始化DataFrame
        repl.run(
            f"df = pd.DataFrame({json.dumps(df_repl.to_dict(orient='records'))})")
        # 执行用户代码
        repl.run(clean_code)
        # 获取执行结果和图表
        result = repl.locals.get('result')
        fig = repl.locals.get('fig')  # 获取图表对象

        if result is None and fig is None:
            return "代码执行完成但未生成结果或图表", raw_code, clean_code, None

        return result, raw_code, clean_code, fig  # 返回 fig 对象本身

    except Exception as e:
        return f"执行错误: {str(e)}", "", "", None


def smart_process_map_question(question, df_columns, client):
    """
    使用 DeepSeek 解析问题，返回匹配字段和展示字段
    """

    prompt = f"""
    请根据以下用户问题，从给定字段中找出最合适的两个字段：
    
    1. 用于匹配路网字段 ROADNO 的字段名（如 '路段编号'）。
    2. 用于地图展示的数值型字段（如 '流量', '速度' 等）。
    
    问题: "{question}"
    可用字段: {', '.join(df_columns)}
    
    请以以下格式输出结果：
    匹配字段: xxx
    展示字段: yyy
    """

    # 调用Deepseek API
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=500
    )

    answer = response.choices[0].message.content.strip()

    print(f"API Response: {answer}")  # 打印API响应的内容

    # 解析结果
    lines = answer.split('\n')
    match_field = None
    value_field = None

    for line in lines:
        if line.startswith("匹配字段: "):
            match_field = line.replace("匹配字段: ", "").strip()
        elif line.startswith("展示字段: "):
            value_field = line.replace("展示字段: ", "").strip()

    print(f"Parsed Match Field: {match_field}")  # 打印解析出的match_field
    print(f"Parsed Value Field: {value_field}")  # 打印解析出的value_field

    if match_field is None or value_field is None:
        raise ValueError("未能正确解析匹配字段或展示字段，请检查问题描述或可用字段列表。")

    return match_field, value_field


def process_od_question(od_data, question, client):
    """
    根据用户问题生成Python代码，对 od_data 进行预处理，并匹配经纬度。
    返回处理后的 DataFrame。
    """

    # 加载坐标表
    df_coords = load_coordinates()

    # 提示词模板
    prompt = f"""基于输入数据列：{od_data.columns.tolist()} 和坐标表 {df_coords.columns.tolist()}
    请根据问题：“{question}”，生成Python代码实现以下功能：
    
    1. 处理原始OD数据（变量名为 'od_data'），请根据问题：“{question}”，判断其中的['O', 'D', 'flow']列
    2. 使用坐标表（变量名为 'df_coords'）中的'OD'列进行匹配
    3. 为O和D字段添加对应的纬度('O_lat', 'D_lat')和经度('O_lon', 'D_lon')
    4. 对原始数据进行必要的处理（如过滤、分组、聚合等），以反映问题需求
    5. 最终结果必须保存到名为 'result_df' 的变量中（格式为DataFrame），并且包含以下列：['O', 'D', 'flow', 'O_lat', 'O_lon', 'D_lat', 'D_lon']
    6. 不要使用任何绘图语句
    7. 只输出Python代码，不要解释或Markdown标记
    """

    # 调用Deepseek API
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=500
    )

    # 提取并清洗代码
    raw_code = response.choices[0].message.content.strip()
    clean_code = raw_code.replace("```python", "").replace("```", "").strip()

    # 构建初始命名空间
    namespace = {
        "pd": pd,
        "df_coords": df_coords,
        "od_data": od_data.copy()
    }
    print("df_coords columns:", df_coords.columns.tolist())
    print("执行代码:", clean_code)

    try:
        # 执行代码
        exec(clean_code, namespace)

        # 获取结果
        result_df = namespace.get("result_df")

        if isinstance(result_df, pd.DataFrame):
            return result_df, clean_code
        else:
            raise ValueError("未找到名为 'result_df' 的有效DataFrame")

    except Exception as e:
        return f"执行错误: {str(e)}", ""


def generate_od_lines(od_data):
    """
    根据给定的OD数据生成期望线图。

    参数:
    - od_data: 包含 O, D, flow, O_lat, O_lon, D_lat, D_lon 的 DataFrame
    返回:
    - fig: Plotly Figure 对象
    """
    # 确保所有必需列存在
    required_cols = ['O', 'D', 'flow', 'O_lat', 'O_lon', 'D_lat', 'D_lon']
    if not all(col in od_data.columns for col in required_cols):
        raise ValueError("缺少必需的数据列")

    max_value = od_data['flow'].max()
    od_data['text'] = od_data.apply(
        lambda row: f"{row['O']} → {row['D']}<br>流值: {row['flow']}", axis=1
    )

    # 初始化图表
    fig = go.Figure()

    # 添加期望线（带箭头）
    for _, row in od_data.iterrows():
        fig.add_trace(go.Scattergeo(
            lat=[row['O_lat'], row['D_lat']],
            lon=[row['O_lon'], row['D_lon']],
            mode='lines+markers',
            line=dict(
                width=row['flow'] / max_value * 20 if max_value > 0 else 1,
                color='blue'
            ),
            # marker=dict(
            #     symbol='arrow-bar-up',  # 箭头符号
            #     size=5,
            #     angleref="previous"     # 自动对齐方向
            # ),
            opacity=0.6,
            hoverinfo='text',
            text=row['text'],
            name=""
        ))

    # 提取所有唯一的城市坐标
    coord_df = od_data[['O', 'O_lat', 'O_lon']].rename(columns={'O': 'city'})
    coord_df = pd.concat([
        coord_df,
        od_data[['D', 'D_lat', 'D_lon']].rename(columns={'D': 'city'})
    ]).drop_duplicates(subset=['city']).reset_index(drop=True)

    # 使用 concat 替代 append
    all_lats = pd.concat(
        [coord_df['O_lat'], coord_df['D_lat']], ignore_index=True)
    all_lons = pd.concat(
        [coord_df['O_lon'], coord_df['D_lon']], ignore_index=True)
    all_cities = coord_df['city']

    # 添加城市点标记（可选）
    fig.add_trace(go.Scattergeo(
        lat=all_lats,
        lon=all_lons,
        mode='markers+text',
        marker=dict(size=6, color='red'),
        text=all_cities,
        textposition="top center",
        textfont=dict(size=16),
        name='城市位置'
    ))

    # 设置地图样式
    fig.update_geos(
        scope="asia",
        resolution=50,
        showcountries=True,
        countrywidth=0.5,
        coastlinecolor="Black",
        landcolor="rgb(240, 240, 240)",
        fitbounds="locations"
    )

    # 布局美化
    fig.update_layout(
        title="出行期望线图（Desire Line）",
        title_x=0.0,
        showlegend=False,
        margin={"r": 0, "t": 40, "l": 0, "b": 0},
        height=800,
        width=1000,
        geo=dict(
            projection_scale=100,  # 根据数据自动缩放
            center=dict(lat=(od_data['O_lat'].mean() + od_data['D_lat'].mean())/2,
                        lon=(od_data['O_lon'].mean() + od_data['D_lon'].mean())/2)
        ),
        hoverlabel=dict(
            font_size=16,   # 设置悬停提示框字体大小
            font_family="Arial"
        )
    )

    return fig


# 初始化 session_state
if 'recording' not in st.session_state:
    st.session_state.recording = False
if 'audio_b64' not in st.session_state:
    st.session_state.audio_b64 = None


# 主界面布局
col1, col2 = st.columns([2, 2])
with col1:
    st.header("❓ 提问区")
    st.subheader("🎤 语音输入")
    
    # 显示已录制的音频
    if st.session_state.audio_b64:
        try:
            audio_bytes = base64.b64decode(st.session_state.audio_b64.split(',')[1])
            st.audio(audio_bytes, format="audio/wav")
        except Exception:
            st.warning("音频数据格式异常")
    
    # 布局按钮
    col_rec1, col_rec2 = st.columns(2)
    
    with col_rec1:
        if not st.session_state.recording:
            if st.button("开始录音"):
                # 嵌入录音 HTML + JS
                components.html("""
                <script>
                    let mediaRecorder;
                    let chunks = [];
    
                    // 开始录音
                    function startRecording() {
                        navigator.mediaDevices.getUserMedia({ audio: true })
                            .then(stream => {
                                mediaRecorder = new MediaRecorder(stream);
                                mediaRecorder.ondataavailable = e => {
                                    chunks.push(e.data);
                                };
                                mediaRecorder.onstop = () => {
                                    const blob = new Blob(chunks, { type: 'audio/wav' });
                                    const reader = new FileReader();
                                    reader.onloadend = () => {
                                        const base64data = reader.result;
                                        window.parent.postMessage(base64data, '*');
                                    };
                                    reader.readAsDataURL(blob);
                                };
                                mediaRecorder.start();
                            });
                    }
    
                    // 停止录音
                    function stopRecording() {
                        if (mediaRecorder && mediaRecorder.state === "recording") {
                            mediaRecorder.stop();
                            chunks = [];
                        }
                    }
    
                    // 页面加载时绑定事件
                    document.addEventListener('DOMContentLoaded', () => {
                        window.addEventListener('stopRecording', stopRecording);
                    });
    
                    // 自动开始录音
                    startRecording();
                </script>
                """, height=0)
                st.session_state.recording = True
                st.rerun()
    
        if st.session_state.recording:
            if st.button("停止录音"):
                # 触发前端停止录音
                components.html("""
                <script>
                    window.dispatchEvent(new Event('stopRecording'));
                </script>
                """, height=0)
                st.session_state.recording = False
                st.rerun()
            st.success("正在录音...")
    
    # 识别语音按钮
    with col_rec2:
        if st.button("识别语音", disabled=not bool(st.session_state.audio_b64)):
            if st.session_state.audio_b64:
                with st.spinner("正在识别语音..."):
                    try:
                        audio_bytes = base64.b64decode(st.session_state.audio_b64.split(',')[1])
                        # 示例：调用语音识别函数
                        # transcribed_text = transcribe_audio(audio_bytes)
                        # st.session_state.transcribed_text = transcribed_text
                        st.success("语音识别成功！")
                    except Exception as e:
                        st.error(f"识别失败: {e}")
            else:
                st.warning("请先录制音频")
    
    # 接收前端发来的 Base64 数据并写入 session_state
    components.html("""
    <script>
        window.addEventListener("message", event => {
            if (event.data.startsWith("data:audio/wav")) {
                // 将 Base64 音频数据通过 URL 参数传回 Streamlit
                const url = new URL(window.parent.location);
                url.searchParams.set('audio', event.data);
                window.parent.location = url;
            }
        });
    </script>
    """, height=0)
    
    # ✅ 使用 st.query_params 读取音频数据
    query_params = st.query_params
    if 'audio' in query_params:
        audio_data = query_params['audio']
        if isinstance(audio_data, list):
            audio_data = audio_data[0]
        st.session_state.audio_b64 = audio_data
        # 清除参数（使用 st.query_params.set）
        new_params = {k: v for k, v in query_params.items() if k != 'audio'}
        st.query_params.clear()
        st.query_params.update(new_params)

    # 模式切换按钮（新增“期望线”选项）
    modes = ["分析问答", "地图可视化", "期望线"]

    # 设置默认选中项（分析问答）
    default_index = modes.index(
        st.session_state.mode) if st.session_state.mode in modes else 0

    mode = st.radio("选择模式", modes, index=default_index)
    st.session_state.mode = mode

    if mode == "地图可视化":
        # 问题输入框
        question = st.text_area(
            "输入您的问题（注意表头名称一定要准确）:",
            height=200,
            placeholder="示例问题:\n- 按照断面编号匹配路段，展示自然合计流量\n- ……",
            key="question_input",
            value=st.session_state.transcribed_text
        )

        if st.button("提交问题"):
            if 'full_data' not in st.session_state or st.session_state.full_data.empty:
                st.error("请先上传数据")
            else:
                df_user = st.session_state.full_data
                client = init_deepseek()
                match_field, value_field = smart_process_map_question(
                    question, df_user.columns.tolist(), client)

                if match_field in df_user.columns and value_field in df_user.columns:
                    gdf = load_gdf()

                    # 使用映射方式绑定 value_field 到 ROADNO 上，并填充 0
                    full_data_with_values = map_values_to_roadno(
                        gdf, df_user, match_field, value_field)

                    st.session_state.matched_data = full_data_with_values
                    st.session_state.map_ready = True
                    st.session_state.value_field = value_field
                    st.rerun()
                else:
                    st.warning(
                        f"无法找到字段 '{match_field}' 或 '{value_field}'，请检查问题描述是否准确。")

    elif mode == "分析问答":
        # 问题输入框
        question = st.text_area(
            "输入您的问题（注意表头名称一定要准确）:",
            height=200,
            placeholder="示例问题:\n- 计算合计流量\n- 找出流量最大/最小的前五个站\n- 选取2025年2月10号到3月9号绘制热力图\n- 按地市聚合下不同OD的自然合计流量\n- ……",
            key="question_input",
            value=st.session_state.transcribed_text
        )

        # 处理流程
        if st.button("提交问题"):
            if uploaded_file and question:
                with st.spinner("分析中..."):
                    try:
                        # 直接从session_state获取完整数据
                        if st.session_state.full_data is not None:
                            client = init_deepseek()
                            result, raw_code, clean_code, fig = process_question(
                                st.session_state.full_data, question, client)
                            final_code_placeholder = st.code(
                                clean_code, language="python")
                            # 存储结果到session state
                            st.session_state.result = result
                            st.session_state.raw_code = raw_code
                            st.session_state.clean_code = clean_code
                            st.session_state.fig = fig  # 存储图表对象
                    except Exception as e:
                        st.error(f"系统错误: {str(e)}")
            elif not uploaded_file:
                st.info("请先上传Excel文件")
            else:
                st.info("请输入问题")

    elif mode == "期望线":
        question = st.text_area(
            "输入您的问题（注意表头名称一定要准确）:",
            height=200,
            placeholder="示例问题:\n- 按地市画期望线\n- 筛选流量大于三万的地市画图\n- ……",
            key="question_input",
            value=st.session_state.transcribed_text
        )

        if st.button("提交问题"):
            if uploaded_file and question:
                with st.spinner("分析中..."):
                    try:
                        od_data = st.session_state.full_data
                        client = init_deepseek()
                        result_df, clean_code = process_od_question(
                            od_data, question, client)
                        if isinstance(result_df, pd.DataFrame):
                            fig = generate_od_lines(result_df)
                            st.session_state.od_fig = fig
                            st.session_state.result = result_df
                            st.success("期望线图已生成！")
                        else:
                            st.error(result_df)
                    except Exception as e:
                        st.error(f"系统错误: {str(e)}")
            elif not uploaded_file:
                st.info("请先上传Excel文件")
            else:
                st.info("请输入问题")

with col2:
    st.header("💡 回答区")

    # 展示分析结果 DataFrame 或文本输出
    if 'result' in st.session_state:
        result = st.session_state.result
        if isinstance(result, pd.DataFrame):
            st.dataframe(result)
        else:
            st.code(str(result), language="python")

    # 获取当前模式
    current_mode = st.session_state.mode if 'mode' in st.session_state else None

    # 根据不同模式展示不同的图表或地图
    if current_mode == "分析问答":
        if 'fig' in st.session_state and st.session_state.fig is not None:
            st.pyplot(st.session_state.fig)

    elif current_mode == "期望线":
        if 'od_fig' in st.session_state and st.session_state.od_fig is not None:
            st.plotly_chart(st.session_state.od_fig, use_container_width=True)

    elif current_mode == "地图可视化":
        if st.session_state.map_ready and 'matched_data' in st.session_state:
            full_gdf = load_gdf()  # 加载完整路网数据用于底图
            m = render_map(full_gdf, st.session_state.matched_data,
                           st.session_state.value_field)
            folium_static(m, width=1200, height=800)
        else:
            st.info("请点击【提交问题】按钮查看结果")
