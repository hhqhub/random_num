import streamlit as st
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# 设置页面配置
st.set_page_config(
    page_title="平整度随机数生成器",
    page_icon="🎲",
    layout="wide"
)

st.title("🎲 平整度随机数生成器")

# ====================  ====================

def calculate_possible_rates(N, L, R, allow=False):
    """
    计算在给定范围内所有可能的合格率
    """
    min_qualified = int(L * N) + 1
    max_qualified = int(R * N)
    
    if not allow:
        min_unqualified = N // 10
        max_possible_qualified = N - min_unqualified
        actual_max_qualified = min(max_qualified, max_possible_qualified)
    else:
        actual_max_qualified = max_qualified
    
    if min_qualified > actual_max_qualified:
        return None, "合格率区间设置不合理，无法生成符合条件的随机数"
    
    possible_rates = []
    for q in range(min_qualified, actual_max_qualified + 1):
        rate = q / N
        possible_rates.append((q, rate))
    
    return possible_rates, None


def random_num_generator(max_value=10, qualified_count=None, delta=5, N=100, allow=False):
    """
    生成符合条件的随机数（返回结果而不是打印）
    """
    unqualified_count = N - qualified_count
    half = N // 2
    cross_groups = half // 5
    
    # 为每个跨部分组分配不合格数
    if not allow:
        group_unqualified = [1 for _ in range(cross_groups)]
        remaining_unqualified = unqualified_count - cross_groups
        
        if remaining_unqualified < 0:
            return None, f"严格模式下，至少需要{cross_groups}个不合格数，但只有{unqualified_count}个"
        
        for _ in range(remaining_unqualified):
            group_idx = random.randint(0, cross_groups - 1)
            group_unqualified[group_idx] += 1
    else:
        group_unqualified = [0 for _ in range(cross_groups)]
        for _ in range(unqualified_count):
            group_idx = random.randint(0, cross_groups - 1)
            if group_unqualified[group_idx] < 10:
                group_unqualified[group_idx] += 1
            else:
                available_groups = [i for i in range(cross_groups) if group_unqualified[i] < 10]
                if available_groups:
                    group_idx = random.choice(available_groups)
                    group_unqualified[group_idx] += 1
    
    # 初始化两部分数据
    part1 = []
    part2 = []
    
    # 为每个跨部分组生成数据
    for i in range(cross_groups):
        unq_count = group_unqualified[i]
        q_count = 10 - unq_count
        
        qualified_numbers = [round(random.uniform(0.1, 4.0), 1) for _ in range(q_count)]
        unqualified_numbers = [round(random.uniform(4.1, max_value), 1) for _ in range(unq_count)]
        
        group_10_numbers = qualified_numbers + unqualified_numbers
        random.shuffle(group_10_numbers)
        
        part1.extend(group_10_numbers[:5])
        part2.extend(group_10_numbers[5:])
    
    # 修复连续相同
    def fix_consecutive_same(arr):
        for i in range(1, len(arr)):
            attempts = 0
            while arr[i] == arr[i-1] and attempts < 1000:
                is_qualified = arr[i] <= 4.0
                prev_num = arr[i-1]
                next_num = arr[i+1] if i < len(arr)-1 else None
                
                if is_qualified:
                    new_num = round(random.uniform(0.1, 4.0), 1)
                else:
                    new_num = round(random.uniform(4.1, max_value), 1)
                
                if new_num != prev_num and (next_num is None or new_num != next_num):
                    arr[i] = new_num
                    break
                
                attempts += 1
                
                if attempts >= 1000:
                    while True:
                        if is_qualified:
                            new_num = round(random.uniform(0.1, 4.0), 1)
                        else:
                            new_num = round(random.uniform(4.1, max_value), 1)
                        
                        if new_num != prev_num:
                            arr[i] = new_num
                            break
    
    fix_consecutive_same(part1)
    fix_consecutive_same(part2)
    
    # 调整delta约束
    def adjust_delta_constraint(arr, delta, max_value):
        arr_len = len(arr)
        max_adjustments = 100
        for i in range(0, arr_len, 5):
            if i + 5 > arr_len:
                break
            group = arr[i:i+5]
            max_val = max(group)
            min_val = min(group)
            adjustments = 0
            
            while max_val - min_val > delta and adjustments < max_adjustments:
                idx = random.randint(0, 4)
                pos = i + idx
                current_num = arr[pos]
                prev_num = arr[pos-1] if pos > 0 else None
                next_num = arr[pos+1] if pos < arr_len-1 else None
                
                is_qualified = current_num <= 4.0
                
                if is_qualified:
                    new_min = max(0.1, min_val)
                    new_max = min(4.0, max_val)
                    if current_num == max_val:
                        new_max = min(new_max, max_val - 0.1)
                    elif current_num == min_val:
                        new_min = max(new_min, min_val + 0.1)
                else:
                    new_min = max(4.1, min_val)
                    new_max = min(max_value, max_val)
                    if current_num == max_val:
                        new_max = min(new_max, max_val - 0.1)
                    elif current_num == min_val:
                        new_min = max(new_min, min_val + 0.1)
                
                if new_min > new_max:
                    if is_qualified:
                        new_min, new_max = 0.1, 4.0
                    else:
                        new_min, new_max = 4.1, max_value
                
                new_num = None
                attempts = 0
                while (new_num is None or new_num == prev_num or new_num == next_num) and attempts < 100:
                    new_num = round(random.uniform(new_min, new_max), 1)
                    attempts += 1
                
                if attempts >= 100:
                    if is_qualified:
                        new_num = round(random.uniform(0.1, 4.0), 1)
                    else:
                        new_num = round(random.uniform(4.1, max_value), 1)
                
                arr[pos] = new_num
                group = arr[i:i+5]
                max_val = max(group)
                min_val = min(group)
                adjustments += 1
    
    adjust_delta_constraint(part1, delta, max_value)
    adjust_delta_constraint(part2, delta, max_value)
    fix_consecutive_same(part1)
    fix_consecutive_same(part2)
    
    numbers = part1 + part2
    
    # 区间统计
    counts = {
        "［0,1］": 0, "(1,2］": 0, "(2,3]": 0, "(3,4]": 0,
        "(4,5]": 0, "(5,6]": 0, "(6,+∞)": 0
    }
    for num in numbers:
        if 0 <= num <= 1:
            counts["［0,1］"] += 1
        elif 1 < num <= 2:
            counts["(1,2］"] += 1
        elif 2 < num <= 3:
            counts["(2,3]"] += 1
        elif 3 < num <= 4:
            counts["(3,4]"] += 1
        elif 4 < num <= 5:
            counts["(4,5]"] += 1
        elif 5 < num <= 6:
            counts["(5,6]"] += 1
        elif num > 6:
            counts["(6,+∞)"] += 1
    
    # 跨部分分组统计
    group_stats = []
    group_size = 5
    total_groups = len(part1) // group_size
    
    for i in range(total_groups):
        part1_group = part1[i*group_size : (i+1)*group_size]
        part2_group = part2[i*group_size : (i+1)*group_size]
        combined_group = part1_group + part2_group
        
        total = len(combined_group)
        qualified = sum(1 for num in combined_group if num <= 4.0)
        unqualified = total - qualified
        over_6 = sum(1 for num in combined_group if num > 6.0)
        
        group_stats.append({
            "组号": i + 1,
            "统计个数": total,
            "合格个数": qualified,
            "不合格个数": unqualified,
            "大于6个数": over_6
        })
    
    result = {
        "part1": part1,
        "part2": part2,
        "numbers": numbers,
        "counts": counts,
        "group_stats": group_stats,
        "max_value": max(numbers),
        "min_value": min(numbers),
        "qualified_count": qualified_count,
        "N": N
    }
    
    return result, None


def plot_histogram(counts, font_type='hei'):
    """
    绘制区间统计柱状图 - 原版格式
    """
    # 根据选择设置字体路径
    if font_type == 'song':
        font_path = r"C:/Windows/Fonts/simsun.ttc"
    elif font_type == 'hei':
        font_path = r"C:/Windows/Fonts/simhei.ttf"
    else:
        font_path = r"C:/Windows/Fonts/simhei.ttf"
    
    # 加载指定字体
    try:
        custom_font = FontProperties(fname=font_path, size=14)
    except:
        custom_font = FontProperties(size=14)
    
    # 从统计结果中拆解横坐标和对应数据
    categories = ['0-1', '1-2', '2-3', '3-4', '4-5', '5-6', '大于6']
    values = list(counts.values())

    # 创建画布
    fig, ax = plt.subplots(figsize=(10, 6))

    # 设置柱子粗细和颜色
    bar_width = 0.3
    bar_color = '#5B9BD5'
    
    # 绘制柱状图
    bars = ax.bar(categories, values, width=bar_width, color=bar_color, zorder=3)

    # 设置网格线在柱体下方
    ax.grid(axis='y', linestyle='-', linewidth=1, color='lightgray', zorder=0)

    # 设置坐标轴标签
    ax.set_xlabel('平整度区间', fontproperties=custom_font)
    ax.set_ylabel('下尺数', fontproperties=custom_font)

    # 设置横坐标刻度位置和标签
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories, fontproperties=custom_font)

    # 处理纵坐标刻度
    max_val = max(values) if max(values) > 0 else 2
    y_upper = max_val if max_val % 2 == 0 else max_val + 1
    ax.set_ylim(0, y_upper)
    y_ticks = np.arange(0, y_upper + 1, 2)
    
    # 设置纵坐标刻度位置和标签
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_ticks.astype(int), fontproperties=custom_font)

    # 优化布局
    plt.tight_layout()
    
    return fig


# ==================== Streamlit 界面 ====================

# 侧边栏 - 参数设置
st.sidebar.header("⚙️ 参数设置")

# 模式选择
mode = st.sidebar.radio(
    "模式选择",
    ["🔓 宽松模式（允许某些组全部合格）", "🔒 严格模式（每组至少1个不合格）"],
    index=0
)
allow = "宽松" in mode

# 参数输入
N = st.sidebar.number_input("随机数数量 N（10的倍数）", min_value=10, max_value=1000, value=60, step=10)
L = st.sidebar.number_input("合格率左区间 L", min_value=0.0, max_value=1.0, value=0.80, step=0.01)
R = st.sidebar.number_input("合格率右区间 R", min_value=0.0, max_value=1.0, value=0.90, step=0.01)
max_value = st.sidebar.number_input("随机数最大值", min_value=4.1, max_value=20.0, value=5.7, step=0.1)
delta = st.sidebar.number_input("每5个数最大差值 Delta", min_value=0.0, max_value=20.0, value=5.9, step=0.1)

# 字体选择
font_type = st.sidebar.selectbox("柱状图字体", ["hei", "song"], index=0, 
                                  format_func=lambda x: "黑体" if x == "hei" else "宋体")

# 验证N
if N % 10 != 0:
    st.sidebar.error("❌ N必须是10的倍数！")
    st.stop()

if L >= R:
    st.sidebar.error("❌ 左区间必须小于右区间！")
    st.stop()

# 步骤1: 计算可能的合格率
st.header("📊 步骤1: 可能的合格率")

possible_rates, error = calculate_possible_rates(N, L, R, allow)

if error:
    st.error(f"❌ {error}")
    st.stop()

st.success(f"✅ 在合格率范围 ({L}, {R}) 内，N={N}，共有 **{len(possible_rates)}** 种可能的合格率")

# 创建合格率表格
rates_df = pd.DataFrame([
    {
        "序号": idx + 1,
        "合格数": q,
        "不合格数": N - q,
        "合格率(小数)": f"{rate:.6f}",
        "合格率(百分比)": f"{rate:.2%}"
    }
    for idx, (q, rate) in enumerate(possible_rates)
])

st.dataframe(rates_df, use_container_width=True, hide_index=True)

# 步骤2: 选择合格率
st.header("🎯 步骤2: 选择合格率")

# 使用下拉框选择
rate_options = [f"{rate:.6f} ({rate:.2%}) - 合格数: {q}" for q, rate in possible_rates]
selected_option = st.selectbox("选择合格率", rate_options)

# 解析选择
selected_idx = rate_options.index(selected_option)
selected_qualified_count, selected_rate = possible_rates[selected_idx]

st.info(f"📌 已选择: 合格率 **{selected_rate:.2%}**，合格数 **{selected_qualified_count}**，不合格数 **{N - selected_qualified_count}**")

# 步骤3: 生成随机数
st.header("🎲 步骤3: 生成随机数")

col1, col2 = st.columns([1, 4])
with col1:
    generate_btn = st.button("🎲 生成随机数", type="primary", use_container_width=True)

# 使用session_state保存生成的结果
if 'result' not in st.session_state:
    st.session_state.result = None

if generate_btn:
    result, error = random_num_generator(
        max_value=max_value,
        qualified_count=selected_qualified_count,
        delta=delta,
        N=N,
        allow=allow
    )
    
    if error:
        st.error(f"❌ {error}")
    else:
        st.session_state.result = result
        st.session_state.font_type = font_type

# 步骤4: 显示结果
if st.session_state.result:
    result = st.session_state.result
    
    st.header("📈 步骤4: 生成结果")
    
    # 统计卡片
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("总数量", result["N"])
    col2.metric("合格数", result["qualified_count"])
    col3.metric("不合格数", result["N"] - result["qualified_count"])
    col4.metric("合格率", f"{result['qualified_count']/result['N']:.2%}")
    col5.metric("最小值", f"{result['min_value']:.1f}")
    col6.metric("最大值", f"{result['max_value']:.1f}")
    
    # 数据详情
    st.subheader("📋 数据详情")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**第一部分 (1/2)**")
        part1_df = pd.DataFrame([
            {"序号": i+1, "数值": f"{num:.1f}", "是否合格": "✅ 是" if num <= 4.0 else "❌ 否"}
            for i, num in enumerate(result["part1"])
        ])
        st.dataframe(part1_df, use_container_width=True, hide_index=True, height=400)
    
    with col2:
        st.write("**第二部分 (2/2)**")
        part2_df = pd.DataFrame([
            {"序号": i+1, "数值": f"{num:.1f}", "是否合格": "✅ 是" if num <= 4.0 else "❌ 否"}
            for i, num in enumerate(result["part2"])
        ])
        st.dataframe(part2_df, use_container_width=True, hide_index=True, height=400)
    
    # 使用原版matplotlib柱状图
    st.subheader("📊 区间统计图")
    fig = plot_histogram(result["counts"], font_type=st.session_state.get('font_type', 'hei'))
    st.pyplot(fig)
    plt.close(fig)
    
    # 分组统计
    st.subheader("📈 跨部分分组统计")
    
    group_df = pd.DataFrame(result["group_stats"])
    
    # 添加合计行
    total_row = {
        "组号": "合计",
        "统计个数": sum(g["统计个数"] for g in result["group_stats"]),
        "合格个数": sum(g["合格个数"] for g in result["group_stats"]),
        "不合格个数": sum(g["不合格个数"] for g in result["group_stats"]),
        "大于6个数": sum(g["大于6个数"] for g in result["group_stats"])
    }
    group_df = pd.concat([group_df, pd.DataFrame([total_row])], ignore_index=True)
    
    st.dataframe(group_df, use_container_width=True, hide_index=True)
    
    # ==================== 导出功能（与源代码格式一致）====================
    st.subheader("📤 导出数据")
    
    # 生成与源代码完全一致的导出文本
    export_lines = []
    
    # 统计信息
    export_lines.append(f"所有随机数统计：")
    export_lines.append(f"最大值：{result['max_value']}")
    export_lines.append(f"最小值：{result['min_value']}")
    export_lines.append("")
    
    # 第一部分
    export_lines.append(f"第一部分（1/2）：")
    for num in result["part1"]:
        export_lines.append(f"{num}\t{'是' if num <= 4.0 else '否'}")
    
    export_lines.append("")
    export_lines.append("")
    
    # 第二部分
    export_lines.append(f"第二部分（2/2）：")
    for num in result["part2"]:
        export_lines.append(f"{num}\t{'是' if num <= 4.0 else '否'}")
    
    export_lines.append("")
    export_lines.append("")
    
    # 最大随机数和合格率
    export_lines.append(f"最大随机数：{result['max_value']}")
    export_lines.append(f"合格率为  {result['qualified_count']/result['N']:.2%}")
    export_lines.append("")
    
    # 跨部分分组统计
    export_lines.append("跨部分分组统计结果：")
    export_lines.append(f"统计个数\t合格个数\t不合格个数\t大于6个数")
    
    total_all = 0
    qualified_all = 0
    unqualified_all = 0
    over_6_all = 0
    
    for g in result["group_stats"]:
        export_lines.append(f"{g['统计个数']}\t{g['合格个数']}\t{g['不合格个数']}\t{g['大于6个数']}")
        total_all += g['统计个数']
        qualified_all += g['合格个数']
        unqualified_all += g['不合格个数']
        over_6_all += g['大于6个数']
    
    # 合计行
    export_lines.append(f"{total_all}\t{qualified_all}\t{unqualified_all}\t{over_6_all}")
    
    export_text = "\n".join(export_lines)
    
    # 大文本框显示导出内容
    st.text_area(
        "导出内容", 
        export_text, 
        height=500,  # 更大的高度
        key="export_area"
    )
    
    # 下载按钮
    col1, col2 = st.columns([1, 4])
    with col1:
        st.download_button(
            label="📥 下载为TXT文件",
            data=export_text,
            file_name="random_numbers.txt",
            mime="text/plain",
            use_container_width=True
        )