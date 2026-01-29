import networkx as nx
import random
import numpy as np
import pandas as pd

# ================= 1. 全局配置 =================
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# 参数配置 (完全对应你的分层随机逻辑)
CONFIG = {
    "上游": {
        "n": 15, "layer": 0, 
        "net_in": (0, 2), "net_out": (5, 8),   
        "e_mean": 112.5, "e_sigma": 5.0,       # 基准排放
        "a_range": (5, 15),                    # 碳流入量范围
        "b_range": (100, 150)                  # 碳流出量范围
    },
    "中游": {
        "n": 11, "layer": 1,
        "net_in": (5, 8), "net_out": (3, 5),
        "e_mean": 75.0, "e_sigma": 5.0,
        "a_range": (40, 70),
        "b_range": (60, 90)
    },
    "下游": {
        "n": 4, "layer": 2,
        "net_in": (12, 16), "net_out": (0, 1),
        "e_mean": 37.5, "e_sigma": 5.0,
        "a_range": (120, 180),
        "b_range": (10, 30)
    }
}

# ================= 2. 核心函数 =================

def get_stratified_random_value(degree, all_degrees, val_range):
    """
    分层随机抽取算法：
    根据度数在群体中的排位，锁定一个小区间，然后在区间内随机。
    """
    min_deg, max_deg = min(all_degrees), max(all_degrees)
    val_min, val_max = val_range
    
    # 特殊情况：如果该组所有节点度数都一样，直接全范围随机
    if max_deg == min_deg:
        return random.uniform(val_min, val_max)
    
    # 1. 计算层级总数 (如度数 5,6,7,8 -> 4层)
    degree_span = max_deg - min_deg + 1
    
    # 2. 计算每层对应的数值宽度
    total_val_span = val_max - val_min
    slice_width = total_val_span / degree_span
    
    # 3. 确定当前度数对应的“小区间”
    layer_index = degree - min_deg
    current_slice_min = val_min + (layer_index * slice_width)
    current_slice_max = current_slice_min + slice_width
    
    # 4. 区间内随机
    return random.uniform(current_slice_min, current_slice_max)

def generate_graph():
    """生成网络结构"""
    nodes = []
    idx = 1
    for key, cfg in CONFIG.items():
        for _ in range(cfg["n"]):
            nodes.append({
                "id": idx, "type": key, "layer": cfg["layer"], "limit": cfg["net_in"],
                "in_d": random.randint(*cfg["net_in"]), 
                "out_d": random.randint(*cfg["net_out"])
            })
            idx += 1
    
    # 简单配平
    diff = sum(n["out_d"] for n in nodes) - sum(n["in_d"] for n in nodes)
    while diff != 0:
        target = random.choice(nodes)
        if diff > 0 and target["in_d"] < target["limit"][1]:
            target["in_d"] += 1; diff -= 1
        elif diff < 0 and target["in_d"] > target["limit"][0]:
            target["in_d"] -= 1; diff += 1
            
    G = nx.DiGraph()
    out_stubs, in_stubs = [], []
    for n in nodes:
        G.add_node(n["id"], type=n["type"], layer=n["layer"])
        out_stubs.extend([n["id"]] * n["out_d"])
        in_stubs.extend([n["id"]] * n["in_d"])
    
    random.shuffle(in_stubs)
    edges = { (u, v) for u, v in zip(out_stubs, in_stubs) if u != v }
    G.add_edges_from(edges)
    return G

def calculate_full_table(G):
    """计算并生成你需要的完整检查表"""
    data_rows = []
    
    # 按类型分组处理，为了获取同组的度数范围
    for n_type, cfg in CONFIG.items():
        group_node_ids = [n for n, attr in G.nodes(data=True) if attr['type'] == n_type]
        group_in_degrees = [G.in_degree(n) for n in group_node_ids]
        group_out_degrees = [G.out_degree(n) for n in group_node_ids]
        
        for node in group_node_ids:
            # 1. 基础数据
            in_d = G.in_degree(node)
            out_d = G.out_degree(node)
            
            # 2. 计算基准排放 e (正态分布)
            e_val = max(0, np.random.normal(cfg["e_mean"], cfg["e_sigma"]))
            
            # 3. 计算碳流入 a (根据入度分层随机)
            a_val = get_stratified_random_value(in_d, group_in_degrees, cfg["a_range"])
            
            # 4. 计算碳流出 b (根据出度分层随机)
            b_val = get_stratified_random_value(out_d, group_out_degrees, cfg["b_range"])
            
            # 5. 计算净流出
            net_flow = a_val - b_val
            
            # 6. 存入行
            data_rows.append({
                "节点ID": node,
                "节点类型": n_type,
                "入度": in_d,
                "出度": out_d,
                "基准直接排放量(e)": round(e_val, 2),
                "碳流入量(a)": round(a_val, 2),
                "碳流出量(b)": round(b_val, 2),
                "净流出值": round(net_flow, 2)
            })
            
    # 转为 DataFrame 并按 ID 排序
    df = pd.DataFrame(data_rows)
    df = df.sort_values("节点ID").reset_index(drop=True)
    return df

# ================= 3. 主程序 =================
if __name__ == "__main__":
    # 1. 生成数据
    G = generate_graph()
    df_final = calculate_full_table(G)
    
    # 2. 设置 Pandas 显示选项 (确保显示所有行、列对齐)
    pd.set_option('display.max_rows', None)     # 显示所有行
    pd.set_option('display.max_columns', None)  # 显示所有列
    pd.set_option('display.width', 1000)        # 防止换行
    pd.set_option('display.unicode.ambiguous_as_wide', True)
    pd.set_option('display.unicode.east_asian_width', True)
    
    # 3. 打印完整表格用于检查
    print("="*80)
    print("【节点全指标检查表】")
    print("说明：")
    print("1. 请检查'入度'与'碳流入量(a)'的关系：同类型中，入度越大，a值应在越高的小区间浮动。")
    print("2. 请检查'出度'与'碳流出量(b)'的关系：同类型中，出度越大，b值应在越高的小区间浮动。")
    print("="*80)
    print(df_final)
    print("="*80)
    
    # 4. 保存文件
    df_final.to_csv("node_check_table.csv", index=False, encoding="utf-8-sig")
    print(f"表格已保存为: node_check_table.csv")