import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar

# ==========================================
# 1. ⚙️ 数据加载 (适配 Excel 文件)
# ==========================================

# 请确保 Excel 文件名与此处一致
excel_file_name = '网络构建及博弈均衡结果.xlsx'

print(f"正在读取文件: {excel_file_name} ...")

try:
    # 读取“网络结构”工作表
    df_network = pd.read_excel(excel_file_name, sheet_name='网络结构')
    
    # 读取“重要性排序”工作表
    df_ranking = pd.read_excel(excel_file_name, sheet_name='重要性排序')
    
    print("✅ 数据读取成功！")
    
except FileNotFoundError:
    print(f"❌ 错误：找不到文件 '{excel_file_name}'。请将代码和 Excel 文件放在同一个文件夹下。")
    exit()
except ValueError as e:
    print(f"❌ 错误：读取工作表失败。请检查 Excel 中是否包含名为 '网络结构' 和 '重要性排序' 的 Sheet。")
    print(f"系统报错信息: {e}")
    exit()
except ImportError:
    print("❌ 错误：缺少 openpyxl 库。请在终端运行: pip install openpyxl")
    exit()

# ==========================================
# 2. 🏗️ 数据预处理
# ==========================================

# 构建参数字典 (Key: Node ID)
nodes_data = {}
for idx, row in df_network.iterrows():
    nid = int(row['节点ID'])
    nodes_data[nid] = {
        'e': row['基准直接排放量(e)'],
        'r': row['净流出值(r)'],
        'mu': row['碳边际减排成本μ'],
        'eta': row['其他企业所产生的边际减排效益η'],
        'alpha': row['讨价还价α']  # 动态读取 alpha
    }

# 获取根据中心度排序的企业ID列表 (从“重要性排序”表中读取)
sorted_node_ids = df_ranking['节点ID'].tolist()

# 全局参数
I_param = 0.2
E_total_initial = 2250

# ==========================================
# 3. 🧠 核心博弈算法
# ==========================================

def get_subset_data(active_ids):
    e_vec = np.array([nodes_data[i]['e'] for i in active_ids])
    r_vec = np.array([nodes_data[i]['r'] for i in active_ids])
    mu_vec = np.array([nodes_data[i]['mu'] for i in active_ids])
    return e_vec, r_vec, mu_vec

def solve_lower_level(active_ids, subsidized_node_id, subsidy_amount, E_limit):
    """底层博弈：求解市场排放量 q 和 碳价 theta"""
    n = len(active_ids)
    e_vec, r_vec, mu_vec = get_subset_data(active_ids)
    
    # 找到获补企业索引
    sub_idx = -1
    if subsidized_node_id is not None:
        try:
            sub_idx = active_ids.index(subsidized_node_id)
        except ValueError: pass

    # 目标函数：最小化总成本
    def objective(q_vec):
        cost_sum = 0.0
        for i in range(n):
            # 基础成本
            term1 = mu_vec[i] * (q_vec[i] + r_vec[i])
            term2 = (I_param / 2.0) * (e_vec[i] - q_vec[i])**2
            cost = term1 + term2
            
            # 关键企业补贴抵扣
            if i == sub_idx:
                e_key = e_vec[i]
                subsidy_term = (subsidy_amount / e_key) * (e_vec[i] - q_vec[i])
                cost -= subsidy_term
            cost_sum += cost
        return cost_sum

    # 约束：总排放 <= 总配额
    sum_r = np.sum(r_vec)
    cons = ({'type': 'eq', 'fun': lambda q: E_limit - sum_r - np.sum(q)}) 
    bnds = [(0.0, None) for _ in range(n)]
    
    # 求解
    x0 = e_vec.copy()
    res = minimize(objective, x0, method='SLSQP', bounds=bnds, constraints=cons, tol=1e-8)
    q_opt = res.x
    
    # 计算影子价格 theta
    thetas = []
    for i in range(n):
        mc = mu_vec[i] - I_param * (e_vec[i] - q_opt[i])
        if i == sub_idx: mc += (subsidy_amount / e_vec[i])
        thetas.append(mc)
    theta_opt = np.mean(thetas)
    
    return q_opt, theta_opt

def solve_bargaining_round(S_available, E_available, key_node, active_ids, delta_val):
    """上层博弈：求解纳什议价 gamma"""
    node_params = nodes_data[key_node]
    alpha_val = node_params['alpha']
    eta = node_params['eta']
    e_key = node_params['e']
    
    def nash_objective(gamma):
        if gamma <= 0.001 or gamma >= 0.999: return 1e9
        subsidy_amt = gamma * S_available
        
        # 预测底层反应
        q_opt, _ = solve_lower_level(active_ids, key_node, subsidy_amt, E_available)
        k_idx = active_ids.index(key_node)
        q_key = q_opt[k_idx]
        
        # 计算效用
        gain_firm = (subsidy_amt / e_key) * (e_key - q_key)
        term_gov = delta_val * gamma + eta * (1 - gamma) - gamma
        gain_gov = S_available * term_gov
        
        if gain_firm <= 1e-6 or gain_gov <= 1e-6: return 1e9
        
        obj = (gain_firm ** alpha_val) * (gain_gov ** (1 - alpha_val))
        return -obj 

    res = minimize_scalar(nash_objective, bounds=(0.001, 0.999), method='bounded')
    return res.x if res.fun != 1e9 else 0.0

# ==========================================
# 4. 🚀 动态迭代主程序
# ==========================================

def run_simulation(case_name, S_init, delta_val):
    print(f"\n{'='*40}")
    print(f"启动模拟: {case_name}")
    print(f"初始 S={S_init}, δ={delta_val}")
    print(f"{'='*40}")
    
    current_S = S_init
    current_E = E_total_initial
    active_nodes = list(df_network['节点ID'])
    candidate_queue = sorted_node_ids.copy()
    
    history = []
    round_num = 0
    
    # 只要有钱且有企业，就一直循环
    while len(candidate_queue) > 0:
        if current_S < 0.05: 
            print(f"\n[停止] 资金不足 (S < 0.05)")
            break
            
        round_num += 1
        key_node = candidate_queue.pop(0)
        
        # 1. 纳什议价
        gamma = solve_bargaining_round(current_S, current_E, key_node, active_nodes, delta_val)
        
        # 2. 市场均衡
        subsidy_given = gamma * current_S
        q_vec, theta = solve_lower_level(active_nodes, key_node, subsidy_given, current_E)
        
        # 3. 记录与更新
        k_idx = active_nodes.index(key_node)
        q_key = q_vec[k_idx]
        r_key = nodes_data[key_node]['r']
        
        print(f"Round {round_num:02d} | Node {key_node} | γ={gamma:.4f} | 获补 {subsidy_given:.2f} | 剩余 {current_S*(1-gamma):.2f}")
        
        history.append({
            'Round': round_num, 'KeyNode': key_node, 'Gamma': gamma, 
            'SubsidyGiven': subsidy_given, 'q_key': q_key, 
            'S_Remaining': current_S * (1 - gamma),
            'E_Remaining': current_E - (q_key + r_key),
            'Theta': theta
        })
        
        current_S *= (1.0 - gamma)
        current_E -= (q_key + r_key)
        active_nodes.remove(key_node)
        
    return pd.DataFrame(history)

# ==========================================
# 5. 执行 Case 1 和 Case 3
# ==========================================

# Case 1 (强激励)
res1 = run_simulation("Case 1", 150, 2)
# 导出结果到 Excel
res1.to_excel('Case1_Result.xlsx', index=False)
print("结果已保存至 Case1_Result.xlsx")

# Case 3 (弱激励)
res3 = run_simulation("Case 3", 50, 1)
# 导出结果到 Excel
res3.to_excel('Case3_Result.xlsx', index=False)
print("结果已保存至 Case3_Result.xlsx")