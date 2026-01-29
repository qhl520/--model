import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar

# ==========================================
# 1. 设置与数据读取
# ==========================================
excel_file = '网络构建及博弈均衡结果.xlsx'

print(f"正在读取文件: {excel_file} ...")
try:
    df_network = pd.read_excel(excel_file, sheet_name='网络结构')
    df_ranking = pd.read_excel(excel_file, sheet_name='重要性排序')
    print("✅ 数据读取成功！")
except FileNotFoundError:
    print("❌ 错误：找不到文件，请确保 Excel 文件与代码在同一目录下。")
    exit()

# 数据预处理
nodes_data = {}
for idx, row in df_network.iterrows():
    nid = int(row['节点ID'])
    nodes_data[nid] = {
        'e': row['基准直接排放量(e)'],
        'r': row['净流出值(r)'],
        'mu': row['碳边际减排成本μ'],
        'eta': row['其他企业所产生的边际减排效益η'],
        'alpha': row['讨价还价α']
    }

sorted_node_ids = df_ranking['节点ID'].tolist()
I_param = 0.2
E_total_initial = 2250

# ==========================================
# 2. 核心算法函数
# ==========================================

def get_subset_data(active_ids):
    e_vec = np.array([nodes_data[i]['e'] for i in active_ids])
    r_vec = np.array([nodes_data[i]['r'] for i in active_ids])
    mu_vec = np.array([nodes_data[i]['mu'] for i in active_ids])
    return e_vec, r_vec, mu_vec

def solve_lower_level(active_ids, subsidized_node_id, subsidy_amount, E_limit):
    """求解市场均衡：返回 q向量 和 碳价theta"""
    n = len(active_ids)
    e_vec, r_vec, mu_vec = get_subset_data(active_ids)
    
    sub_idx = -1
    if subsidized_node_id is not None:
        try:
            sub_idx = active_ids.index(subsidized_node_id)
        except ValueError: pass

    def objective(q_vec):
        cost_sum = 0.0
        for i in range(n):
            term1 = mu_vec[i] * (q_vec[i] + r_vec[i])
            term2 = (I_param / 2.0) * (e_vec[i] - q_vec[i])**2
            cost = term1 + term2
            if i == sub_idx:
                e_key = e_vec[i]
                subsidy_term = (subsidy_amount / e_key) * (e_vec[i] - q_vec[i])
                cost -= subsidy_term
            cost_sum += cost
        return cost_sum

    sum_r = np.sum(r_vec)
    cons = ({'type': 'ineq', 'fun': lambda q: E_limit - sum_r - np.sum(q)}) 
    bnds = [(0.0, None) for _ in range(n)]
    
    x0 = e_vec.copy()
    res = minimize(objective, x0, method='SLSQP', bounds=bnds, constraints=cons, tol=1e-8)
    q_opt = res.x
    
    # 计算 Theta
    thetas = []
    for i in range(n):
        mc = mu_vec[i] - I_param * (e_vec[i] - q_opt[i])
        if i == sub_idx: mc += (subsidy_amount / e_vec[i])
        thetas.append(mc)
    theta_opt = np.mean(thetas)
    
    return q_opt, theta_opt

def solve_bargaining_round(S_available, E_available, key_node, active_ids, delta_val):
    node_params = nodes_data[key_node]
    alpha_val = node_params['alpha']
    eta = node_params['eta']
    e_key = node_params['e']
    
    def nash_objective(gamma):
        if gamma <= 0.001 or gamma >= 0.999: return 1e9
        subsidy_amt = gamma * S_available
        q_opt, _ = solve_lower_level(active_ids, key_node, subsidy_amt, E_available)
        k_idx = active_ids.index(key_node)
        q_key = q_opt[k_idx]
        gain_firm = (subsidy_amt / e_key) * (e_key - q_key)
        term_gov = delta_val * gamma + eta * (1 - gamma) - gamma
        gain_gov = S_available * term_gov
        if gain_firm <= 1e-6 or gain_gov <= 1e-6: return 1e9
        obj = (gain_firm ** alpha_val) * (gain_gov ** (1 - alpha_val))
        return -obj 

    res = minimize_scalar(nash_objective, bounds=(0.001, 0.999), method='bounded')
    return res.x if res.fun != 1e9 else 0.0

# ==========================================
# 3. 详细模拟与导出逻辑
# ==========================================

def run_simulation_full_details(case_name, S_init, delta_val):
    print(f"\n{'='*20} 正在计算: {case_name} {'='*20}")
    
    current_S = S_init
    current_E = E_total_initial
    active_nodes = list(df_network['节点ID'])
    candidate_queue = sorted_node_ids.copy()
    
    summary_records = []
    detail_records = []
    
    round_num = 0
    while len(candidate_queue) > 0:
        if current_S < 0.05: break
            
        round_num += 1
        key_node = candidate_queue.pop(0)
        
        # 1. 计算本轮博弈
        gamma = solve_bargaining_round(current_S, current_E, key_node, active_nodes, delta_val)
        subsidy_given = gamma * current_S
        q_vec, theta = solve_lower_level(active_nodes, key_node, subsidy_given, current_E)
        
        # 2. 提取关键信息
        k_idx = active_nodes.index(key_node)
        q_key = q_vec[k_idx]
        r_key = nodes_data[key_node]['r']
        
        # 3. 记录汇总表
        summary_records.append({
            '轮次': round_num,
            '关键企业ID': key_node,
            '初始补贴S': current_S,
            '初始配额E': current_E,
            'Gamma': gamma,
            '关键企业获补': subsidy_given,
            '关键企业决策排放(q)': q_key,
            '市场碳价': theta
        })
        
        # 4. 记录明细表 (遍历所有企业)
        for i, nid in enumerate(active_nodes):
            q_val = q_vec[i]
            is_key = (nid == key_node)
            subsidy_received = subsidy_given if is_key else 0.0
            
            detail_records.append({
                '轮次': round_num,
                '企业ID': nid,
                '角色': '关键企业' if is_key else '普通企业',
                '初始排放e': nodes_data[nid]['e'],
                '净流出r': nodes_data[nid]['r'],
                '决策排放q': q_val,
                '获得补贴': subsidy_received,
                '本轮碳价': theta
            })
            
        # 更新状态
        current_S *= (1.0 - gamma)
        current_E -= (q_key + r_key)
        active_nodes.remove(key_node)
        
    return pd.DataFrame(summary_records), pd.DataFrame(detail_records)

# ==========================================
# 4. 执行并保存文件
# ==========================================

# 您可以取消注释运行想要的 Case
df_sum1, df_det1 = run_simulation_full_details("Case 1 (强激励)", 150, 2)
with pd.ExcelWriter("Case1_全流程结果.xlsx") as writer:
    df_sum1.to_excel(writer, sheet_name="汇总信息", index=False)
    df_det1.to_excel(writer, sheet_name="所有轮次明细", index=False)

df_sum3, df_det3 = run_simulation_full_details("Case 3 (弱激励)", 50, 1)
with pd.ExcelWriter("Case3_全流程结果.xlsx") as writer:
    df_sum3.to_excel(writer, sheet_name="汇总信息", index=False)
    df_det3.to_excel(writer, sheet_name="所有轮次明细", index=False)

print("\n✅ 计算完成！结果已保存到 Excel 文件中。")
print("请打开 Excel 查看 '所有轮次明细' Sheet 获取每一轮的详细数据。")