import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# ---------- 论文级绘图风格设置 (Paper Style) ----------
config = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "SimHei"], # 英文 Times, 中文 SimHei
    "font.sans-serif": ["SimHei"],
    "mathtext.fontset": "stix",        # 数学公式字体
    "axes.unicode_minus": False,
    "axes.linewidth": 1.2,
    "xtick.major.width": 1.2,
    "ytick.major.width": 1.2,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
    "figure.dpi": 300
}
plt.rcParams.update(config)

# ---------- 文件与参数 ----------
# 确保使用您的 Excel 文件名
CASE_FILES = {
    "Case 1": "Case1_全流程结果.xlsx",
    "Case 2": "Case2_全流程结果.xlsx",
    "Case 3": "Case3_全流程结果.xlsx"
}
TOP_K = 10
I_PARAM = 0.2
OUTPUT_DIR = "figures_paper_comparison"  # 输出到新文件夹


# ---------- 成本函数 ----------
def cost_C(q, e, r, mu, subsidy, I):
    """
    成本函数
    subsidy: 获得的补贴额 (S_i)
    """
    # 注意：subsidy 在公式中是减项 -> - (s/e)*(e-q)
    return mu * (q + r) + 0.5 * I * (e - q) ** 2 - (subsidy / e) * (e - q)

def MC(q, e, r, mu, subsidy, I):
    """
    边际成本
    MC = mu - I(e-q) + s/e
    """
    return mu - I * (e - q) + subsidy / e


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 加载所有Case的数据
    all_case_data = {}
    for case_name, case_file in CASE_FILES.items():
        print(f"Reading Excel file: {case_file} ...")
        try:
            df_sum = pd.read_excel(case_file, sheet_name="汇总信息")
            df_det = pd.read_excel(case_file, sheet_name="所有轮次明细")
        except FileNotFoundError:
            print(f"Error: File {case_file} not found.")
            continue
        except ValueError as e:
            print(f"Error reading sheets: {e}")
            continue

        # 清理列名
        df_sum.columns = df_sum.columns.str.strip()
        df_det.columns = df_det.columns.str.strip()
        
        all_case_data[case_name] = {"sum": df_sum, "det": df_det}

    if not all_case_data:
        print("No case data loaded.")
        return

    # 获取第一个Case的关键企业列表（作为基准）
    first_case_name = list(all_case_data.keys())[0]
    df_sum = all_case_data[first_case_name]["sum"]
    
    # ---------- 筛选关键企业 ----------
    key_tasks = [] 
    seen_combinations = set()

    for idx, row in df_sum.iterrows():
        try:
            fid = int(row["关键企业ID"])
            f_round = int(row["轮次"])
            combo = (fid, f_round)
            if combo not in seen_combinations:
                key_tasks.append(combo)
                seen_combinations.add(combo)
            if len(key_tasks) >= TOP_K:
                break
        except KeyError:
            continue

    print(f"Target Firms (ID, Round): {key_tasks}")

    # ---------- 颜色方案 ----------
    case_colors = {
        "Case 1": "#003366",  # 深蓝色
        "Case 2": "#CC6600",  # 橙色
        "Case 3": "#009933"   # 绿色
    }
    
    # ---------- 循环作图 ----------
    for fid, f_round in key_tasks:
        # 创建一张图，包含所有Case的曲线
        fig, ax = plt.subplots(figsize=(10, 7))
        
        y_top_global = 0
        q_max_global = 0
        
        # 第一遍：计算全局的坐标轴范围
        for case_name, case_data in all_case_data.items():
            df_det = case_data["det"]
            df_sum = case_data["sum"]
            
            # 1. 获取数据
            cond_det = (df_det["企业ID"] == fid) & (df_det["轮次"] == f_round)
            df_target_det = df_det[cond_det]
            if df_target_det.empty: 
                continue
            
            row_det = df_target_det.iloc[0]
            e = row_det["初始排放e"]
            r = row_det["净流出r"]
            q_star = row_det["决策排放q"]
            subsidy = row_det["获得补贴"]

            cond_sum = (df_sum["关键企业ID"] == fid) & (df_sum["轮次"] == f_round)
            df_target_sum = df_sum[cond_sum]
            if df_target_sum.empty: 
                continue

            row_sum = df_target_sum.iloc[0]
            mu = row_sum["关键企业μ"]

            q_min = 0.01 
            q_max = max(1.5 * q_star, 1.1 * e)
            q_max_global = max(q_max_global, q_max)
            
            # 计算y轴范围
            C_vals = cost_C(np.array([q_star]), e, r, mu, subsidy, I_PARAM)
            AC_star = C_vals[0] / q_star
            MC_star = MC(np.array([q_star]), e, r, mu, subsidy, I_PARAM)[0]
            
            C_vals_no = cost_C(np.array([q_star]), e, r, mu, 0, I_PARAM)
            AC_no_at_star = C_vals_no[0] / q_star
            MC_no_at_star = MC(np.array([q_star]), e, r, mu, 0, I_PARAM)[0]
            
            y_ref_max = max(AC_star, MC_star, AC_no_at_star, MC_no_at_star)
            if y_ref_max > 0:
                y_top_global = max(y_top_global, y_ref_max * 1.4)
            else:
                y_top_global = max(y_top_global, y_ref_max * 0.6 + 5)
        
        # 第二遍：绘制所有Case的曲线
        first_case = True
        for case_name, case_data in all_case_data.items():
            df_det = case_data["det"]
            df_sum = case_data["sum"]
            color = case_colors.get(case_name, "#000000")
            
            # 1. 获取数据
            cond_det = (df_det["企业ID"] == fid) & (df_det["轮次"] == f_round)
            df_target_det = df_det[cond_det]
            if df_target_det.empty: 
                continue
            
            row_det = df_target_det.iloc[0]
            e = row_det["初始排放e"]
            r = row_det["净流出r"]
            q_star = row_det["决策排放q"]
            subsidy = row_det["获得补贴"]

            cond_sum = (df_sum["关键企业ID"] == fid) & (df_sum["轮次"] == f_round)
            df_target_sum = df_sum[cond_sum]
            if df_target_sum.empty: 
                continue

            row_sum = df_target_sum.iloc[0]
            mu = row_sum["关键企业μ"]

            # 2. 计算曲线数据
            q_min = 0.01 
            q_max = max(1.5 * q_star, 1.1 * e)
            q_vals = np.linspace(q_min, q_max_global, 400)

            # A. 有补贴 (Actual Scenario)
            C_vals = cost_C(q_vals, e, r, mu, subsidy, I_PARAM)
            AC_vals = C_vals / q_vals

            # B. 无补贴 (Baseline Scenario, subsidy=0)
            if first_case:
                C_vals_no = cost_C(q_vals, e, r, mu, 0, I_PARAM)
                AC_vals_no = C_vals_no / q_vals
                ax.plot(q_vals, AC_vals_no, label="AC (no-sub)", color='#666666', 
                        linestyle=':', linewidth=1.5, alpha=0.6, zorder=2)
                first_case = False

            # --- 绘制有补贴AC曲线 ---
            ax.plot(q_vals, AC_vals, label=f"{case_name} AC", color=color, 
                    linestyle='-', linewidth=2, zorder=3)

        # --- 坐标轴美化 (L-Shape) ---
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_position(('data', 0))
        ax.spines['bottom'].set_position(('data', 0))
        
        ax.set_xlim(left=0, right=q_max_global)
        ax.set_ylim(bottom=0, top=y_top_global)

        # 标签
        ax.set_xlabel(r"Emission Level ($q$)", loc='right', fontsize=12)
        ax.set_ylabel(r"Cost", loc='top', rotation=0, fontsize=12)
        
        # 标题
        ax.set_title(f"Cost Structure Comparison: Firm {fid} (Round {f_round})", 
                     fontsize=13, fontweight='bold', pad=15)

        # 图例
        ax.legend(loc='lower right', frameon=False, ncol=2, fontsize=10)

        plt.tight_layout()

        # 保存
        fname = f"{OUTPUT_DIR}/Comparison_Round{f_round}_Firm{fid}.png"
        plt.savefig(fname, dpi=600, bbox_inches='tight')
        plt.close()

        print(f"✅ Saved: {fname}")

    print("\n🎉 Visualization with Subsidy Comparison completed!")

if __name__ == "__main__":
    main()