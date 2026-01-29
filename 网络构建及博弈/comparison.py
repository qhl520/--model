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
CASE_FILE = "Case1_全流程结果.xlsx"
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

    print(f"Reading Excel file: {CASE_FILE} ...")
    try:
        df_sum = pd.read_excel(CASE_FILE, sheet_name="汇总信息")
        df_det = pd.read_excel(CASE_FILE, sheet_name="所有轮次明细")
    except FileNotFoundError:
        print(f"Error: File {CASE_FILE} not found.")
        return
    except ValueError as e:
        print(f"Error reading sheets: {e}")
        return

    # 清理列名
    df_sum.columns = df_sum.columns.str.strip()
    df_det.columns = df_det.columns.str.strip()

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

    # ---------- 循环作图 ----------
    for fid, f_round in key_tasks:
        # 1. 获取数据
        cond_det = (df_det["企业ID"] == fid) & (df_det["轮次"] == f_round)
        df_target_det = df_det[cond_det]
        if df_target_det.empty: continue
        
        row_det = df_target_det.iloc[0]
        e = row_det["初始排放e"]
        r = row_det["净流出r"]
        q_star = row_det["决策排放q"]
        subsidy = row_det["获得补贴"] # 实际获得的补贴

        cond_sum = (df_sum["关键企业ID"] == fid) & (df_sum["轮次"] == f_round)
        df_target_sum = df_sum[cond_sum]
        if df_target_sum.empty: continue

        row_sum = df_target_sum.iloc[0]
        mu = row_sum["关键企业μ"]

        # 2. 计算曲线数据
        q_min = 0.01 
        q_max = max(1.5 * q_star, 1.1 * e)
        q_vals = np.linspace(q_min, q_max, 400)

        # A. 有补贴 (Actual Scenario)
        C_vals = cost_C(q_vals, e, r, mu, subsidy, I_PARAM)
        AC_vals = C_vals / q_vals
        MC_vals = MC(q_vals, e, r, mu, subsidy, I_PARAM)

        # B. 无补贴 (Baseline Scenario, subsidy=0)
        C_vals_no = cost_C(q_vals, e, r, mu, 0, I_PARAM)
        AC_vals_no = C_vals_no / q_vals
        MC_vals_no = MC(q_vals, e, r, mu, 0, I_PARAM)

        # 3. 计算均衡点 (基于有补贴的实际决策)
        AC_star = cost_C(q_star, e, r, mu, subsidy, I_PARAM) / q_star
        MC_star = MC(q_star, e, r, mu, subsidy, I_PARAM)

        # 4. 确定 Y 轴上限
        # 需同时考虑有补贴和无补贴曲线在 q* 附近的值，保证都被包含
        # 获取 q* 处无补贴的 AC/MC 值用于定界
        AC_no_at_star = cost_C(q_star, e, r, mu, 0, I_PARAM) / q_star
        MC_no_at_star = MC(q_star, e, r, mu, 0, I_PARAM)
        
        y_ref_max = max(AC_star, MC_star, AC_no_at_star, MC_no_at_star)
        y_ref_min = min(AC_star, MC_star, AC_no_at_star, MC_no_at_star)
        
        # 简单的自适应上限
        if y_ref_max > 0:
            y_top = y_ref_max * 1.4
        else:
            y_top = y_ref_max * 0.6 + 5 # 处理负值情况
            
        # ---------- 绘制 ----------
        fig, ax = plt.subplots(figsize=(7, 5))

        # --- 1. 绘制无补贴基准线 (No Subsidy) ---
        # 样式：点状线 (:)，透明度稍高，作为背景参考
        # AC No Sub
        ax.plot(q_vals, AC_vals_no, label=r"$AC_{no-sub}$", color='#003366', 
                linestyle=':', linewidth=1.5, alpha=0.6)
        # MC No Sub
        ax.plot(q_vals, MC_vals_no, label=r"$MC_{no-sub}$", color='#8B0000', 
                linestyle=':', linewidth=1.5, alpha=0.6)

        # --- 2. 绘制有补贴实际线 (With Subsidy) ---
        # 样式：AC实线 (-)，MC长虚线 (--)，颜色饱满
        # AC Actual
        ax.plot(q_vals, AC_vals, label=r"$AC_{subsidy}$", color='#003366', 
                linestyle='-', linewidth=2, zorder=3)
        # MC Actual
        ax.plot(q_vals, MC_vals, label=r"$MC_{subsidy}$", color='#8B0000', 
                linestyle='--', linewidth=2, zorder=3)

        # --- 3. 标记均衡点 (q*) ---
        #ax.scatter([q_star], [AC_star], color='white', edgecolor='#003366', s=50, zorder=4, marker='o')
        #ax.scatter([q_star], [MC_star], color='white', edgecolor='#8B0000', s=50, zorder=4, marker='o')

        # 引导线
        #ax.vlines(x=q_star, ymin=-999, ymax=y_top, colors='gray', linestyles=':', linewidth=1, alpha=0.5)
        # 仅为实际点画水平引导线
        #ax.hlines(y=AC_star, xmin=0, xmax=q_star, colors='gray', linestyles=':', linewidth=1, alpha=0.3)
        #ax.hlines(y=MC_star, xmin=0, xmax=q_star, colors='gray', linestyles=':', linewidth=1, alpha=0.3)

        # --- 4. 坐标轴美化 (L-Shape) ---
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_position(('data', 0))
        ax.spines['bottom'].set_position(('data', 0))
        
        ax.set_xlim(left=0, right=q_max)
        ax.set_ylim(bottom=0, top=y_top) # 强制从0开始，除非数值全是负的

        # 标签
        ax.set_xlabel(r"Emission Level ($q$)", loc='right')
        ax.set_ylabel(r"Cost", loc='top', rotation=0)
        
        # 标注 q*
        #ax.text(q_star, -y_top*0.05, r"$q^*$", ha='center', va='top', fontsize=12)

        # 标题
        ax.set_title(f"Cost Structure Analysis: Firm {fid} (Round {f_round})", pad=20)

        # 图例：放在最合适的位置，通常右下
        ax.legend(loc='lower right', frameon=False, ncol=2) # 分两列显示更整齐

        plt.tight_layout()

        # 保存
        fname = f"{OUTPUT_DIR}/Comparison_Round{f_round}_Firm{fid}.png"
        plt.savefig(fname, dpi=600)
        plt.close()

        print(f"✅ Saved: {fname}")

    print("\n🎉 Visualization with Subsidy Comparison completed!")

if __name__ == "__main__":
    main()
