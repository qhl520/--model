import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['font.sans-serif'] = ['SimHei']   # 黑体
matplotlib.rcParams['axes.unicode_minus'] = False     # 解决负号显示问题

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ================== 可调参数 ==================
CASE_FILE = "Case1_全流程结果.xlsx"
NETWORK_FILE = "网络构建及博弈均衡结果.xlsx"

TOP_K = 10          # 前 K 个关键企业
I_PARAM = 0.2       # I 参数（与博弈代码一致）
OUTPUT_DIR = "figures"
# ============================================


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


# -------- 成本函数（与你博弈代码一致） --------
def cost_C(q, e, r, mu, subsidy, I):
    """
    C(q) = μ(q+r) + I/2*(e-q)^2 - (subsidy/e)*(e-q)
    """
    return (
        mu * (q + r)
        + 0.5 * I * (e - q) ** 2
        - (subsidy / e) * (e - q)
    )


def MC(q, e, r, mu, subsidy, I):
    """
    MC(q) = dC/dq = μ - I(e-q) + subsidy/e
    """
    return mu - I * (e - q) + subsidy / e


# ================== 主流程 ==================
def main():
    ensure_dir(OUTPUT_DIR)

    # ---------- 读取 Case 文件 ----------
    xls = pd.ExcelFile(CASE_FILE)
    df_summary = pd.read_excel(xls, sheet_name="汇总信息")
    df_detail = pd.read_excel(xls, sheet_name="所有轮次明细")

    # ---------- 读取网络文件（μ） ----------
    df_net = pd.read_excel(NETWORK_FILE, sheet_name="网络结构")
    mu_map = dict(
        zip(df_net["节点ID"], df_net["碳边际减排成本μ"])
    )

    # ---------- 选取前 K 个关键企业 ----------
    key_ids = []
    for x in df_summary["关键企业ID"]:
        if x not in key_ids:
            key_ids.append(int(x))
        if len(key_ids) >= TOP_K:
            break

    print("选取的关键企业：", key_ids)

    # ---------- 循环画图 ----------
    for firm_id in key_ids:
        row = df_detail[df_detail["企业ID"] == firm_id].iloc[0]

        # 基本参数
        e = row["初始排放e"]
        r = row["净流出r"]
        q_star = row["决策排放q"]
        subsidy = row["获得补贴"]
        mu = mu_map[firm_id]

        # q 取值范围
        q_min = max(1e-4, 0.05 * q_star)
        q_max = max(e * 1.1, 1.5 * q_star)
        q_vals = np.linspace(q_min, q_max, 400)

        # 计算 AC / MC
        C_vals = cost_C(q_vals, e, r, mu, subsidy, I_PARAM)
        AC_vals = C_vals / q_vals
        MC_vals = MC(q_vals, e, r, mu, subsidy, I_PARAM)

        # 均衡点
        AC_star = cost_C(q_star, e, r, mu, subsidy, I_PARAM) / q_star
        MC_star = MC(q_star, e, r, mu, subsidy, I_PARAM)

        # ---------- 画图 ----------
        plt.figure(figsize=(8, 5))
        plt.plot(q_vals, AC_vals, label="AC(q)", linewidth=2)
        plt.plot(q_vals, MC_vals, label="MC(q)", linestyle="--", linewidth=2)

        #plt.scatter(q_star, AC_star, color="black", zorder=5, label="AC(q*)")
        #plt.scatter(q_star, MC_star, color="red", zorder=5, label="MC(q*)")

        plt.xlabel("决策排放 q")
        plt.ylabel("成本")
        plt.title(
            f"企业 {firm_id} 的成本曲线\n"
            f"μ={mu:.2f}, 补贴={subsidy:.0f}, e={e:.2f}, r={r:.2f}"
        )

        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()

        fname = f"{OUTPUT_DIR}/firm_{firm_id}_AC_MC.png"
        plt.savefig(fname, dpi=300)
        plt.close()

        print(f"已保存：{fname}")

    print("\n✅ 所有关键企业成本曲线绘制完成！")


# ================== 运行 ==================
if __name__ == "__main__":
    main()
