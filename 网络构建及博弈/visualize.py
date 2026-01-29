import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

# ---------- 中文字体（避免警告） ----------
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# ---------- 文件与参数 ----------
CASE_FILE = "Case1_全流程结果.xlsx"
TOP_K = 10
I_PARAM = 0.2
OUTPUT_DIR = "figures"


# ---------- 成本函数 ----------
def cost_C(q, e, r, mu, subsidy, I):
    """
    非合作博弈成本函数
    C(q) = μ(q+r) + I/2*(e-q)^2 - (s/e)*(e-q)
    """
    return mu * (q + r) + 0.5 * I * (e - q) ** 2 - (subsidy / e) * (e - q)


def MC(q, e, r, mu, subsidy, I):
    """
    边际成本
    MC(q) = μ - I(e-q) + s/e
    """
    return mu - I * (e - q) + subsidy / e


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ---------- 读取 Excel ----------
    df_sum = pd.read_excel(CASE_FILE, sheet_name="汇总信息")
    df_det = pd.read_excel(CASE_FILE, sheet_name="所有轮次明细")

    # ---------- 前 TOP_K 个关键企业 ----------
    key_ids = []
    for x in df_sum["关键企业ID"]:
        if x not in key_ids:
            key_ids.append(int(x))
        if len(key_ids) >= TOP_K:
            break

    print("选取的关键企业：", key_ids)

    # ---------- 作图 ----------
    for fid in key_ids:
        # 明细表：e, r, q, subsidy
        row_det = df_det[df_det["企业ID"] == fid].iloc[0]

        e = row_det["初始排放e"]
        r = row_det["净流出r"]
        q_star = row_det["决策排放q"]
        subsidy = row_det["获得补贴"]

        # 汇总表：μ（已经内生化）
        row_sum = df_sum[df_sum["关键企业ID"] == fid].iloc[0]
        mu = row_sum["关键企业μ"]

        # q 取值范围
        q_min = max(1e-4, 0.05 * q_star)
        q_max = max(1.5 * q_star, 1.1 * e)
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
        plt.plot(q_vals, MC_vals, "--", label="MC(q)", linewidth=2)

        plt.scatter(q_star, AC_star, c="black", zorder=5)
        plt.scatter(q_star, MC_star, c="red", zorder=5)

        plt.xlabel("决策排放 q")
        plt.ylabel("成本")
        plt.title(f"企业 {fid} 的成本曲线")

        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()

        fname = f"{OUTPUT_DIR}/firm_{fid}_AC_MC.png"
        plt.savefig(fname, dpi=300)
        plt.close()

        print(f"已保存：{fname}")

    print("\n✅ 最终版可视化完成（仅 AC / MC）")


if __name__ == "__main__":
    main()
