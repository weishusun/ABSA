import os
import pandas as pd

inp = r".\outputs\phone\aspect_counts_phone.xlsx"
out_dir = r".\outputs\phone\by_product"
os.makedirs(out_dir, exist_ok=True)

df = pd.read_excel(inp)

def safe_name(s: str) -> str:
    return (s or "").replace("/", "_").replace("\\", "_").replace(" ", "").replace(":", "_")

for (brand, model), g in df.groupby(["brand", "model"], sort=True):
    fname = f"{safe_name(str(brand))}_{safe_name(str(model))}.xlsx"
    path = os.path.join(out_dir, fname)

    g2 = g.sort_values("hit_count", ascending=False).reset_index(drop=True)
    pivot = g.pivot_table(
        index="aspect_l1", columns="aspect_l2", values="hit_count",
        aggfunc="sum", fill_value=0
    )

    with pd.ExcelWriter(path, engine="openpyxl") as w:
        g2.to_excel(w, sheet_name="L1L2_counts", index=False)
        pivot.to_excel(w, sheet_name="pivot")

# 生成索引表，方便你快速定位每个产品文件
index = (
    df[["brand", "model"]].drop_duplicates()
      .sort_values(["brand", "model"])
      .assign(file=lambda x: x["brand"].astype(str).map(safe_name) + "_" + x["model"].astype(str).map(safe_name) + ".xlsx")
)
index.to_excel(os.path.join(out_dir, "INDEX.xlsx"), index=False)

print("[OK] wrote per-product excels to:", out_dir)
