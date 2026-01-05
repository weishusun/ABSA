# scripts/init_aspects_phone.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pandas as pd

try:
    import pyarrow.dataset as ds
except Exception:
    ds = None

try:
    import yaml
except Exception as e:
    raise RuntimeError("需要安装 pyyaml：pip install pyyaml") from e


# -----------------------------
# 基础配置
# -----------------------------
ALLOW_ENG = {
    "wifi", "wi-fi", "wifi6", "wifi7", "bluetooth", "gps", "nfc",
    "oled", "lcd", "pwm", "dc", "hdr", "cpu", "gpu", "ufs", "ram", "rom",
    "ip68", "ip67", "ip", "vc", "pd", "qc"
}
RE_ASCII = re.compile(r"^[A-Za-z0-9\-\+\.]+$")

RE_BAD = re.compile(
    r"^(?:"
    r"用户|机型|市场|技术|效果|细节|英寸|旗舰|体验|产品|手机|电脑|汽车|美妆|"
    r"东西|情况|地方|方面|时候|时间|原因|"
    r")$"
)


def load_brand_model_blocklist(clean_parquet: Path) -> Set[str]:
    """
    从 clean_sentences.parquet 提取 brand/model，加入 blocklist，避免候选被品牌/机型污染。
    """
    if ds is None:
        print("[WARN] pyarrow.dataset 不可用，跳过 brand/model blocklist 自动提取。")
        return set()

    dataset = ds.dataset(str(clean_parquet), format="parquet")
    tab = dataset.to_table(columns=["brand", "model"]).to_pandas()
    blk = set()
    for c in ["brand", "model"]:
        for v in tab[c].dropna().astype(str).unique().tolist():
            v = v.strip()
            if v:
                blk.add(v)
                blk.add(v.lower())
    return blk


def is_bad_term(term: str, blocklist: Set[str]) -> bool:
    t = str(term).strip()
    if not t:
        return True
    if RE_BAD.match(t):
        return True
    if t in blocklist or t.lower() in blocklist:
        return True
    # 机型后缀
    if t.lower() in {"pro", "ultra", "max", "plus", "se"}:
        return True
    # 纯英文/数字串：只保留少量白名单
    if RE_ASCII.match(t) and t.lower() not in ALLOW_ENG:
        return True
    # 太长一般是噪声
    if len(t) > 10:
        return True
    return False


def pick_terms_by_patterns(
    candidates: pd.DataFrame,
    patterns: List[str],
    min_df: int,
    blocklist: Set[str],
    extra_seeds: List[str] | None = None,
) -> List[str]:
    out: Set[str] = set()

    # 先加保底种子
    for s in (extra_seeds or []):
        s = str(s).strip()
        if s and not is_bad_term(s, blocklist):
            out.add(s)

    sub = candidates[candidates["df"] >= min_df].copy()
    terms = sub["term"].astype(str).tolist()

    regs = [re.compile(p) for p in patterns]
    for t in terms:
        if is_bad_term(t, blocklist):
            continue
        for r in regs:
            if r.search(t):
                out.add(t)
                break

    # 排序：中文优先、短词优先、字典序
    return sorted(out, key=lambda x: (RE_ASCII.match(x) is not None, len(x), x))


def write_lexicon(path: Path, terms: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "# one term per line\n" + "\n".join(terms) + "\n"
    path.write_text(content, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True, help="aspect_candidates_phone.xlsx 路径")
    ap.add_argument("--clean-parquet", default="", help="可选：clean_sentences.parquet，用于自动屏蔽 brand/model")
    ap.add_argument("--out-dir", default="aspects/phone", help="输出词典目录")
    ap.add_argument("--config-out", default="configs/aspects_phone.yaml", help="输出配置文件路径")
    ap.add_argument("--min-df", type=int, default=200, help="候选词 df 阈值（建议 100~500）")
    ap.add_argument("--top-unmapped", type=int, default=3000, help="输出未映射候选的最大行数")
    args = ap.parse_args()

    cand_path = Path(args.candidates)
    if not cand_path.exists():
        raise FileNotFoundError(f"candidates 不存在：{cand_path}")

    out_dir = Path(args.out_dir)
    config_out = Path(args.config_out)
    min_df = int(args.min_df)

    df = pd.read_excel(cand_path, sheet_name="candidates")
    if "term" not in df.columns or "df" not in df.columns or "tf" not in df.columns:
        raise RuntimeError("candidates 表需要包含 term/df/tf 列。")

    # blocklist：品牌/机型 + 常见噪声词
    blocklist = {
        "用户", "机型", "市场", "技术", "效果", "细节", "英寸", "旗舰", "体验",
        "产品", "手机", "苹果", "小米", "三星", "魅族", "一加", "红米", "荣耀", "中兴", "努比亚",
        "vivo", "oppo", "iqoo", "zte", "nubia", "meizu", "samsung", "galaxy", "iphone",
    }
    if args.clean_parquet:
        blk2 = load_brand_model_blocklist(Path(args.clean_parquet))
        blocklist |= blk2

    lex_dir = out_dir / "lexicons"
    lex_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------
    # phone 的 L1+L2 初始骨架（seed bins）
    # 完整性靠 tag_aspects 的覆盖率 Gate + unmapped/uncovered 残差闭环迭代
    # -----------------------------
    SPEC: List[Tuple[str, str, List[str], List[str]]] = [
        ("价格与性价比", "价格", [r"价格|售价|价位|定价|优惠|券|补贴|降价|涨价"], ["价格", "售价", "优惠", "券", "补贴"]),
        ("价格与性价比", "性价比", [r"性价比|划算|值不值|不值|值"], ["性价比", "划算", "值不值"]),

        ("性能与游戏", "性能", [r"性能|流畅|卡顿|掉帧|帧率|响应|延迟"], ["性能", "流畅", "卡顿", "掉帧", "帧率"]),
        ("性能与游戏", "芯片与处理器", [r"芯片|处理器|cpu|gpu|骁龙|天玑|麒麟|核心|制程"], ["芯片", "处理器", "CPU", "GPU", "骁龙", "天玑", "麒麟"]),
        ("性能与游戏", "内存与存储", [r"内存|运存|ram|存储|rom|ufs|闪存|容量|读写"], ["内存", "运存", "存储", "UFS", "闪存", "容量"]),
        ("性能与游戏", "游戏体验", [r"游戏|原神|王者|吃鸡|高帧|稳帧"], ["游戏", "原神", "王者", "吃鸡", "稳帧"]),

        ("续航与充电", "续航", [r"续航|待机|耗电|省电|功耗|电量|电池|衰减|健康"], ["续航", "待机", "耗电", "功耗", "电量", "电池", "衰减", "电池健康"]),
        ("续航与充电", "充电与快充", [r"充电|快充|闪充|充满|充电器|充电头|无线充|反向充|pd|qc"], ["充电", "快充", "充电器", "无线充", "反向充", "PD", "QC"]),

        ("屏幕与显示", "亮度与户外", [r"亮度|尼特|户外|阳光|可视|反光"], ["亮度", "户外", "阳光下", "可视"]),
        ("屏幕与显示", "护眼与调光", [r"护眼|pwm|频闪|dc调光|蓝光|调光"], ["护眼", "PWM", "频闪", "DC调光", "蓝光"]),
        ("屏幕与显示", "刷新率与触控", [r"刷新率|触控|触摸|采样率|跟手"], ["刷新率", "触控", "触摸", "采样率", "跟手"]),
        ("屏幕与显示", "色彩与分辨率", [r"色彩|色准|对比度|分辨率|清晰度|hdr|oled|lcd"], ["色彩", "色准", "对比度", "分辨率", "清晰度", "HDR", "OLED", "LCD"]),

        ("影像与视频", "拍照综合", [r"拍照|照片|成像|画质|解析|细节|像素|白平衡"], ["拍照", "照片", "成像", "画质", "解析", "细节", "像素", "白平衡"]),
        ("影像与视频", "夜景", [r"夜景|暗光|低光|噪点"], ["夜景", "暗光", "低光", "噪点"]),
        ("影像与视频", "人像", [r"人像|虚化|肤色|美颜"], ["人像", "虚化", "肤色", "美颜"]),
        ("影像与视频", "视频与防抖", [r"视频|录像|防抖|稳定|帧率"], ["视频", "录像", "防抖", "稳定"]),
        ("影像与视频", "变焦与镜头", [r"变焦|长焦|广角|超广角|潜望|镜头|焦段|微距"], ["变焦", "长焦", "广角", "超广角", "潜望", "镜头", "焦段", "微距"]),
        ("影像与视频", "对焦与抓拍", [r"对焦|抓拍|快门"], ["对焦", "抓拍", "快门"]),

        ("系统与软件", "系统体验", [r"系统|ui|桌面|交互|动画"], ["系统", "UI", "桌面", "交互"]),
        ("系统与软件", "广告与推送", [r"广告|推送|通知|弹窗|开屏|预装"], ["广告", "推送", "通知", "弹窗", "开屏", "预装"]),
        ("系统与软件", "BUG与更新", [r"bug|闪退|卡死|死机|更新|升级|补丁"], ["bug", "闪退", "卡死", "死机", "更新", "升级", "补丁"]),
        ("系统与软件", "应用与后台", [r"应用|app|兼容|权限|后台|杀后台"], ["应用", "APP", "兼容", "权限", "后台", "杀后台"]),

        ("信号与连接", "蜂窝信号与通话", [r"信号|5g|4g|通话|掉线|断流|基站"], ["信号", "5G", "4G", "通话", "掉线", "断流", "基站"]),
        ("信号与连接", "WiFi与蓝牙", [r"wifi|wi-fi|蓝牙|热点|断连|延迟"], ["WiFi", "蓝牙", "热点", "断连", "延迟"]),
        ("信号与连接", "定位与NFC", [r"gps|定位|nfc"], ["GPS", "定位", "NFC"]),

        ("外观与做工", "外观设计", [r"外观|设计|颜值|配色|颜色|质感"], ["外观", "设计", "颜值", "配色", "质感"]),
        ("外观与做工", "手感与重量", [r"手感|重量|厚度|尺寸|握持|硌手"], ["手感", "重量", "厚度", "尺寸", "握持", "硌手"]),
        ("外观与做工", "做工与材质", [r"做工|材质|玻璃|金属|塑料|边框|按键|缝隙|防水|ip"], ["做工", "材质", "玻璃", "金属", "塑料", "边框", "按键", "缝隙", "防水", "IP68"]),

        ("发热与散热", "发热", [r"发热|发烫|温度|烫手"], ["发热", "发烫", "温度", "烫手"]),
        ("发热与散热", "散热与降频", [r"散热|vc|均热板|热管|降频|温控|稳帧"], ["散热", "VC", "均热板", "热管", "降频", "温控"]),

        ("音频与振感", "扬声器与音质", [r"扬声器|音质|外放|立体声|听筒|麦克风"], ["扬声器", "音质", "外放", "立体声", "听筒", "麦克风"]),
        ("音频与振感", "马达与震动", [r"马达|震动|振感|线性马达"], ["马达", "震动", "振感", "线性马达"]),

        ("可靠性与服务", "质量与故障", [r"故障|坏|翻车|漏液|绿屏|烧屏|进灰|掉漆|断触"], ["故障", "翻车", "漏液", "绿屏", "烧屏", "进灰", "掉漆", "断触"]),
        ("可靠性与服务", "售后服务", [r"售后|客服|保修|维修|换机|退货"], ["售后", "客服", "保修", "维修", "换机", "退货"]),
        ("可靠性与服务", "物流与包装", [r"物流|快递|包装|到货|破损"], ["物流", "快递", "包装", "到货", "破损"]),
    ]

    # 生成词表文件 + 记录映射
    aspect_items = []
    for l1, l2, patterns, seeds in SPEC:
        terms = pick_terms_by_patterns(df, patterns, min_df=min_df, blocklist=blocklist, extra_seeds=seeds)
        lex_path = lex_dir / f"{l1}__{l2}.txt"
        write_lexicon(lex_path, terms)
        aspect_items.append((l1, l2, lex_path))

    # YAML config
    cfg = {"domain": "phone", "aspects": [], "settings": {}}
    l1_map: Dict[str, Dict[str, str]] = defaultdict(dict)
    for l1, l2, lex in aspect_items:
        l1_map[l1][l2] = str(lex.as_posix())

    for l1, l2s in l1_map.items():
        cfg["aspects"].append({"l1": l1, "l2": l2s})

    cfg["settings"] = {
        "coverage_gate": {"l1_min_rate": 0.85, "l2_min_rate": 0.70, "unclassified_max_rate": 0.10},
        "match": {"engine": "flashtext", "dedup_hits": True, "case_sensitive": False},
    }

    config_out.parent.mkdir(parents=True, exist_ok=True)
    config_out.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")

    print(f"[OK] wrote config: {config_out}")
    print(f"[OK] wrote lexicons under: {lex_dir}")

    # -----------------------------
    # 防漏机制：输出“高 df 未映射候选词”
    # -----------------------------
    assigned = set()
    for p in lex_dir.glob("*.txt"):
        for line in p.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if s and not s.startswith("#"):
                assigned.add(s)

    cand_hi = df[df["df"] >= min_df].copy()
    cand_hi["term"] = cand_hi["term"].astype(str)
    cand_hi["is_assigned"] = cand_hi["term"].apply(lambda x: x in assigned)
    unmapped = cand_hi[~cand_hi["is_assigned"]].copy()
    unmapped = unmapped.sort_values(["df", "tf"], ascending=[False, False])
    unmapped = unmapped[~unmapped["term"].apply(lambda x: is_bad_term(x, blocklist))]

    unmapped_path = out_dir / "unmapped_terms.xlsx"
    unmapped.head(int(args.top_unmapped)).to_excel(unmapped_path, index=False)
    print(f"[OK] wrote unmapped high-df terms: {unmapped_path}")


if __name__ == "__main__":
    main()
