import streamlit as st
import pandas as pd
import plotly.express as px
import subprocess
import os
import sys
import collections
import re
import time
import yaml
import shutil
import json
import difflib
import sqlite3
from pathlib import Path
import datetime

# --- 0. 依赖检查 ---
try:
    import jieba

    HAS_JIEBA = True
except ImportError:
    HAS_JIEBA = False

# --- 1. 全局配置与样式 ---
st.set_page_config(
    page_title="ABSA 舆情分析系统",
    layout="wide",
    page_icon="⚡",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* 全局字体与重置 */
    .stApp { font-family: "Inter", system-ui, sans-serif; }
    /* 侧边栏优化 */
    [data-testid="stSidebar"] { background-color: #f8f9fa; border-right: 1px solid #e9ecef; }
    /* 标题样式 */
    h1 { font-weight: 700 !important; color: #111827; }
    h2, h3 { font-weight: 600 !important; color: #374151; }
    /* 卡片容器增强 */
    [data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 0.75rem;
        box-shadow: 0 1px 3px 0 rgb(0 0 0 / 0.1);
        background-color: white;
        padding: 1rem;
    }
    /* 进度条颜色品牌化 */
    .stProgress > div > div > div > div { background-color: #3b82f6; }
    /* 代码块字体 */
    code { font-family: 'JetBrains Mono', monospace; }
</style>
""", unsafe_allow_html=True)

# --- 2. 动态路径初始化 ---
ROOT = Path(__file__).resolve().parent
CONFIGS_DIR = ROOT / "configs" / "domains"
python_exe = sys.executable

# --- 3. 侧边栏：核心配置区 ---
with st.sidebar:
    st.title("⚙️ 系统配置")

    with st.container(border=True):
        st.subheader("1. 工作区设置")
        default_ws = os.environ.get("ABSA_WORKSPACE", str(ROOT / "workspace_data"))
        user_ws_input = st.text_input("数据存放目录 (Workspace)", value=default_ws,
                                      help="所有输入/输出数据将存放在此目录下")
        WORKSPACE = Path(user_ws_input).resolve()
        INPUTS_DIR = WORKSPACE / "inputs"
        OUTPUTS_DIR = WORKSPACE / "outputs"

        if not WORKSPACE.exists():
            st.warning("⚠️ 目录不存在")
            if st.button("创建工作区文件夹"):
                try:
                    INPUTS_DIR.mkdir(parents=True, exist_ok=True)
                    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
                    st.success("✅ 已创建！")
                    time.sleep(0.5)
                    st.rerun()
                except Exception as e:
                    st.error(f"创建失败: {e}")
        else:
            st.caption(f"✅ 状态: 已连接")

    st.markdown("---")

    with st.container(border=True):
        st.subheader("2. 任务参数")
        domain = st.selectbox("📦 领域 (Domain)", ["car", "phone", "laptop", "beauty"], index=0)
        run_id = st.text_input("🏷️ 任务标识 (Run ID)", value="prod_v1_full")

    st.markdown("---")

    # ==================== 智能 LLM 配置版 ====================
    with st.container(border=True):
        st.subheader("3. 🧠 LLM 模型配置")

        LLM_PRESETS = {
            "OpenAI (官方)": {"base_url": "https://api.openai.com/v1", "models": ["gpt-4o-mini", "gpt-4o"]},
            "DeepSeek (深度求索)": {"base_url": "https://api.deepseek.com",
                                    "models": ["deepseek-chat", "deepseek-coder"]},
            "Moonshot (Kimi)": {"base_url": "https://api.moonshot.cn/v1", "models": ["moonshot-v1-8k"]},
            "Aliyun (通义千问)": {"base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                                  "models": ["qwen-plus"]},
            "自定义 (Custom)": {"base_url": "", "models": []}
        }

        with st.expander("配置详情 (点击展开)", expanded=False):
            provider = st.selectbox("API 服务商", options=list(LLM_PRESETS.keys()), index=1)
            selected_preset = LLM_PRESETS[provider]

            env_key = os.environ.get("OPENAI_API_KEY", "")
            user_key = st.text_input("API Key", value=env_key, type="password", key="idx_api_key")

            default_base = selected_preset["base_url"]
            if provider == "自定义 (Custom)" and os.environ.get("OPENAI_BASE_URL"):
                default_base = os.environ.get("OPENAI_BASE_URL")

            user_base_raw = st.text_input("Base URL", value=default_base, key=f"idx_base_url_{provider}")
            user_base = re.sub(r"[\[\]\(\)]", "", user_base_raw).split("http")[-1]
            if user_base: user_base = "http" + user_base.strip()

            if provider == "自定义 (Custom)":
                default_model = os.environ.get("OPENAI_MODEL_NAME", "")
                user_model = st.text_input("模型名称", value=default_model)
            else:
                model_options = selected_preset["models"] + ["📝 手动输入..."]
                selected_model_opt = st.selectbox("选择模型", model_options, key=f"idx_model_sel_{provider}")
                user_model = st.text_input("请输入模型名称",
                                           value="") if selected_model_opt == "📝 手动输入..." else selected_model_opt

            if user_key: os.environ["OPENAI_API_KEY"] = user_key.strip()
            if user_base: os.environ["OPENAI_BASE_URL"] = user_base.strip()
            if user_model: os.environ["OPENAI_MODEL_NAME"] = user_model.strip()

            if st.button("🔌 测试连接", use_container_width=True):
                if not user_key:
                    st.error("请先填写 API Key")
                else:
                    try:
                        from openai import OpenAI

                        with st.spinner(f"正在连接 {provider}..."):
                            client = OpenAI(api_key=user_key, base_url=user_base)
                            resp = client.chat.completions.create(model=user_model,
                                                                  messages=[{"role": "user", "content": "Hi"}],
                                                                  max_tokens=5)
                            st.toast(f"✅ 连接成功! {resp.choices[0].message.content}", icon="🟢")
                    except Exception as e:
                        st.error(f"❌ 连接失败: {e}")

    st.markdown("---")
    page = st.radio("流程导航", ["0️⃣ 数据准备", "1️⃣ 覆盖率实验室", "2️⃣ 训练与推理", "3️⃣ 数据看板 (DB版)"])


# --- 4. 核心工具函数 ---
def get_files(domain):
    base = OUTPUTS_DIR / domain
    config_path = CONFIGS_DIR / domain / "aspects.yaml"
    return {
        "raw_dir": INPUTS_DIR,
        "clean": base / "clean_sentences.parquet",
        "aspect": base / "aspect_sentences.parquet",
        "config": config_path,
        "excel": base / "runs",
        "db": base / "stats.db"  # 新增数据库路径
    }


def run_command_with_progress(cmd_list, desc="执行任务中..."):
    with st.status(desc, expanded=True) as status:
        st.write(f"🔧 **Command:** `{' '.join(cmd_list)}`")
        progress_bar = st.progress(0)
        log_area = st.empty()
        logs = []

        env = os.environ.copy()
        env["ABSA_WORKSPACE"] = str(WORKSPACE)
        env["PYTHONIOENCODING"] = "utf-8"

        try:
            process = subprocess.Popen(cmd_list, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                       encoding='utf-8', errors='replace', cwd=str(ROOT), env=env)
            line_count = 0
            while True:
                line = process.stdout.readline()
                if not line and process.poll() is not None: break
                if line:
                    clean_line = line.strip()
                    logs.append(clean_line)
                    line_count += 1
                    log_area.code("\n".join(logs[-8:]), language="bash")
                    print(clean_line, flush=True)
                    current_prog = min(95, int((line_count % 100) + (line_count / 200)))
                    progress_bar.progress(current_prog)
            process.wait()
            progress_bar.progress(100)
            if process.returncode == 0:
                status.update(label="✅ 任务完成", state="complete", expanded=False)
                return True
            else:
                status.update(label="❌ 任务失败", state="error", expanded=True)
                return False
        except Exception as e:
            st.error(f"无法启动进程: {e}")
            return False


@st.cache_data(ttl=60)
def analyze_coverage(clean_path, aspect_path):
    if not clean_path.exists() or not aspect_path.exists(): return None, None, None
    try:
        df_clean = pd.read_parquet(clean_path)
        df_aspect = pd.read_parquet(aspect_path)
    except:
        return None, None, None

    total = len(df_clean)
    if total == 0: return 0.0, [], pd.DataFrame()

    clean_col = next((col for col in ['text', 'sentence', 'content'] if col in df_clean.columns), None)
    aspect_col = next((col for col in ['text', 'sentence', 'content'] if col in df_aspect.columns), None)
    id_col = next((col for col in ['sentence_id', 'id', 'row_id'] if col in df_clean.columns), None)

    df_uncovered = pd.DataFrame()
    if id_col and id_col in df_aspect.columns:
        covered_ids = set(df_aspect[id_col].unique())
        df_uncovered = df_clean[~df_clean[id_col].isin(covered_ids)].copy()
    elif clean_col and aspect_col:
        covered_texts = set(df_aspect[aspect_col].unique())
        df_uncovered = df_clean[~df_clean[clean_col].isin(covered_texts)].copy()

    coverage = (total - len(df_uncovered)) / total
    suggestions = []
    if not df_uncovered.empty and clean_col:
        sample_text = df_uncovered[clean_col].dropna().head(2000).astype(str).tolist()
        text_corpus = " ".join(sample_text)
        words = jieba.cut(text_corpus) if HAS_JIEBA else re.split(r'\W+', text_corpus)
        stopwords = {'这个', '那个', 'the', 'and', 'not'}
        words = [w for w in words if len(w) > 1 and w not in stopwords]
        suggestions = collections.Counter(words).most_common(30)

    return coverage, suggestions, df_uncovered


paths = get_files(domain)

# -----------------------------------------------------------------------------
# 0️⃣ 数据准备 (新增翻译功能)
# -----------------------------------------------------------------------------
if page == "0️⃣ 数据准备":
    st.title("🗂️ Step 00: 数据准备 & 翻译")
    current_domain_input_dir = INPUTS_DIR / domain

    # 新增 Tabs
    tab_trans, tab_clean = st.tabs(["🌐 辅助工具: 数据翻译", "🧹 核心流程: 清洗与标准化"])

    # --- Tab 1: 批量翻译工具 (支持实时日志滚动) ---
    with tab_trans:
        st.subheader("🌐 批量数据翻译")
        st.info("自动扫描指定目录下的所有 JSON/JSONL 文件，并批量调用 LLM 进行翻译。")

        # 1. 设置源目录
        default_trans_dir = WORKSPACE / "Translation"
        scan_dir_input = st.text_input("📂 翻译源目录 (结果也将保存在此根目录下)",
                                       value=str(default_trans_dir))
        scan_path = Path(scan_dir_input)

        # 2. 扫描文件
        found_files = []
        if scan_path.exists():
            found_files.extend(list(scan_path.rglob("*.json")))
            found_files.extend(list(scan_path.rglob("*.jsonl")))

        # --- 🕵️ 数据侦探 ---
        suggested_content_key = "content"
        suggested_id_key = "id"
        preview_data = None
        if found_files:
            try:
                sample_file = found_files[0]
                with open(sample_file, 'r', encoding='utf-8') as f:
                    first_char = f.read(1)
                    f.seek(0)
                    if first_char == '[' or first_char == '{':
                        try:
                            data = json.load(f)
                            if isinstance(data, dict):
                                for k, v in data.items():
                                    if isinstance(v, list) and len(v) > 0 and isinstance(v[0], dict):
                                        preview_data = v[0];
                                        break
                                if not preview_data: preview_data = data
                            elif isinstance(data, list) and len(data) > 0:
                                preview_data = data[0]
                        except:
                            pass
                    if not preview_data:
                        line = f.readline()
                        preview_data = json.loads(line)
                if preview_data:
                    keys = list(preview_data.keys())
                    for k in ['content', 'text', 'body', 'review', 'comment', 'detail']:
                        if k in keys: suggested_content_key = k; break
                    for k in ['id', 'review_id', 'comment_id', 'uuid', 'row_id']:
                        if k in keys: suggested_id_key = k; break
                    st.success(f"✅ 自动侦测成功！已根据 `{sample_file.name}` 匹配字段。")
                    with st.expander("🕵️ 查看源数据样本 (点击展开)", expanded=False):
                        st.json(preview_data)
            except Exception as e:
                st.warning(f"无法预览数据结构: {e}")

        # --------------------------------
        if not found_files:
            if not scan_path.exists():
                st.warning(f"目录不存在: {scan_path}")
            else:
                st.warning("该目录下未找到 .json 或 .jsonl 文件")
        else:
            st.write(f"📊 共发现 {len(found_files)} 个文件")
            c1, c2 = st.columns(2)
            with c1:
                content_key = st.text_input("内容字段名 (Content Key)", value=suggested_content_key)
            with c2:
                id_key = st.text_input("ID 字段名 (ID Key)", value=suggested_id_key)

            if st.button("🚀 开始批量翻译", type="primary"):
                if not os.environ.get("OPENAI_API_KEY"):
                    st.error("❌ 请先在左侧侧边栏配置 API Key！")
                else:
                    target_output_dir = scan_path
                    progress_bar = st.progress(0)
                    success_files = []

                    # 使用 st.status 包裹整个批量任务
                    with st.status("正在执行批量翻译任务...", expanded=True) as status:
                        log_container = st.empty()  # 创建一个空的容器用于显示滚动日志

                        for idx, src_file in enumerate(found_files):
                            parent_name = src_file.parent.name
                            safe_name = f"{parent_name}_{src_file.stem}.json"
                            target_file = target_output_dir / safe_name

                            status.update(label=f"🔄 [{idx + 1}/{len(found_files)}] 正在处理: {src_file.name}")
                            st.write(f"📄 文件: `{src_file.name}` -> `{safe_name}`")

                            cmd = [
                                python_exe, "-u", str(ROOT / "scripts" / "tools" / "translate_raw_tool.py"),
                                "--input", str(src_file),
                                "--output", str(target_file),
                                "--content-key", content_key,
                                "--id-key", id_key,
                                "--model", os.environ.get("OPENAI_MODEL_NAME", "deepseek-chat"),
                                "--base-url", os.environ.get("OPENAI_BASE_URL", "https://api.deepseek.com"),
                                "--api-key", os.environ.get("OPENAI_API_KEY"),
                                "--batch-size", "2"
                            ]

                            env = os.environ.copy()
                            env["PYTHONIOENCODING"] = "utf-8"

                            # [关键修改] 使用 Popen 实现实时日志流
                            try:
                                process = subprocess.Popen(
                                    cmd,
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT,  # 将错误日志也合并到输出流
                                    text=True,
                                    encoding='utf-8',
                                    errors='replace',
                                    env=env,
                                    bufsize=1  # 行缓冲，确保实时输出
                                )

                                # 实时读取日志
                                recent_logs = []
                                while True:
                                    line = process.stdout.readline()
                                    if not line and process.poll() is not None:
                                        break
                                    if line:
                                        clean_line = line.strip()
                                        if clean_line:
                                            recent_logs.append(clean_line)
                                            # 只保留最后 15 行，防止界面卡顿
                                            if len(recent_logs) > 15:
                                                recent_logs.pop(0)
                                            # 实时刷新日志框
                                            log_container.code("\n".join(recent_logs), language="bash")

                                if process.returncode == 0:
                                    success_files.append(target_file)
                                else:
                                    st.error(f"❌ 文件 {src_file.name} 翻译失败")

                            except Exception as e:
                                st.error(f"启动进程失败: {e}")

                            progress_bar.progress((idx + 1) / len(found_files))

                        status.update(label=f"✅ 任务完成！成功: {len(success_files)}/{len(found_files)}",
                                      state="complete")

                    if success_files:
                        st.divider()
                        st.subheader("👀 结果验证 (Verification)")
                        st.success(f"🎉 翻译完成！文件已保存在: `{target_output_dir}`")

                        last_file = success_files[-1]
                        st.write(f"**🔎 抽检文件:** `{last_file.name}`")

                        try:
                            # 修改读取逻辑以支持标准 JSON
                            with open(last_file, 'r', encoding='utf-8') as f:
                                # JSON 文件不能按行读，直接 load 前几个即可
                                data = json.load(f)
                                preview_lines = data[:3]  # 取前3个

                            if preview_lines:
                                st.caption("👇 以下是该文件的前 3 条翻译结果，请检查中文是否正常：")
                                st.json(preview_lines)
                        except Exception as e:
                            st.error(f"无法读取预览文件: {e}")
    # --- Tab 2: 原有清洗流程 ---
    with tab_clean:
        with st.container(border=True):
            col1, col2 = st.columns([1, 1])
            with col1:
                st.subheader("1. 扫描与清洗")
                scan_dir_input = st.text_input("扫描目标子目录", value=str(current_domain_input_dir))
                scan_path = Path(scan_dir_input)
                valid_files = []

                if scan_path.exists():
                    found_files = list(scan_path.rglob('*.*'))
                    valid_files = [f for f in found_files if f.suffix in ['.json', '.jsonl', '.txt']]
                    if valid_files:
                        st.success(f"✅ 发现 {len(valid_files)} 个源文件")
                        with st.expander("查看文件列表"):
                            st.write([f.name for f in valid_files])
                    else:
                        st.warning("⚠️ 该目录下为空或没有支持的文件格式")
                else:
                    st.error("❌ 目录不存在")
                    if st.button(f"创建文件夹: {domain}"):
                        scan_path.mkdir(parents=True, exist_ok=True)
                        st.rerun()

                if valid_files:
                    target_output_file = OUTPUTS_DIR / domain / "clean_sentences.parquet"
                    force_overwrite = st.checkbox("强制覆盖 (清除旧数据)", value=True)

                    if st.button("▶️ 开始清洗 (Run Step 00)", type="primary", use_container_width=True):
                        if force_overwrite and target_output_file.exists():
                            try:
                                target_output_file.unlink()
                            except:
                                pass

                        cmd = [
                            python_exe, "-u", str(ROOT / "scripts" / "step00_ingest_json_to_clean_sentences.py"),
                            "--domain", domain,
                            "--data-root", str(scan_path),
                            "--output", str(target_output_file)
                        ]
                        if run_command_with_progress(cmd, desc="正在清洗数据..."):
                            st.balloons()
                            time.sleep(1)
                            st.rerun()

            with col2:
                st.subheader("2. 结果预览")
                if paths['clean'].exists():
                    try:
                        df = pd.read_parquet(paths['clean'])
                        st.metric("清洗后语料", f"{len(df):,}", delta="Ready")
                        st.dataframe(df.head(10), height=300, hide_index=True, use_container_width=True)
                    except Exception as e:
                        st.error(f"读取失败: {e}")
                else:
                    st.info("暂无结果，请先运行清洗任务。")
# -----------------------------------------------------------------------------
# 1️⃣ 覆盖率实验室 (修复版：直连 Worker 获取进度)
# -----------------------------------------------------------------------------
elif page == "1️⃣ 覆盖率实验室":
    st.title("🧪 Step 01: 覆盖率优化")

    if not paths['clean'].exists():
        st.error(f"⚠️ 找不到数据：`{paths['clean']}`\n请先返回 Step 00 执行清洗。")
        st.stop()

    if 'coverage_data' not in st.session_state:
        st.session_state.coverage_data = None

    with st.container(border=True):
        col1, col2 = st.columns([1, 1])

        # --- 左侧：生产 (Tagging) ---
        with col1:
            st.subheader("1. 规则匹配 (Tagging)")
            st.info("执行正则脚本，生成 Aspect 数据。")

            if paths['config'].exists():
                st.caption(f"✅ 规则文件: `{paths['config'].name}`")

                if st.button("▶️ 运行规则匹配", type="primary", use_container_width=True):
                    # [关键修改] 直接调用 worker 脚本，跳过 runner，以便获取实时 stdout
                    worker_script = ROOT / "scripts" / "tag_aspects.py"
                    output_dir = paths['aspect'].parent

                    cmd = [
                        python_exe, "-u", str(worker_script),
                        "--input", str(paths['clean']),
                        "--config", str(paths['config']),
                        "--output-dir", str(output_dir),
                        # 降低批次大小以获得更频繁的进度更新
                        "--batch-size", "50000"
                    ]

                    if run_command_with_progress(cmd, desc="正则匹配计算中..."):
                        st.success("匹配完成！请点击右侧按钮进行分析。")
                        st.session_state.coverage_data = None
                        time.sleep(1)
                        st.rerun()
            else:
                st.error("❌ 缺失配置文件")
                st.markdown(f"请在代码仓库创建:\n`{paths['config']}`")

        with col2:
            st.subheader("2. 效果分析 (Analysis)")
            st.info("计算覆盖率并挖掘遗漏词。")

            if paths['aspect'].exists():
                mtime = time.ctime(os.path.getmtime(paths['aspect']))
                st.caption(f"✅ 数据已就绪 (更新于: {mtime})")

                if st.button("📊 开始分析覆盖率", use_container_width=True):
                    with st.spinner("正在分析数据..."):
                        cov, sugg, df_un = analyze_coverage(paths['clean'], paths['aspect'])
                        st.session_state.coverage_data = {
                            "coverage": cov, "suggestions": sugg, "uncovered_df": df_un
                        }
            else:
                st.warning("请先运行左侧的规则匹配。")

    # 结果展示区
    data = st.session_state.coverage_data
    if data:
        st.divider()
        m1, m2, m3 = st.columns(3)
        cov = data["coverage"]
        m1.metric("覆盖率", f"{cov:.1%}", delta_color="normal" if cov > 0.5 else "inverse")
        m2.metric("未匹配", f"{len(data['uncovered_df']):,}")
        m3.metric("总量", f"{len(pd.read_parquet(paths['clean'])):,}")

        if data["suggestions"]:
            with st.container(border=True):
                st.subheader("🧠 智能建议")
                st.caption("以下词汇频繁出现但未被覆盖：")
                tags_html = "<div style='display: flex; flex-wrap: wrap; gap: 6px;'>"
                for word, count in data["suggestions"][:20]:
                    tags_html += f"<span style='background:#eff6ff;padding:4px 10px;border-radius:12px;font-size:0.9em;border:1px solid #bfdbfe;color:#1e40af'><b>{word}</b> <small style='opacity:0.7'>({count})</small></span>"
                tags_html += "</div>"
                st.markdown(tags_html, unsafe_allow_html=True)

        st.divider()
        # --- AI 优化区域 (简洁版，无 L1 开关) ---
        st.subheader("🧠 AI 规则进化")

        c_edit, c_view = st.columns([1, 1])
        with c_edit:
            # 状态管理
            if "yaml_content" not in st.session_state:
                if paths['config'].exists():
                    st.session_state.yaml_content = paths['config'].read_text(encoding='utf-8')
                else:
                    st.session_state.yaml_content = ""

            if "pending_yaml" not in st.session_state:
                st.session_state.pending_yaml = None

            # 按钮区
            b1, b2 = st.columns(2)
            with b1:
                # 默认严格模式：只填词，不加 L1
                if st.button("🤖 1. AI 智能分析", use_container_width=True):
                    # [修复] 统一使用全称 'suggestions'，并用 .get() 防御
                    if not data.get("suggestions"):
                        st.warning("无新词")
                    else:
                        with st.spinner("AI 思考中..."):
                            w_list = [x[0] for x in data["suggestions"]]
                            cmd = [python_exe, str(ROOT / "scripts" / "optimize_rules.py"),
                                   "--yaml-path", str(paths['config']),
                                   "--suggestions", json.dumps(w_list),
                                   "--domain", domain]

                            # 注入环境
                            env = os.environ.copy()
                            env["PYTHONIOENCODING"] = "utf-8"

                            res = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', env=env)

                            if "<<<YAML_START>>>" in res.stdout:
                                new_y = res.stdout.split("<<<YAML_START>>>")[1].split("<<<YAML_END>>>")[0].strip()
                                st.session_state.pending_yaml = new_y
                                st.toast("分析完成！")
                            else:
                                st.error("AI调用失败，请检查API Key或网络")
                                st.code(res.stdout + "\n" + res.stderr)

            with b2:
                if st.session_state.pending_yaml:
                    if st.button("✅ 2. 确认并应用", type="primary", use_container_width=True):
                        st.session_state.yaml_content = st.session_state.pending_yaml
                        paths['config'].write_text(st.session_state.pending_yaml, encoding='utf-8')
                        st.session_state.pending_yaml = None
                        st.success("已保存！请重新运行匹配。")
                        time.sleep(1)
                        st.rerun()

            # 编辑器 / 对比视图
            if st.session_state.pending_yaml:
                st.info("👇 变更预览 (左：原版 | 右：新版)")

                # 计算差异统计
                old_lines = st.session_state.yaml_content.splitlines()
                new_lines = st.session_state.pending_yaml.splitlines()
                diff = difflib.unified_diff(old_lines, new_lines, lineterm="")
                added_count = sum(1 for line in diff if line.startswith('+') and not line.startswith('+++'))
                st.caption(f"✨ AI 建议新增约 **{added_count}** 行配置 (主要是新 Terms)")

                d1, d2 = st.columns(2)
                d1.code(st.session_state.yaml_content, language="yaml")
                d2.code(st.session_state.pending_yaml, language="yaml")
            else:
                txt = st.text_area("编辑器", value=st.session_state.yaml_content, height=400)
                if txt != st.session_state.yaml_content:
                    st.session_state.yaml_content = txt
                if st.button("💾 手动保存配置"):
                    paths['config'].write_text(txt, encoding='utf-8')
                    st.success("已保存")

        with c_view:
            st.write("**🔍 未覆盖样本**")
            st.dataframe(data['uncovered_df'].head(50), height=350, hide_index=True, use_container_width=True)

# -----------------------------------------------------------------------------
# 2️⃣ 训练与推理 (分步独立版 - 修正参数拼写)
# -----------------------------------------------------------------------------
elif page == "2️⃣ 训练与推理":
    st.title("⚙️ Step 02-05: 生产流水线 (分步执行)")
    script_path = str(ROOT / "scripts" / "route_b_sentiment" / "pipeline.py")


    # 定义检查点路径 (用于状态显示的辅助函数)
    def check_status(step_name):
        run_dir = OUTPUTS_DIR / domain / "runs" / run_id
        if step_name == "02":
            f = run_dir / "step02_pseudo" / "train_pseudolabel.parquet"
            if f.exists(): return f"✅ 已完成 (大小: {f.stat().st_size / 1024 / 1024:.2f} MB)"
        elif step_name == "03":
            # 检查是否有 Checkpoint 存档
            ckpt_dir = run_dir / "step03_model" / "ckpt"
            if ckpt_dir.exists() and any(p.name.startswith("checkpoint-") for p in ckpt_dir.iterdir()):
                return "🔄 训练中 (有存档)"
            # 检查是否有最终模型
            config = run_dir / "step03_model" / "config.json"
            if config.exists(): return "✅ 已完成 (模型已保存)"
        return "⬜ 未开始"


    # --- 顶部：全局设置 ---
    with st.container(border=True):
        st.subheader("全局设置")
        c1, c2 = st.columns(2)
        with c1:
            st.text_input("当前任务 ID", value=run_id, disabled=True)
        with c2:
            st.info(f"📂 数据将存储在: `outputs/{domain}/runs/{run_id}`")

    # --- 核心：分步 Tab ---
    tab2, tab3, tab4 = st.tabs([
        "🧠 Step 02: 伪标签 (API)",
        "🔥 Step 03: 训练 (GPU)",
        "🔎 Step 04: 验证 (推理)"
    ])

    # ==================== Step 02: Teacher (DeepSeek) ====================
    with tab2:
        st.subheader("Step 02: 生成伪标签 (Teacher)")
        st.caption("此步骤调用 DeepSeek/OpenAI 接口，**不消耗显卡**，不会导致过热。")
        st.caption(f"当前状态: {check_status('02')}")

        c1, c2, c3 = st.columns(3)
        with c1:
            sample_size = st.number_input(
                "🎯 采样数量 (Sample Size)",
                min_value=10, max_value=10000, value=500, step=100,
                help="向 API 发送多少条数据。测试建议 100，生产建议 2000+"
            )
        with c2:
            batch_size_api = st.number_input("API 批次 (Batch Size)", value=10, help="每批请求多少条")

        with c3:
            # --- 成本估算器 ---
            est_cost = (sample_size * 150) / 1000 * 0.00015  # 假设 $0.15 / 1M tokens
            st.metric("预计 Token", f"~{sample_size * 150:,}")
            st.caption(f"预计费用: < ${est_cost:.4f}")

        st.divider()

        if st.button("🚀 运行 Step 02 (生成数据)", type="primary"):
            cmd = [
                python_exe, "-u", script_path,
                "--domain", domain, "--run-id", run_id,  # <--- 修正：使用中划线 --run-id
                "--input-aspect-sentences", str(paths['aspect']),
                "--steps", "02",
                "--step02-max-rows", "0"
            ]
            # 注入环境变量
            os.environ["ABSA_SAMPLE_SIZE"] = str(sample_size)

            run_command_with_progress(cmd, desc="正在呼叫 DeepSeek 老师...")

    # ==================== Step 03: Student (Training) ====================
    with tab3:
        st.subheader("Step 03: 模型训练 (Student)")
        st.caption("⚠️ **高负载预警**：此步骤会满载显卡。请确保 Step 02 已完成。")

        # --- 检测续传状态 ---
        ckpt_dir_path = OUTPUTS_DIR / domain / "runs" / run_id / "step03_model" / "ckpt"
        last_ckpt_info = None

        if ckpt_dir_path.exists():
            try:
                # 寻找 checkpoint-XXX 文件夹
                ckpts = [p for p in ckpt_dir_path.iterdir() if p.is_dir() and p.name.startswith("checkpoint-")]
                if ckpts:
                    # 找数字最大的 (最新)
                    latest = max(ckpts, key=lambda p: int(p.name.split("-")[-1]))
                    step_num = latest.name.split("-")[-1]

                    # 找 timestamp
                    state_file = latest / "trainer_state.json"
                    ts = state_file.stat().st_mtime if state_file.exists() else latest.stat().st_mtime
                    time_str = datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")

                    last_ckpt_info = {
                        "step": step_num,
                        "time": time_str,
                        "path": str(latest)
                    }
            except Exception as e:
                print(f"Ckpt check error: {e}")

        # --- UI 布局 ---
        c_settings, c_actions = st.columns([1, 1])

        with c_settings:
            st.markdown("#### 1. 参数设置")
            # 模型选择
            model_map = {
                "hfl/chinese-macbert-base": "🏆 推荐: MacBERT Base",
                "bert-base-chinese": "🧊 轻量: BERT Base (防过热)",
            }
            selected_base_model = st.selectbox("基座模型", options=list(model_map.keys()),
                                               format_func=lambda x: model_map[x])

            # 硬件参数
            with st.expander("🔥 硬件参数 (防过热设置)", expanded=True):
                batch_size = st.select_slider("Batch Size", options=[1, 2, 4, 8, 16], value=4)
                grad_accum = st.select_slider("Grad Accum", options=[1, 2, 4, 8, 16], value=4)
                epochs = st.number_input("Epochs", value=3, min_value=1)
                st.caption(f"等效 Batch Size = {batch_size * grad_accum}")

        with c_actions:
            st.markdown("#### 2. 执行操作")

            # --- 场景 A: 存在旧存档 ---
            if last_ckpt_info:
                st.success(f"检测到存档: Step {last_ckpt_info['step']}")
                st.caption(f"存档时间: {last_ckpt_info['time']}")

                if st.button("▶️ 继续训练 (Resume)", type="primary", use_container_width=True):
                    st.info("正在恢复... (参数将自动沿用上次训练的配置)")
                    cmd = [
                        python_exe, "-u", script_path,
                        "--domain", domain, "--run-id", run_id,  # <--- 修正：使用中划线 --run-id
                        "--input-aspect-sentences", str(paths['aspect']),
                        "--steps", "03",
                        "--base-model", selected_base_model,
                        "--num-train-epochs", str(epochs),
                        "--batch-size", str(batch_size),
                        "--grad-accum", str(grad_accum),
                        "--resume"
                    ]
                    run_command_with_progress(cmd, desc=f"正在从 Step {last_ckpt_info['step']} 恢复...")

                if st.button("🗑️ 放弃旧进度，重新开始", type="secondary", use_container_width=True):
                    import shutil

                    try:
                        shutil.rmtree(ckpt_dir_path)
                        st.toast("已删除旧存档，请点击下方‘开始训练’")
                        time.sleep(1)
                        st.rerun()
                    except Exception as e:
                        st.error(f"删除失败: {e}")

            # --- 场景 B: 无存档 (或已删除) ---
            else:
                if st.button("🔥 开始新训练 (Start)", type="primary", use_container_width=True):
                    if "✅" not in check_status('02'):
                        st.error("请先在 Tab 1 完成 Step 02！")
                    else:
                        cmd = [
                            python_exe, "-u", script_path,
                            "--domain", domain, "--run-id", run_id,  # <--- 修正：使用中划线 --run-id
                            "--input-aspect-sentences", str(paths['aspect']),
                            "--steps", "03",
                            "--base-model", selected_base_model,
                            "--num-train-epochs", str(epochs),
                            "--batch-size", str(batch_size),
                            "--grad-accum", str(grad_accum)
                        ]
                        run_command_with_progress(cmd, desc="正在开始新训练...")

    # ==================== Step 04: Inference ====================
    with tab4:
        st.subheader("Step 04 & 05: 推理与报表")
        st.caption("使用训练好的模型进行全量预测。")

        default_model_path = ""
        potential_model = OUTPUTS_DIR / domain / "runs" / run_id / "step03_model"
        if potential_model.exists():
            default_model_path = str(potential_model)

        # 1. 模型路径
        model_path_input = st.text_input("模型路径", value=default_model_path)

        # 2. [新增] 性能与散热设置
        with st.expander("❄️ 性能与散热设置 (Performance & Cooling)", expanded=True):
            c_batch, c_cool = st.columns(2)
            with c_batch:
                infer_bs = st.select_slider(
                    "推理 Batch Size",
                    options=[4, 8, 16, 32, 64, 128],
                    value=8,
                    help="越小越稳定，越大越快（但显存发热大）。笔记本建议 8-16。"
                )
            with c_cool:
                enable_cool = st.checkbox(
                    "🧊 开启“散热喘息”模式",
                    value=True,
                    help="每批次计算后暂停 0.5 秒，防止显卡长期满载导致过热关机。"
                )
                cool_time = 0.5 if enable_cool else 0.0
                if enable_cool:
                    st.caption(f"✅ 已启用: 每次计算歇 {cool_time}s")

        st.divider()

        c_run, c_resume = st.columns([1, 1])

        # 公共参数构造
        base_cmd = [
            python_exe, "-u", script_path,
            "--domain", domain, "--run-id", run_id,
            "--input-aspect-sentences", str(paths['aspect']),
            "--steps", "04,05,web",
            "--reuse-model", model_path_input,
            # [新增] 注入 UI 参数
            "--step04-batch-size", str(infer_bs),
            "--step04-cool-down-time", str(cool_time)
        ]

        with c_run:
            if st.button("⚡ 重新推理 (清除旧数据)", type="primary", use_container_width=True):
                if not model_path_input:
                    st.error("未找到模型路径")
                else:
                    run_command_with_progress(base_cmd, desc="正在重新推理...")

        with c_resume:
            if st.button("▶️ 继续推理 (断点续传)", use_container_width=True):
                if not model_path_input:
                    st.error("未找到模型路径")
                else:
                    # 追加 resume
                    resume_cmd = base_cmd + ["--resume"]
                    run_command_with_progress(resume_cmd, desc="正在恢复推理...")

# -----------------------------------------------------------------------------
# 3️⃣ 数据看板 (DB版 - 包含正负面构成饼图)
# -----------------------------------------------------------------------------
elif page == "3️⃣ 数据看板 (DB版)":
    st.title("📈 结果洞察 (Database Driven)")

    db_path = paths['db']

    # 智能路径识别
    base_run_dir = OUTPUTS_DIR / domain / "runs" / run_id / "step04_pred"
    if (base_run_dir / "asc_pred_ds").exists():
        pred_dir = base_run_dir / "asc_pred_ds"
    else:
        pred_dir = base_run_dir

    # --- 1. 数据库同步区 ---
    with st.expander("🔄 数据同步 (Sync DB)", expanded=False):
        col_db1, col_db2 = st.columns([2, 1])
        with col_db1:
            st.info(f"数据库路径: `{db_path}`")
        with col_db2:
            if st.button("🚀 聚合最新结果入库", use_container_width=True, type="primary"):
                if not pred_dir.exists():
                    st.error(f"❌ 找不到推理结果目录: {pred_dir}")
                else:
                    cmd = [
                        python_exe, "-u", str(ROOT / "scripts" / "tools" / "aggregate_to_db.py"),
                        "--pred-ds", str(pred_dir),
                        "--db-path", str(db_path)
                    ]
                    if run_command_with_progress(cmd, desc="正在聚合 Parquet 到 SQLite..."):
                        st.success("✅ 入库完成！")
                        time.sleep(1)
                        st.rerun()

    # --- 2. 动态看板 ---
    if db_path.exists():
        try:
            conn = sqlite3.connect(str(db_path))

            # --- 筛选器 ---
            st.subheader("🔍 筛选条件")
            f_col1, f_col2 = st.columns(2)

            # 获取品牌列表
            try:
                brands = pd.read_sql(
                    "SELECT DISTINCT brand FROM daily_sentiment_stats WHERE brand IS NOT NULL AND brand != '' ORDER BY brand",
                    conn)['brand'].tolist()
            except:
                brands = []

            if not brands:
                st.warning("⚠️ 数据库中暂无有效品牌数据，请先点击上方“聚合”按钮或检查 Step 00 数据清洗。")
            else:
                with f_col1:
                    sel_brands = st.multiselect("选择品牌 (Brand)", brands, default=brands[:5])

                # 级联获取型号
                models = []
                if sel_brands:
                    ph = ",".join([f"'{b}'" for b in sel_brands])
                    models = pd.read_sql(
                        f"SELECT DISTINCT model FROM daily_sentiment_stats WHERE brand IN ({ph}) AND model IS NOT NULL ORDER BY model",
                        conn)['model'].tolist()
                with f_col2:
                    sel_models = st.multiselect("选择型号 (Model)", models, default=models[:10] if models else [])

                st.divider()

                if sel_brands and sel_models:
                    # 构造 SQL 条件
                    brands_ph = ",".join([f"'{x}'" for x in sel_brands])
                    models_ph = ",".join([f"'{x}'" for x in sel_models])
                    where_clause = f"brand IN ({brands_ph}) AND model IN ({models_ph})"

                    # --- 核心指标 KPI ---
                    kpi_sql = f"""
                        SELECT 
                            SUM(count) as total_cnt,
                            SUM(CASE WHEN sentiment='POS' THEN count ELSE 0 END) as pos_cnt,
                            SUM(CASE WHEN sentiment='NEG' THEN count ELSE 0 END) as neg_cnt
                        FROM daily_sentiment_stats
                        WHERE {where_clause}
                    """
                    df_kpi = pd.read_sql(kpi_sql, conn)
                    total = df_kpi['total_cnt'].iloc[0] or 0
                    pos_r = (df_kpi['pos_cnt'].iloc[0] or 0) / total if total > 0 else 0
                    neg_r = (df_kpi['neg_cnt'].iloc[0] or 0) / total if total > 0 else 0

                    k1, k2, k3, k4 = st.columns(4)
                    k1.metric("总声量 (Volume)", f"{total:,}")
                    k2.metric("正面率 (Pos Rate)", f"{pos_r:.1%}", delta_color="normal")
                    k3.metric("负面率 (Neg Rate)", f"{neg_r:.1%}", delta_color="inverse")
                    k4.metric("净推荐值 (NPS Proxy)", f"{(pos_r - neg_r) * 100:.1f}")

                    st.markdown("---")

                    # === 获取聚合数据用于画图 ===
                    pie_sql = f"""
                        SELECT aspect, sentiment, SUM(count) as count
                        FROM daily_sentiment_stats
                        WHERE {where_clause} AND aspect IS NOT NULL AND aspect != ''
                        GROUP BY 1, 2
                    """
                    df_pie = pd.read_sql(pie_sql, conn)

                    # === 第一排：总体情感 + 旭日图 ===
                    st.subheader("🥧 情感分布透视")
                    r1_c1, r1_c2 = st.columns([1, 1])

                    # 1. 总体情感饼图
                    with r1_c1:
                        st.markdown("##### 🟢 总体情感占比 (Global Sentiment)")
                        if not df_pie.empty:
                            df_global_pie = df_pie.groupby("sentiment")["count"].sum().reset_index()
                            fig_g_pie = px.pie(df_global_pie, values='count', names='sentiment',
                                               color='sentiment',
                                               color_discrete_map={'POS': '#10b981', 'NEG': '#ef4444',
                                                                   'NEU': '#9ca3af'},
                                               hole=0.4)
                            fig_g_pie.update_traces(textinfo='percent+label')
                            st.plotly_chart(fig_g_pie, use_container_width=True)
                        else:
                            st.caption("暂无数据")

                    # 2. 旭日图
                    with r1_c2:
                        st.markdown("##### ☀️ 各方面正负面分布 (Aspect Sunburst)")
                        if not df_pie.empty:
                            top_aspects = df_pie.groupby("aspect")["count"].sum().nlargest(15).index.tolist()
                            df_pie['aspect_clean'] = df_pie['aspect'].apply(
                                lambda x: x if x in top_aspects else 'Other')

                            fig_sun = px.sunburst(df_pie, path=['aspect_clean', 'sentiment'], values='count',
                                                  color='sentiment',
                                                  color_discrete_map={'POS': '#10b981', 'NEG': '#ef4444',
                                                                      'NEU': '#9ca3af', '(?)': '#ddd'})
                            st.plotly_chart(fig_sun, use_container_width=True)
                        else:
                            st.caption("暂无数据")

                    # === [NEW] 第二排：正负面评价的具体构成 ===
                    st.divider()
                    st.subheader("🎭 正/负面评价的具体构成")
                    st.caption("下图分别展示：在所有**好评**中各方面的占比，以及在所有**差评**中各方面的占比。")

                    pn_c1, pn_c2 = st.columns(2)


                    # 辅助函数：只取 Top 10，其他的合并为 Other，防止饼图太碎
                    def get_top_aspects_df(source_df, sentiment_label, top_n=12):
                        subset = source_df[source_df['sentiment'] == sentiment_label].copy()
                        if subset.empty: return subset

                        # 按数量排序
                        subset = subset.sort_values('count', ascending=False)

                        # 取前 N 个
                        top_items = subset.head(top_n)

                        # 计算 "Other"
                        other_count = subset.iloc[top_n:]['count'].sum()
                        if other_count > 0:
                            # 构造一行 Other 数据
                            other_row = pd.DataFrame(
                                {'aspect': ['Other'], 'sentiment': [sentiment_label], 'count': [other_count]})
                            return pd.concat([top_items, other_row], ignore_index=True)
                        return top_items


                    # 3. 正面构成饼图
                    with pn_c1:
                        st.markdown("##### 👍 正面评价都在夸什么 (Positive Mix)")
                        df_pos_pie = get_top_aspects_df(df_pie, 'POS')
                        if not df_pos_pie.empty:
                            # 使用 Pastel 色系，看起来比较柔和
                            fig_p = px.pie(df_pos_pie, values='count', names='aspect', hole=0.3,
                                           color_discrete_sequence=px.colors.qualitative.Pastel)
                            fig_p.update_traces(textposition='inside', textinfo='percent+label')
                            st.plotly_chart(fig_p, use_container_width=True)
                        else:
                            st.info("无正面评价数据")

                    # 4. 负面构成饼图
                    with pn_c2:
                        st.markdown("##### 👎 负面评价都在骂什么 (Negative Mix)")
                        df_neg_pie = get_top_aspects_df(df_pie, 'NEG')
                        if not df_neg_pie.empty:
                            # 使用 Set3 色系，与左边区分开
                            fig_n = px.pie(df_neg_pie, values='count', names='aspect', hole=0.3,
                                           color_discrete_sequence=px.colors.qualitative.Set3)
                            fig_n.update_traces(textposition='inside', textinfo='percent+label')
                            st.plotly_chart(fig_n, use_container_width=True)
                        else:
                            st.info("无负面评价数据")

                    st.markdown("---")

                    # === 第三排：堆叠条形图 ===
                    if not df_pie.empty:
                        st.markdown("##### 📊 各方面情感比例对比 (Stacked Bar)")
                        # 计算每个 Aspect 的总量，用于排序，只显示 Top 20
                        aspect_totals = df_pie.groupby('aspect')['count'].sum().sort_values(ascending=False).head(
                            20).index
                        df_bar = df_pie[df_pie['aspect'].isin(aspect_totals)]

                        fig_stack = px.bar(df_bar, x='aspect', y='count', color='sentiment',
                                           color_discrete_map={'POS': '#10b981', 'NEG': '#ef4444', 'NEU': '#9ca3af'},
                                           category_orders={"aspect": aspect_totals})
                        # 设为百分比堆叠，方便看“好评率”对比
                        fig_stack.update_layout(barnorm='percent', xaxis_title=None, yaxis_title="Percentage")
                        st.plotly_chart(fig_stack, use_container_width=True)

                    st.markdown("---")

                    # === 第四排：趋势与排名 ===
                    c1, c2 = st.columns([2, 1])

                    # 5. 每日趋势图
                    with c1:
                        st.markdown("##### 📅 声量与情感趋势 (Daily Trend)")
                        trend_sql = f"""
                            SELECT date, sentiment, SUM(count) as count
                            FROM daily_sentiment_stats
                            WHERE {where_clause}
                            GROUP BY 1, 2
                            ORDER BY 1
                        """
                        df_trend = pd.read_sql(trend_sql, conn)
                        fig_trend = px.line(df_trend, x='date', y='count', color='sentiment',
                                            color_discrete_map={'POS': '#10b981', 'NEG': '#ef4444', 'NEU': '#9ca3af'},
                                            markers=True)
                        st.plotly_chart(fig_trend, use_container_width=True)

                    # 6. 负面 Aspect 排名
                    with c2:
                        st.markdown("##### 🚨 Top 10 负面关注点")
                        aspect_sql = f"""
                            SELECT aspect, SUM(count) as cnt
                            FROM daily_sentiment_stats
                            WHERE {where_clause} AND sentiment='NEG'
                            GROUP BY 1
                            ORDER BY 2 DESC
                            LIMIT 10
                        """
                        df_aspect = pd.read_sql(aspect_sql, conn)
                        if not df_aspect.empty:
                            fig_bar = px.bar(df_aspect, x='cnt', y='aspect', orientation='h',
                                             color_discrete_sequence=['#ef4444'])
                            fig_bar.update_layout(yaxis={'categoryorder': 'total ascending'})
                            st.plotly_chart(fig_bar, use_container_width=True)
                        else:
                            st.info("无负面数据")

                else:
                    st.info("👈 请先在左侧选择品牌和型号以查看数据。")

            conn.close()
        except Exception as e:
            st.error(f"数据库读取错误: {e}")
            import traceback

            st.code(traceback.format_exc())
    else:
        st.info("等待数据库初始化...")