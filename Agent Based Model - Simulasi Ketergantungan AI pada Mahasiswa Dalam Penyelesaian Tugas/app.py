# -*- coding: utf-8 -*-
import streamlit as st
import mesa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

st.set_page_config(
    page_title="Ketergantungan AI Mahasiswa",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
    --bg: #f5f0e8;
    --bg-card: #ffffff;
    --ink: #1a1a1a;
    --ink-secondary: #4a4a4a;
    --ink-muted: #888888;
    --border: #1a1a1a;
    --yellow: #ffd166;
    --yellow-light: #fff3cc;
    --blue: #a8d8ff;
    --blue-dark: #5ba4e0;
    --pink: #ffb3c6;
    --pink-dark: #e8849b;
    --green: #b8e6c8;
    --green-dark: #4caf73;
    --shadow: 4px 4px 0px #1a1a1a;
    --shadow-sm: 3px 3px 0px #1a1a1a;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}

.stApp {
    background: var(--bg);
    color: var(--ink);
}

[data-testid="stSidebar"] {
    display: none !important;
}

.top-panel {
    background: var(--bg-card);
    border: 3px solid var(--border);
    border-radius: 4px;
    padding: 0.8rem 1.2rem;
    margin-bottom: 1rem;
    box-shadow: var(--shadow-sm);
}

[data-testid="stExpander"] .stSlider label,
[data-testid="stExpander"] .stSelectbox label,
[data-testid="stExpander"] p,
[data-testid="stExpander"] label,
[data-testid="stExpander"] .stMarkdown h5,
.stSlider label,
.stSelectbox label {
    color: var(--ink) !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 0.88rem !important;
    font-weight: 600 !important;
}

[data-testid="stMetric"] {
    background: var(--bg-card);
    border: 3px solid var(--border);
    border-radius: 4px;
    padding: 1rem 1.2rem;
    box-shadow: var(--shadow-sm);
}
[data-testid="stMetric"] label {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.72rem !important;
    color: var(--ink-muted) !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
[data-testid="stMetricValue"] {
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 700;
    font-size: 1.8rem !important;
    color: var(--ink) !important;
}
[data-testid="stMetricDelta"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.75rem !important;
    font-weight: 500 !important;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 0;
    border-bottom: 3px solid var(--border);
    background: transparent;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 0.88rem;
    font-weight: 700;
    color: var(--ink-muted);
    padding: 0.7rem 1.4rem;
    border: none;
    background: transparent;
}
.stTabs [data-baseweb="tab"]:hover {
    color: var(--ink);
}
.stTabs [aria-selected="true"] {
    color: var(--ink) !important;
    background: var(--yellow) !important;
    border: 3px solid var(--border) !important;
    border-bottom: 3px solid var(--yellow) !important;
    margin-bottom: -3px;
    border-radius: 4px 4px 0 0;
}

h1 {
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 700 !important;
    font-size: 2.2rem !important;
    color: var(--ink) !important;
    line-height: 1.15 !important;
    letter-spacing: -0.02em;
}
h2 {
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600;
    font-size: 1.1rem !important;
    color: var(--ink-secondary) !important;
}
h3 {
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 1.15rem !important;
    color: var(--ink) !important;
    font-weight: 700 !important;
}

hr {
    border: none;
    border-top: 3px solid var(--border);
    margin: 1.5rem 0;
}

.stButton > button {
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 0.9rem;
    font-weight: 700;
    background: var(--yellow);
    color: var(--ink);
    border: 3px solid var(--border);
    border-radius: 4px;
    padding: 0.5rem 1.8rem;
    box-shadow: var(--shadow-sm);
    transition: all 0.1s ease;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}
.stButton > button:hover {
    box-shadow: 1px 1px 0px var(--border);
    transform: translate(2px, 2px);
}

.stCaption, small {
    font-family: 'JetBrains Mono', monospace !important;
    color: var(--ink-muted) !important;
    font-size: 0.78rem !important;
}

.stAlert {
    background: var(--yellow-light);
    border: 3px solid var(--border);
    color: var(--ink);
    font-family: 'Inter', sans-serif;
    border-radius: 4px;
    box-shadow: var(--shadow-sm);
}

.neo-card {
    background: var(--bg-card);
    border: 3px solid var(--border);
    border-radius: 4px;
    padding: 1.2rem 1.5rem;
    margin: 0.8rem 0;
    box-shadow: var(--shadow);
}

.neo-card-yellow {
    background: var(--yellow);
    border: 3px solid var(--border);
    border-radius: 4px;
    padding: 1rem 1.5rem;
    margin: 0.8rem 0;
    box-shadow: var(--shadow);
}

.neo-card-blue {
    background: var(--blue);
    border: 3px solid var(--border);
    border-radius: 4px;
    padding: 1rem 1.5rem;
    margin: 0.8rem 0;
    box-shadow: var(--shadow);
}

.neo-card-pink {
    background: var(--pink);
    border: 3px solid var(--border);
    border-radius: 4px;
    padding: 1rem 1.5rem;
    margin: 0.8rem 0;
    box-shadow: var(--shadow);
}

.state-tag {
    display: inline-block;
    padding: 3px 12px;
    border: 2px solid var(--border);
    border-radius: 3px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    font-weight: 600;
    margin: 3px 4px;
    box-shadow: 2px 2px 0px var(--border);
}
.tag-mandiri { background: var(--green); color: var(--ink); }
.tag-hibrida { background: var(--yellow); color: var(--ink); }
.tag-penuh { background: var(--pink); color: var(--ink); }

.formula-strip {
    background: var(--ink);
    color: #ffffff;
    border: 3px solid var(--border);
    border-radius: 4px;
    padding: 0.8rem 1.2rem;
    margin: 0.8rem 0;
    text-align: center;
    font-family: 'JetBrains Mono', monospace;
    font-size: 1rem;
    font-weight: 600;
    box-shadow: var(--shadow-sm);
}

[data-testid="stExpander"] {
    border: 3px solid var(--border) !important;
    border-radius: 4px !important;
    background: var(--bg-card) !important;
    box-shadow: var(--shadow-sm) !important;
}

div[data-testid="stNumberInput"] input {
    font-family: 'JetBrains Mono', monospace !important;
    background: var(--bg-card);
    color: var(--ink);
    border: 2px solid var(--border);
    border-radius: 3px;
}

.section-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--ink-muted);
    margin-bottom: 0.3rem;
}
</style>
""", unsafe_allow_html=True)


PALETTE = {
    "bg":       "#f5f0e8",
    "card":     "#ffffff",
    "ink":      "#1a1a1a",
    "grid":     "#e8e3db",
    "muted":    "#888888",
    "blue":     "#5ba4e0",
    "green":    "#4caf73",
    "yellow":   "#e0a800",
    "orange":   "#e08a3a",
    "red":      "#d94f4f",
    "purple":   "#8b6cc1",
    "pink":     "#d97090",
}


def apply_theme(ax, fig):
    fig.patch.set_facecolor(PALETTE["bg"])
    ax.set_facecolor(PALETTE["card"])
    ax.tick_params(colors=PALETTE["muted"], labelsize=9, width=1.5)
    ax.xaxis.label.set_color(PALETTE["ink"])
    ax.yaxis.label.set_color(PALETTE["ink"])
    ax.title.set_color(PALETTE["ink"])
    for spine in ax.spines.values():
        spine.set_edgecolor(PALETTE["ink"])
        spine.set_linewidth(2)
    ax.grid(True, color=PALETTE["grid"], linewidth=0.8, linestyle="-", alpha=0.8)
    ax.set_axisbelow(True)
    ax.xaxis.label.set_fontsize(10)
    ax.yaxis.label.set_fontsize(10)
    ax.xaxis.label.set_fontweight("600")
    ax.yaxis.label.set_fontweight("600")
    ax.title.set_fontsize(12)
    ax.title.set_fontweight("700")


class StudentAgent(mesa.Agent):
    def __init__(self, model, a_init, r, f):
        super().__init__(model)
        self.A = a_init
        self.R = r
        self.F = f
        self.history = [a_init]
        self.state = self._get_state()

    def _get_state(self):
        if self.A < 0.4:
            return "Mandiri"
        elif self.A < 0.7:
            return "Hibrida"
        return "Ketergantungan_Penuh"

    def step(self):
        T = self.model.task_difficulty

        D = self.model.detection_power
        if self.model.scenario_type == "tanpa_intervensi":
            D = 0.0
        elif self.model.scenario_type == "reaktif":
            D = D if self.A > 0.8 else 0.0
        elif self.model.scenario_type == "preventif":
            D = D * 1.2

        pressure_up = T * self.F
        deterrence = self.R * D * self.A
        delta_A = pressure_up - deterrence
        self.A = max(0.0, min(1.0, self.A + delta_A))
        self.state = self._get_state()
        self.history.append(self.A)


class AIDependencyModel(mesa.Model):
    def __init__(self, num_agents=100, task_difficulty=0.12,
                 detection_power=0.10, scenario_type="normal",
                 high_fomo=False, r_mean=0.5,
                 a_init_mean=0.25, f_mean=0.4,
                 total_weeks=16):
        super().__init__()
        self.num_agents = num_agents
        self.task_difficulty = task_difficulty
        self.detection_power = detection_power
        self.scenario_type = scenario_type
        self.total_weeks = total_weeks
        self.current_step = 0

        actual_f = 0.75 if high_fomo else f_mean
        for _ in range(num_agents):
            StudentAgent(
                model=self,
                a_init=np.clip(np.random.normal(a_init_mean, 0.12), 0.0, 0.5),
                r=np.clip(np.random.normal(r_mean, 0.15), 0.1, 1.0),
                f=np.clip(np.random.normal(actual_f, 0.2), 0.1, 1.0),
            )

        self.datacollector = mesa.DataCollector(
            model_reporters={
                "Rata_rata_A": lambda m: np.mean([a.A for a in m.agents]),
                "Std_A": lambda m: np.std([a.A for a in m.agents]),
                "Mandiri": lambda m: sum(1 for a in m.agents if a.state == "Mandiri"),
                "Hibrida": lambda m: sum(1 for a in m.agents if a.state == "Hibrida"),
                "Ketergantungan_Penuh": lambda m: sum(1 for a in m.agents if a.state == "Ketergantungan_Penuh"),
                "Max_A": lambda m: np.max([a.A for a in m.agents]),
                "Min_A": lambda m: np.min([a.A for a in m.agents]),
            }
        )

    def step(self):
        self.current_step += 1
        agents = list(self.agents)
        random.shuffle(agents)
        for agent in agents:
            agent.step()
        self.datacollector.collect(self)


def run_simulation(T, D, scenario_type="normal", high_fomo=False,
                   num_agents=100, weeks=16,
                   r_mean=0.5, a_init_mean=0.25, f_mean=0.4):
    model = AIDependencyModel(
        num_agents=num_agents,
        task_difficulty=T,
        detection_power=D,
        scenario_type=scenario_type,
        high_fomo=high_fomo,
        r_mean=r_mean,
        a_init_mean=a_init_mean,
        f_mean=f_mean,
        total_weeks=weeks,
    )
    for _ in range(weeks):
        model.step()
    data = model.datacollector.get_model_vars_dataframe()
    data.index = data.index + 1
    return data, model


def monte_carlo(T, D, runs=50, scenario_type="normal", high_fomo=False,
                r_mean=0.5, a_init_mean=0.25, f_mean=0.4):
    frames = []
    for _ in range(runs):
        model = AIDependencyModel(
            num_agents=100,
            task_difficulty=T,
            detection_power=D,
            scenario_type=scenario_type,
            high_fomo=high_fomo,
            r_mean=r_mean,
            a_init_mean=a_init_mean,
            f_mean=f_mean,
        )
        for _ in range(16):
            model.step()
        df = model.datacollector.get_model_vars_dataframe()
        df["Minggu"] = df.index + 1
        frames.append(df)
    return pd.concat(frames)


st.markdown("# Simulasi Dinamika Ketergantungan Mahasiswa Informatika pada Generative AI dalam Penyelesaian Tugas")

st.markdown("""
<div class="neo-card" style="margin-top:0.5rem;">
    <p style="font-family:'Inter',sans-serif; font-size:0.92rem; margin:0; color:#4a4a4a; line-height:1.7;">
        Model Agent-Based (Mesa) yang mensimulasikan dinamika ketergantungan AI
        di kalangan mahasiswa selama 16 minggu perkuliahan. Setiap agen merupakan mahasiswa
        dengan parameter unik. Stresor berupa tugas &amp; deadline mendorong jalan pintas AI,
        sementara deteksi dosen bertindak sebagai intervensi penekan.
    </p>
    <div style="margin-top:0.6rem;">
        <span class="state-tag tag-mandiri">Mandiri (K&lt;0.4)</span>
        <span class="state-tag tag-hibrida">Hibrida (0.4≤K&lt;0.7)</span>
        <span class="state-tag tag-penuh">Ketergantungan Penuh (K≥0.7)</span>
    </div>
</div>
""", unsafe_allow_html=True)

with st.expander("Panel Kontrol", expanded=True):
    st.markdown("""
    <div class="formula-strip" style="margin-top:0; margin-bottom:0.8rem;">
        K(t+1) = K(t) + (T x F) - (R x D x K)
    </div>
    """, unsafe_allow_html=True)
    st.caption("K=Level Ketergantungan AI (0–1), T=Kesulitan Tugas, F=FOMO, R=Pemahaman, D=Deteksi Dosen")

    pc1, pc2, pc3, pc4 = st.columns(4)

    with pc1:
        st.markdown("##### Atribut Agen")
        num_students = st.slider("Jumlah Mahasiswa", 10, 500, 100, 10,
                                  help="Total individu dalam simulasi")
        a_init_mean = st.slider("Ketergantungan Awal (A0)", 0.05, 0.50, 0.25, 0.05,
                                 help="Rata-rata level ketergantungan AI awal")

    with pc2:
        st.markdown("##### Profil Mahasiswa")
        r_mean = st.slider("Pemahaman Fundamental (R)", 0.1, 1.0, 0.5, 0.05,
                            help="Kapasitas internal mahasiswa menahan godaan AI")
        f_mean = st.slider("FOMO / Pengaruh Teman (F)", 0.1, 1.0, 0.4, 0.05,
                            help="Pengali mendorong perilaku copy-paste")

    with pc3:
        st.markdown("##### Variabel Lingkungan")
        task_diff = st.slider("Kesulitan Tugas & Deadline (T)", 0.0, 0.30, 0.12, 0.01,
                               help="Beban tugas dan tekanan waktu")

    with pc4:
        st.markdown("##### Intervensi & Monte Carlo")
        detection_power = st.slider("Deteksi Dosen (D)", 0.0, 0.5, 0.10, 0.01,
                                     help="Responsi, AI detector, pengurangan nilai")
        mc_runs = st.slider("Jumlah Iterasi MC", 10, 200, 50, 10,
                             help="Pengulangan simulasi untuk analisis statistik")

col_run_center = st.columns([2, 1, 2])
with col_run_center[1]:
    run_btn = st.button("JALANKAN")

st.markdown("---")

if "results" not in st.session_state or run_btn:
    with st.spinner("Menjalankan simulasi..."):
        data_main, model_main = run_simulation(
            task_diff, detection_power,
            num_agents=num_students, r_mean=r_mean,
            a_init_mean=a_init_mean, f_mean=f_mean
        )

        res_no, _ = run_simulation(T=task_diff, D=0.0,
                                    scenario_type="tanpa_intervensi",
                                    r_mean=r_mean, a_init_mean=a_init_mean,
                                    f_mean=f_mean)
        res_reaktif, _ = run_simulation(T=task_diff, D=detection_power,
                                         scenario_type="reaktif",
                                         r_mean=r_mean, a_init_mean=a_init_mean,
                                         f_mean=f_mean)
        res_preventif, _ = run_simulation(T=task_diff, D=detection_power,
                                           scenario_type="preventif",
                                           r_mean=r_mean, a_init_mean=a_init_mean,
                                           f_mean=f_mean)
        res_fomo, _ = run_simulation(T=task_diff, D=detection_power,
                                      high_fomo=True,
                                      r_mean=r_mean, a_init_mean=a_init_mean,
                                      f_mean=f_mean)

        all_scenarios = pd.concat([
            res_no.assign(Skenario="Tanpa Intervensi", Minggu=res_no.index),
            res_reaktif.assign(Skenario="Reaktif (A>0.8)", Minggu=res_reaktif.index),
            res_preventif.assign(Skenario="Preventif (Rutin)", Minggu=res_preventif.index),
            res_fomo.assign(Skenario="FOMO Tinggi", Minggu=res_fomo.index),
        ])

        mc_no = monte_carlo(T=task_diff, D=0.0,
                             scenario_type="tanpa_intervensi",
                             runs=mc_runs,
                             r_mean=r_mean, a_init_mean=a_init_mean,
                             f_mean=f_mean)
        mc_ada = monte_carlo(T=task_diff, D=detection_power,
                              scenario_type="preventif",
                              runs=mc_runs,
                              r_mean=r_mean, a_init_mean=a_init_mean,
                              f_mean=f_mean)

        agent_data = []
        for agent in model_main.agents:
            for week_idx, a_val in enumerate(agent.history):
                agent_data.append({
                    "Agent": agent.unique_id,
                    "Minggu": week_idx,
                    "A": a_val,
                    "R": agent.R,
                    "F": agent.F,
                })
        agent_df = pd.DataFrame(agent_data)

        st.session_state.results = {
            "main": data_main,
            "model": model_main,
            "scenarios": all_scenarios,
            "mc_no": mc_no,
            "mc_ada": mc_ada,
            "agent_df": agent_df,
        }

res = st.session_state.results
data_main = res["main"]
all_scen = res["scenarios"]
mc_no = res["mc_no"]
mc_ada = res["mc_ada"]
agent_df = res["agent_df"]

final = data_main.iloc[-1]
w1_A = data_main.iloc[0]["Rata_rata_A"]
w16_A = final["Rata_rata_A"]
delta = w16_A - w1_A

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Ketergantungan AI (Minggu 16)", f"{w16_A:.3f}", f"{delta:+.3f}")
m2.metric("Mandiri", f"{int(final['Mandiri'])} mhs")
m3.metric("Hibrida", f"{int(final['Hibrida'])} mhs")
m4.metric("Ketergantungan Penuh", f"{int(final['Ketergantungan_Penuh'])} mhs")
pct_danger = final["Ketergantungan_Penuh"] / num_students * 100
m5.metric("Zona Bahaya", f"{pct_danger:.1f}%")

st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs([
    "Simulasi Utama",
    "Komparasi 4 Skenario",
    "Monte Carlo",
    "Eksplorasi Agen",
])


with tab1:
    st.markdown("### Hasil Simulasi")

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.subplots_adjust(wspace=0.3, hspace=0.42, left=0.07, right=0.95, top=0.94, bottom=0.07)

    ax1 = axes[0, 0]
    ax1.plot(data_main.index, data_main["Rata_rata_A"],
             color=PALETTE["blue"], linewidth=2.5, zorder=5)
    ax1.fill_between(data_main.index,
                      data_main["Rata_rata_A"] - data_main["Std_A"],
                      data_main["Rata_rata_A"] + data_main["Std_A"],
                      color=PALETTE["blue"], alpha=0.15)
    ax1.axhline(0.7, color=PALETTE["red"], linestyle="--", linewidth=1.5, alpha=0.8,
                 label="Batas Ketergantungan Penuh (0.7)")
    ax1.axhline(0.4, color=PALETTE["yellow"], linestyle="--", linewidth=1.5, alpha=0.8,
                 label="Batas Hibrida (0.4)")
    ax1.set_title("Rata-rata Tingkat Ketergantungan AI")
    ax1.set_xlabel("Minggu")
    ax1.set_ylabel("Level Ketergantungan (K)")
    ax1.set_xlim(1, 16)
    ax1.set_ylim(0, 1.05)
    ax1.set_xticks(range(1, 17))
    ax1.legend(fontsize=7.5, loc="upper left", framealpha=0.9, edgecolor=PALETTE["ink"])
    apply_theme(ax1, fig)

    ax2 = axes[0, 1]
    ax2.stackplot(data_main.index,
                   data_main["Mandiri"],
                   data_main["Hibrida"],
                   data_main["Ketergantungan_Penuh"],
                   labels=["Mandiri", "Hibrida", "Ketergantungan Penuh"],
                   colors=[PALETTE["green"], PALETTE["yellow"], PALETTE["red"]],
                   alpha=0.8)
    ax2.set_title("Distribusi Status Mahasiswa")
    ax2.set_xlabel("Minggu")
    ax2.set_ylabel("Jumlah Mahasiswa")
    ax2.set_xlim(1, 16)
    ax2.set_xticks(range(1, 17))
    ax2.legend(fontsize=7.5, loc="upper left", framealpha=0.9, edgecolor=PALETTE["ink"])
    apply_theme(ax2, fig)

    ax3 = axes[1, 0]
    ax3.plot(data_main.index, data_main["Max_A"],
             color=PALETTE["red"], linewidth=2, label="Maks K", linestyle="-.")
    ax3.plot(data_main.index, data_main["Rata_rata_A"],
             color=PALETTE["blue"], linewidth=2.5, label="Rata-rata K")
    ax3.plot(data_main.index, data_main["Min_A"],
             color=PALETTE["green"], linewidth=2, label="Min K", linestyle="-.")
    ax3.fill_between(data_main.index, data_main["Min_A"], data_main["Max_A"],
                      color=PALETTE["blue"], alpha=0.08)
    ax3.set_title("Rentang Ketergantungan (Min - Maks)")
    ax3.set_xlabel("Minggu")
    ax3.set_ylabel("Level Ketergantungan (K)")
    ax3.set_xlim(1, 16)
    ax3.set_ylim(0, 1.05)
    ax3.set_xticks(range(1, 17))
    ax3.legend(fontsize=7.5, framealpha=0.9, edgecolor=PALETTE["ink"])
    apply_theme(ax3, fig)

    ax4 = axes[1, 1]
    last_week = agent_df["Minggu"].max()
    last_agents = agent_df[agent_df["Minggu"] == last_week]
    ax4.hist(last_agents["A"], bins=20, color=PALETTE["blue"],
             alpha=0.8, edgecolor=PALETTE["ink"], linewidth=1)
    ax4.axvline(0.7, color=PALETTE["red"], linestyle="--", linewidth=1.5,
                 label="Batas Ketergantungan Penuh")
    ax4.axvline(0.4, color=PALETTE["yellow"], linestyle="--", linewidth=1.5,
                 label="Batas Hibrida")
    ax4.set_title("Distribusi Ketergantungan Mahasiswa")
    ax4.set_xlabel("Level Ketergantungan (K)")
    ax4.set_ylabel("Jumlah Mahasiswa")
    ax4.legend(fontsize=7.5, framealpha=0.9, edgecolor=PALETTE["ink"])
    apply_theme(ax4, fig)

    st.pyplot(fig)
    plt.close(fig)

    with st.expander("Lihat Data Mentah"):
        display_df = data_main.copy()
        display_df.index.name = "Minggu"
        display_df = display_df.rename(columns={
            "Rata_rata_A": "Rata-rata K",
            "Std_A": "Std K",
            "Max_A": "Maks K",
            "Min_A": "Min K",
            "Ketergantungan_Penuh": "Ketergantungan Penuh",
        })
        st.dataframe(
            display_df.style
                .format("{:.3f}", subset=["Rata-rata K", "Std K", "Maks K", "Min K"])
                .format("{:.0f}", subset=["Mandiri", "Hibrida", "Ketergantungan Penuh"]),
            use_container_width=True,
        )


with tab2:
    st.markdown("### Komparasi 4 Skenario What-If")

    st.markdown("""
    <div class="neo-card" style="line-height:1.8;">
        <span class="section-label">Deskripsi Skenario</span><br>
        <b>1. Tanpa Intervensi</b> — Tidak ada deteksi dosen, mahasiswa bebas copy-paste AI.<br>
        <b>2. Reaktif</b> — Dosen baru bertindak saat ketergantungan sudah parah (A &gt; 0.8).<br>
        <b>3. Preventif</b> — Responsi & AI detector diterapkan rutin sejak awal (D x 1.2).<br>
        <b>4. FOMO Tinggi</b> — Lingkungan sosial sangat mendorong penggunaan AI (F tinggi).
    </div>
    """, unsafe_allow_html=True)

    scenario_colors = {
        "Tanpa Intervensi": PALETTE["red"],
        "Reaktif (A>0.8)": PALETTE["orange"],
        "Preventif (Rutin)": PALETTE["green"],
        "FOMO Tinggi": PALETTE["purple"],
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2))
    fig.subplots_adjust(wspace=0.28, left=0.06, right=0.95, top=0.9, bottom=0.14)

    for scenario, grp in all_scen.groupby("Skenario"):
        ax1.plot(grp["Minggu"], grp["Rata_rata_A"],
                 color=scenario_colors.get(scenario, PALETTE["blue"]),
                 linewidth=2.5, label=scenario)

    ax1.axhline(0.7, color=PALETTE["red"], linestyle=":", linewidth=1.2, alpha=0.6,
                 label="Batas Ketergantungan Penuh")
    ax1.set_title("Dinamika Ketergantungan AI (4 Skenario")
    ax1.set_xlabel("Minggu Perkuliahan")
    ax1.set_ylabel("Rata-rata Level Ketergantungan (K)")
    ax1.set_xlim(1, 16)
    ax1.set_ylim(0, 1.05)
    ax1.set_xticks(range(1, 17))
    ax1.legend(fontsize=7.5, framealpha=0.9, edgecolor=PALETTE["ink"])
    apply_theme(ax1, fig)

    scenarios_list = ["Tanpa Intervensi", "Reaktif (A>0.8)", "Preventif (Rutin)", "FOMO Tinggi"]
    w16_kp = []
    w16_names = []
    for scen_name in scenarios_list:
        grp = all_scen[all_scen["Skenario"] == scen_name]
        w16 = grp[grp["Minggu"] == 16]
        if len(w16) > 0:
            w16_kp.append(int(w16.iloc[0]["Ketergantungan_Penuh"]))
            w16_names.append(scen_name.replace(" ", "\n"))

    colors = [scenario_colors.get(s, PALETTE["blue"]) for s in scenarios_list]
    bars = ax2.bar(range(len(w16_kp)), w16_kp, color=colors, alpha=0.9,
                    edgecolor=PALETTE["ink"], linewidth=2, width=0.6)
    ax2.set_xticks(range(len(w16_names)))
    ax2.set_xticklabels(w16_names, fontsize=8)
    ax2.set_title("Mahasiswa Ketergantungan Penuh")
    ax2.set_ylabel("Jumlah Mahasiswa")
    for bar, val in zip(bars, w16_kp):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 str(val), ha='center', va='bottom', fontsize=11,
                 fontweight='bold', color=PALETTE["ink"])
    apply_theme(ax2, fig)

    st.pyplot(fig)
    plt.close(fig)

    st.markdown("### Ringkasan Minggu 16")
    summary_rows = []
    for scen_name in scenarios_list:
        grp = all_scen[all_scen["Skenario"] == scen_name]
        w16 = grp[grp["Minggu"] == 16]
        if len(w16) > 0:
            last = w16.iloc[0]
            summary_rows.append({
                "Skenario": scen_name,
                "Rata-rata K": f"{last['Rata_rata_A']:.3f}",
                "Mandiri": int(last["Mandiri"]),
                "Hibrida": int(last["Hibrida"]),
                "Ketergantungan Penuh": int(last["Ketergantungan_Penuh"]),
            })
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)


with tab3:
    st.markdown("### Analisis Monte Carlo")
    st.caption(f"Menggunakan {mc_runs} iterasi per kondisi | Garis = rata-rata, area = +/-1 standar deviasi")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2))
    fig.subplots_adjust(wspace=0.28, left=0.07, right=0.95, top=0.9, bottom=0.14)

    for mc_data, label, color in [
        (mc_no, "Tanpa Intervensi", PALETTE["red"]),
        (mc_ada, "Dosen Aktif (Preventif)", PALETTE["green"]),
    ]:
        grouped = mc_data.groupby("Minggu")["Rata_rata_A"]
        mean = grouped.mean()
        std = grouped.std()
        ax1.plot(mean.index, mean.values, color=color, linewidth=2.5, label=label)
        ax1.fill_between(mean.index, mean - std, mean + std, color=color, alpha=0.15)

    ax1.axhline(0.7, color=PALETTE["red"], linestyle=":", linewidth=1.2, alpha=0.6)
    ax1.set_title("Rata-rata Ketergantungan (Distribusi Monte Carlo)")
    ax1.set_xlabel("Minggu Perkuliahan")
    ax1.set_ylabel("Rata-rata Ketergantungan (K)")
    ax1.set_xlim(1, 16)
    ax1.set_xticks(range(1, 17))
    ax1.legend(fontsize=8.5, framealpha=0.9, edgecolor=PALETTE["ink"])
    apply_theme(ax1, fig)

    w16_no = mc_no[mc_no["Minggu"] == 16]["Ketergantungan_Penuh"]
    w16_ada = mc_ada[mc_ada["Minggu"] == 16]["Ketergantungan_Penuh"]

    all_vals = pd.concat([w16_no, w16_ada])
    bins = np.linspace(max(0, all_vals.min() - 3), all_vals.max() + 3, 22)
    ax2.hist(w16_no, bins=bins, color=PALETTE["red"], alpha=0.6,
              label="Tanpa Intervensi", edgecolor=PALETTE["ink"], linewidth=0.8)
    ax2.hist(w16_ada, bins=bins, color=PALETTE["green"], alpha=0.6,
              label="Dosen Aktif", edgecolor=PALETTE["ink"], linewidth=0.8)
    ax2.axvline(w16_no.mean(), color=PALETTE["red"], linestyle="--", linewidth=1.8)
    ax2.axvline(w16_ada.mean(), color=PALETTE["green"], linestyle="--", linewidth=1.8)
    ax2.set_title("Distribusi Ketergantungan Penuh")
    ax2.set_xlabel("Jumlah Mahasiswa")
    ax2.set_ylabel("Frekuensi")
    ax2.legend(fontsize=8.5, framealpha=0.9, edgecolor=PALETTE["ink"])
    apply_theme(ax2, fig)

    st.pyplot(fig)
    plt.close(fig)

    st.markdown("### Analitik Statistik (Minggu 16)")
    c1, c2, c3 = st.columns(3)
    pred_no = w16_no.mean()
    pred_ada = w16_ada.mean()
    selisih = pred_no - pred_ada

    c1.metric("Tanpa Intervensi", f"{pred_no:.1f} mhs",
              help="Rata-rata mahasiswa ketergantungan penuh dari Monte Carlo")
    c2.metric("Dosen Aktif", f"{pred_ada:.1f} mhs",
              f"-{selisih:.1f} dari tanpa intervensi")
    efektivitas = (selisih / pred_no * 100) if pred_no > 0 else 0
    c3.metric("Efektivitas Deteksi", f"{efektivitas:.1f}% pengurangan")

    with st.expander("Statistik Lengkap Monte Carlo"):
        mc_stats = pd.DataFrame({
            "Metrik": ["Rata-rata KP", "Std", "Min", "Max", "Median"],
            "Tanpa Intervensi": [f"{w16_no.mean():.1f}", f"{w16_no.std():.1f}",
                                  f"{w16_no.min():.0f}", f"{w16_no.max():.0f}", f"{w16_no.median():.0f}"],
            "Dosen Aktif": [f"{w16_ada.mean():.1f}", f"{w16_ada.std():.1f}",
                             f"{w16_ada.min():.0f}", f"{w16_ada.max():.0f}", f"{w16_ada.median():.0f}"],
        })
        st.dataframe(mc_stats, use_container_width=True, hide_index=True)


with tab4:
    st.markdown("### Eksplorasi Individual Agen")

    col_f1, col_f2 = st.columns(2)
    with col_f1:
        num_sample = st.slider("Jumlah sampel agen", 5, 50, 15, 5)
    with col_f2:
        sort_by = st.selectbox("Urutkan berdasarkan", [
            "Ketergantungan Akhir (Tertinggi)",
            "Ketergantungan Akhir (Terendah)",
            "Pemahaman Fundamental (Tertinggi)",
            "FOMO Tertinggi",
        ])

    last_week = agent_df["Minggu"].max()
    final_agents = agent_df[agent_df["Minggu"] == last_week].copy()

    if sort_by == "Ketergantungan Akhir (Tertinggi)":
        final_agents = final_agents.nlargest(num_sample, "A")
    elif sort_by == "Ketergantungan Akhir (Terendah)":
        final_agents = final_agents.nsmallest(num_sample, "A")
    elif sort_by == "Pemahaman Fundamental (Tertinggi)":
        final_agents = final_agents.nlargest(num_sample, "R")
    else:
        final_agents = final_agents.nlargest(num_sample, "F")

    selected_ids = final_agents["Agent"].tolist()
    sample_df = agent_df[agent_df["Agent"].isin(selected_ids)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2))
    fig.subplots_adjust(wspace=0.28, left=0.07, right=0.95, top=0.9, bottom=0.14)

    cmap_vals = plt.cm.cool(np.linspace(0.15, 0.85, len(selected_ids)))
    for agent_id, color in zip(selected_ids, cmap_vals):
        agent_hist = sample_df[sample_df["Agent"] == agent_id]
        ax1.plot(agent_hist["Minggu"], agent_hist["A"],
                 color=color, linewidth=1.3, alpha=0.75)

    ax1.axhline(0.7, color=PALETTE["red"], linestyle="--", linewidth=1.2, alpha=0.7)
    ax1.axhline(0.4, color=PALETTE["yellow"], linestyle="--", linewidth=1.2, alpha=0.7)
    ax1.set_title(f"Trajektori {num_sample} Mahasiswa Terpilih")
    ax1.set_xlabel("Minggu")
    ax1.set_ylabel("Level Ketergantungan (K)")
    ax1.set_xlim(0, 16)
    ax1.set_ylim(0, 1.05)
    apply_theme(ax1, fig)

    sc = ax2.scatter(final_agents["R"], final_agents["A"],
                      c=final_agents["F"], cmap="YlOrRd", s=90, alpha=0.85,
                      edgecolors=PALETTE["ink"], linewidth=1, zorder=5)
    ax2.set_title("Pemahaman (R) vs Ketergantungan (K)")
    ax2.set_xlabel("Pemahaman Fundamental (R)")
    ax2.set_ylabel("Ketergantungan Akhir (K)")
    ax2.axhline(0.7, color=PALETTE["red"], linestyle="--", linewidth=1, alpha=0.6)
    ax2.axhline(0.4, color=PALETTE["yellow"], linestyle="--", linewidth=1, alpha=0.6)
    sm = plt.cm.ScalarMappable(cmap="YlOrRd",
                                norm=plt.Normalize(final_agents["F"].min(), final_agents["F"].max()))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax2, shrink=0.8, pad=0.02)
    cbar.set_label("FOMO (F)", fontsize=9, color=PALETTE["muted"])
    cbar.ax.tick_params(labelsize=8, colors=PALETTE["muted"])
    cbar.outline.set_edgecolor(PALETTE["ink"])
    cbar.outline.set_linewidth(1.5)
    apply_theme(ax2, fig)

    st.pyplot(fig)
    plt.close(fig)

    with st.expander("Data Mahasiswa Terpilih (Minggu 16)"):
        display_agents = final_agents[["Agent", "A", "R", "F"]].copy()
        display_agents.columns = ["ID Mahasiswa", "Ketergantungan (K)", "Pemahaman (R)", "FOMO (F)"]
        st.dataframe(
            display_agents.style.format({
                "Ketergantungan (K)": "{:.3f}",
                "Pemahaman (R)": "{:.3f}",
                "FOMO (F)": "{:.3f}",
            }),
            use_container_width=True,
            hide_index=True,
        )


st.markdown("---")
st.markdown("""
<div style="text-align:center; padding: 0.5rem 0;">
    <span style="font-family:'JetBrains Mono',monospace; font-size:0.75rem; color:#888888; font-weight:500;">
        Agent-Based Model (Mesa) | 16 Minggu Perkuliahan |
        K(t+1) = K(t) + (T x F) - (R x D x K)
    </span>
</div>
""", unsafe_allow_html=True)
