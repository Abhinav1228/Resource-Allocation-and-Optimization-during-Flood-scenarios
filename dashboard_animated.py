"""
dashboard_animated.py
─────────────────────
Smooth-animation version of the DisasterAI Command Center.

Key difference from dashboard.py:
  All animation frames are pre-built as Plotly go.Frame objects and sent to
  the browser in one shot. Plotly's built-in animation engine then handles
  Play / Pause / scrubbing entirely client-side — no st.rerun(), no iframe
  rebuild, no page flash between frames.

Run with:
  streamlit run dashboard_animated.py
"""

import streamlit as st
import numpy as np
import os
import rasterio
import plotly.graph_objects as go
from scipy.optimize import linear_sum_assignment

from env.terrain_loader import TerrainLoader
from env.data_loader import DataLoader
from env.hazard_injection import HazardInjector
from env.hazard_propagation import HazardPropagation
from env.environment import DisasterEnvironment

# ─────────────────────────── PAGE CONFIG ───────────────────────────

st.set_page_config(
    page_title="DisasterAI Animated — Command Center",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main .block-container { padding-top: 1rem; }

    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #0f3460;
        border-radius: 12px;
        padding: 12px 16px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    div[data-testid="stMetric"] label {
        color: #a0aec0 !important;
        font-size: 0.85rem !important;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #e2e8f0 !important;
        font-weight: 700 !important;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1b2a 0%, #1b2838 100%);
    }
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #e2e8f0 !important;
    }

    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #e94560 0%, #c62a40 100%) !important;
        border: none !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        padding: 0.6rem 1rem !important;
        border-radius: 8px !important;
    }
    .stButton > button:not([kind="primary"]) {
        border: 1px solid #0f3460 !important;
        border-radius: 8px !important;
    }

    h1 {
        background: linear-gradient(90deg, #00d2ff, #3a7bd5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.2rem !important;
    }

    /* Remove Plotly's default white background bleed */
    .js-plotly-plot .plotly .modebar {
        background: rgba(13,27,42,0.8) !important;
        border-radius: 6px;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────── CACHED LOADERS ───────────────────────────

@st.cache_resource
def load_terrain_and_roads():
    base_dir = os.path.dirname(__file__)
    tif_files = [
        os.path.join(base_dir, "env", "datasets", "n18_e072_1arc_v3.tif"),
        os.path.join(base_dir, "env", "datasets", "n19_e072_1arc_v3.tif")
    ]
    terrain = TerrainLoader(tif_files)
    terrain.load_and_crop_dem()
    terrain.download_road_network()
    rem = terrain.compute_rem(river_name="Ulhas River")
    return terrain, rem

@st.cache_resource
def load_flood_sources(_terrain, _rem):
    loader = DataLoader()
    flood_events = loader.load_flood_events()
    injector = HazardInjector(_terrain.transform, _rem.shape)
    source_pixels = injector.inject_from_events(flood_events)
    if not source_pixels:
        print("Using coastal low-elevation flood injection points...")
        source_pixels = HazardInjector.find_coastal_sources(_rem, num_sources=4)
    return source_pixels

# ─────────────────────────── HELPERS ───────────────────────────

def rc_to_latlon(transform, r, c):
    x, y = rasterio.transform.xy(transform, int(r), int(c))
    return float(y), float(x)

def heuristic_dispatch(env):
    idle_units = [u for u in env.units if u.status == "idle"]
    active_incs = env.incident_manager.get_active_incidents()
    if not idle_units or not active_incs:
        return []
    cost_matrix = np.zeros((len(idle_units), len(active_incs)))
    for i, u in enumerate(idle_units):
        for j, inc in enumerate(active_incs):
            dist = abs(u.r - inc.r) + abs(u.c - inc.c)
            risk_penalty = (1.0 - inc.risk_level) * 1000
            cost_matrix[i, j] = dist + risk_penalty
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return [(idle_units[i].id, active_incs[j].id) for i, j in zip(row_ind, col_ind)]

# ─────────────────────────── PLOTLY ANIMATION BUILDER ───────────────────────────

def build_plotly_animation(terrain, frames_data, speed_ms=800):
    """
    Converts all simulation frames into a Plotly figure with go.Frame objects.

    The browser receives all frames at once and handles Play/Pause/scrubbing
    entirely in JavaScript — zero Python round-trips during playback.

    Satellite tiles: Esri World Imagery via custom Mapbox raster layer.
    No Mapbox token required (uses "white-bg" base style + overlay tiles).
    """
    center_lat = (terrain.min_lat + terrain.max_lat) / 2
    center_lon = (terrain.min_lon + terrain.max_lon) / 2
    transition_ms = max(speed_ms // 3, 80)

    # Pre-collect all road edges once (they share the same graph)
    all_edges = [
        (u, v)
        for u, v in terrain.road_graph.edges()
        if u in terrain.node_to_rc and v in terrain.node_to_rc
    ] if terrain.road_graph else []

    plotly_frames = []
    slider_steps  = []

    for i, frame in enumerate(frames_data):
        flood_depth = frame["flood_depth"]
        info        = frame["info"]

        # ── Flood density layer ──────────────────────────────────────────
        rows, cols = np.where(flood_depth > 0.05)
        f_lats = [rc_to_latlon(terrain.transform, r, c)[0] for r, c in zip(rows, cols)]
        f_lons = [rc_to_latlon(terrain.transform, r, c)[1] for r, c in zip(rows, cols)]
        f_z    = [min(float(flood_depth[r, c]), 15.0)      for r, c in zip(rows, cols)]

        # Fallback: Densitymapbox needs at least one point
        if not f_lats:
            f_lats, f_lons, f_z = [center_lat], [center_lon], [0.0]

        # ── Road traces (open / blocked) ─────────────────────────────────
        open_lats,    open_lons    = [], []
        blocked_lats, blocked_lons = [], []
        for u, v in all_edges:
            r1, c1 = terrain.node_to_rc[u]
            r2, c2 = terrain.node_to_rc[v]
            lat1, lon1 = rc_to_latlon(terrain.transform, r1, c1)
            lat2, lon2 = rc_to_latlon(terrain.transform, r2, c2)
            if flood_depth[r1, c1] > 0.15 or flood_depth[r2, c2] > 0.15:
                blocked_lats += [lat1, lat2, None]
                blocked_lons += [lon1, lon2, None]
            else:
                open_lats += [lat1, lat2, None]
                open_lons += [lon1, lon2, None]

        # ── Victim markers ───────────────────────────────────────────────
        vic_lats, vic_lons, vic_colors, vic_text = [], [], [], []
        for inc_r, inc_c, risk, resolved, inc_id in frame["incidents"]:
            lat, lon = rc_to_latlon(terrain.transform, inc_r, inc_c)
            vic_lats.append(lat)
            vic_lons.append(lon)
            if resolved:
                vic_colors.append("#00e676")
                vic_text.append(f"<b>Victim #{inc_id}</b><br>RESCUED ✓")
            elif risk > 0.5:
                vic_colors.append("#ff1744")
                vic_text.append(f"<b>Victim #{inc_id}</b><br>CRITICAL — {risk:.0%}")
            else:
                vic_colors.append("#ff9800")
                vic_text.append(f"<b>Victim #{inc_id}</b><br>Active — {risk:.0%}")

        # ── Rescue unit markers + active route paths ─────────────────────
        unit_lats, unit_lons, unit_text = [], [], []
        path_lats, path_lons = [], []
        for u_r, u_c, u_status, u_id, u_path, u_target in frame["units"]:
            lat, lon = rc_to_latlon(terrain.transform, u_r, u_c)
            unit_lats.append(lat)
            unit_lons.append(lon)
            status_icon = "🟢 IDLE" if u_status == "idle" else ("🔵 EN-ROUTE" if u_status == "en-route" else "🟠 BUSY")
            target_str  = f" → Victim #{u_target}" if u_target is not None else ""
            unit_text.append(f"<b>Unit #{u_id}</b><br>{status_icon}{target_str}")
            if u_status == "en-route" and u_path:
                path_lats.append(lat)
                path_lons.append(lon)
                for pr, pc in u_path:
                    plat, plon = rc_to_latlon(terrain.transform, pr, pc)
                    path_lats.append(plat)
                    path_lons.append(plon)
                path_lats.append(None)
                path_lons.append(None)

        # ── Assemble traces for this frame ───────────────────────────────
        traces = [
            # 0: Flood heatmap
            go.Densitymapbox(
                lat=f_lats, lon=f_lons, z=f_z,
                radius=18,
                colorscale=[
                    [0.0,  "rgba(13,71,161,0)"],
                    [0.2,  "rgba(13,71,161,0.3)"],
                    [0.5,  "rgba(21,101,192,0.55)"],
                    [0.8,  "rgba(30,136,229,0.7)"],
                    [1.0,  "rgba(144,202,249,0.85)"],
                ],
                showscale=False,
                opacity=0.85,
                name="🌊 Flood",
                hoverinfo="skip",
            ),
            # 1: Open roads
            go.Scattermapbox(
                lat=open_lats, lon=open_lons,
                mode="lines",
                line=dict(width=1, color="rgba(255,255,255,0.18)"),
                name="Open Roads",
                hoverinfo="skip",
            ),
            # 2: Blocked roads
            go.Scattermapbox(
                lat=blocked_lats, lon=blocked_lons,
                mode="lines",
                line=dict(width=3, color="#ff1744"),
                name="🚧 Blocked Roads",
                hoverinfo="skip",
                opacity=0.9,
            ),
            # 3: Rescue route paths
            go.Scattermapbox(
                lat=path_lats, lon=path_lons,
                mode="lines",
                line=dict(width=3, color="#00e676"),
                name="🟢 Rescue Routes",
                hoverinfo="skip",
                opacity=0.9,
            ),
            # 4: Victims
            go.Scattermapbox(
                lat=vic_lats, lon=vic_lons,
                mode="markers",
                marker=dict(size=13, color=vic_colors, opacity=1.0),
                text=vic_text,
                hoverinfo="text",
                name="Victims",
            ),
            # 5: Rescue units
            go.Scattermapbox(
                lat=unit_lats, lon=unit_lons,
                mode="markers",
                marker=dict(
                    size=17,
                    color="#00bcd4",
                    opacity=1.0,
                    symbol="circle",
                ),
                text=unit_text,
                hoverinfo="text",
                name="🚑 Rescue Units",
            ),
        ]

        frame_title = (
            f"⏱ Step {i+1}/{len(frames_data)} — "
            f"🌊 {int(np.sum(flood_depth > 0.05)):,} cells flooded · "
            f"🆘 {info['active_incidents']} active victims · "
            f"🚑 {info['units_busy']} units moving"
        )

        plotly_frames.append(
            go.Frame(
                data=traces,
                name=str(i),
                layout=go.Layout(title_text=frame_title),
            )
        )
        slider_steps.append({
            "args": [[str(i)], {
                "frame": {"duration": speed_ms, "redraw": True},
                "mode": "immediate",
                "transition": {"duration": transition_ms},
            }],
            "label": str(i + 1),
            "method": "animate",
        })

    # ── Assemble figure ──────────────────────────────────────────────────
    fig = go.Figure(data=plotly_frames[0].data, frames=plotly_frames)

    fig.update_layout(
        # Esri satellite tiles, no Mapbox token needed
        mapbox={
            "style": "white-bg",
            "center": {"lat": center_lat, "lon": center_lon},
            "zoom": 13.5,
            "layers": [{
                "below": "traces",
                "sourcetype": "raster",
                "sourceattribution": "Esri World Imagery",
                "source": [
                    "https://server.arcgisonline.com/ArcGIS/rest/services/"
                    "World_Imagery/MapServer/tile/{z}/{y}/{x}"
                ],
            }],
        },
        title=dict(
            text=f"⏱ Step 1/{len(frames_data)} — Press ▶ Play to animate",
            font=dict(color="#e2e8f0", size=14),
            x=0.01,
        ),
        height=660,
        margin={"r": 0, "t": 44, "l": 0, "b": 0},
        paper_bgcolor="#0d1b2a",
        plot_bgcolor="#0d1b2a",
        font=dict(color="#e2e8f0"),
        legend=dict(
            bgcolor="rgba(13,27,42,0.85)",
            bordercolor="#0f3460",
            borderwidth=1,
            font=dict(color="#e2e8f0", size=12),
            x=0.01, y=0.99,
            xanchor="left", yanchor="top",
        ),
        # Play / Pause buttons
        updatemenus=[{
            "type": "buttons",
            "direction": "left",
            "showactive": False,
            "x": 0.12,
            "y": -0.04,
            "xanchor": "right",
            "yanchor": "top",
            "bgcolor": "#1a1a2e",
            "bordercolor": "#0f3460",
            "font": {"color": "#e2e8f0", "size": 13},
            "buttons": [
                {
                    "label": "▶  Play",
                    "method": "animate",
                    "args": [None, {
                        "frame": {"duration": speed_ms, "redraw": True},
                        "fromcurrent": True,
                        "transition": {"duration": transition_ms, "easing": "linear"},
                    }],
                },
                {
                    "label": "⏸  Pause",
                    "method": "animate",
                    "args": [[None], {
                        "frame": {"duration": 0, "redraw": False},
                        "mode": "immediate",
                        "transition": {"duration": 0},
                    }],
                },
            ],
        }],
        # Frame scrubber slider
        sliders=[{
            "steps": slider_steps,
            "transition": {"duration": transition_ms},
            "x": 0.13,
            "len": 0.87,
            "y": -0.04,
            "currentvalue": {
                "prefix": "🕐 Frame: ",
                "visible": True,
                "xanchor": "right",
                "font": {"color": "#e2e8f0", "size": 14},
            },
            "font": {"color": "#e2e8f0"},
            "bgcolor": "#1a1a2e",
            "bordercolor": "#0f3460",
            "tickcolor": "#0f3460",
        }],
    )

    return fig

# ─────────────────────────── MAIN ───────────────────────────

def main():
    st.title("🛰️ DisasterAI — Smooth Animation Mode")

    terrain, rem = load_terrain_and_roads()
    source_pixels = load_flood_sources(terrain, rem)
    sources_rc = [(r, c) for r, c, _ in source_pixels]

    # ── Sidebar ──────────────────────────────────────────────────────────
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/flood.png", width=60)
        st.markdown("### Simulation Controls")
        st.markdown("Configure and launch the disaster scenario below.")
        st.markdown("---")

        num_steps   = st.slider("⏱ Duration (steps)",  5,  60, 30)
        num_victims = st.slider("👥 Number of Victims", 5,  20, 12)
        num_units   = st.slider("🚑 Rescue Units",      3,  10,  5)

        st.markdown("---")

        anim_speed = st.select_slider(
            "🎬 Animation Speed",
            options=["Slow", "Normal", "Fast"],
            value="Normal",
            help="Controls the frame duration in the client-side Plotly animation."
        )
        speed_map = {"Slow": 1400, "Normal": 750, "Fast": 300}
        speed_ms  = speed_map[anim_speed]

        st.markdown("---")

        run_btn   = st.button("▶ LAUNCH SIMULATION", type="primary", use_container_width=True)
        reset_btn = st.button("🔄 Reset",                            use_container_width=True)

        st.markdown("---")
        st.markdown("### 🗺️ Map Legend")
        st.markdown("""
        - 🔵 **Blue heatmap** — Flood water extent
        - 🔴 **Red lines** — Blocked / flooded roads
        - ⚪ **White lines** — Open roads
        - 🟢 **Green lines** — Active rescue routes
        - 🟠 **Orange dot** — Active victim
        - 🔴 **Red dot** — Critical victim (risk > 50 %)
        - 🟢 **Green dot** — Rescued victim
        - 🔵 **Cyan dot** — Rescue unit
        """)
        st.markdown("---")
        st.info(
            "**Smooth Animation**\n\n"
            "All frames are pre-built and sent to the browser in one go. "
            "Press ▶ Play in the map's controls — the animation runs entirely "
            "in JavaScript with no page reloads.",
            icon="✨"
        )

    # ── Reset ─────────────────────────────────────────────────────────────
    if reset_btn:
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

    # ── Run Simulation ────────────────────────────────────────────────────
    if run_btn:
        flood_depth = np.zeros_like(rem)
        for r, c, lvl in source_pixels:
            flood_depth[r, c] += lvl

        env = DisasterEnvironment(
            rem, terrain.road_graph, terrain.node_to_rc, flood_depth,
            num_units=num_units, num_incidents=num_victims
        )
        propagator = HazardPropagation(rem)
        frames = []

        progress = st.progress(0, text="🌊 Simulating & pre-building animation frames…")

        for step in range(num_steps):
            env.flood_depth = propagator.propagate(
                env.flood_depth, sources_rc, continuous_inflow_volume=20.0
            )
            actions = heuristic_dispatch(env)
            state, reward, done, info = env.step(actions=actions)

            frames.append({
                "flood_depth": env.flood_depth.copy(),
                "units": [
                    (u.r, u.c, u.status, u.id,
                     [terrain.node_to_rc[n] for n in u.path_nodes] if u.path_nodes else [],
                     u.target_incident.id if u.target_incident else None)
                    for u in env.units
                ],
                "incidents": [
                    (inc.r, inc.c, inc.risk_level, inc.is_resolved, inc.id)
                    for inc in env.incident_manager.incidents
                ],
                "info": info.copy(),
            })

            flooded_cells = int(np.sum(env.flood_depth > 0.05))
            progress.progress(
                (step + 1) / num_steps,
                text=(
                    f"Step {step+1}/{num_steps} — "
                    f"🌊 {flooded_cells} cells flooded · "
                    f"🆘 {info['active_incidents']} active incidents"
                )
            )
            if done:
                break

        progress.empty()
        st.session_state.frames      = frames
        st.session_state.speed_ms    = speed_ms
        st.session_state.num_victims = num_victims
        st.rerun()

    # ── Display Results ───────────────────────────────────────────────────
    if st.session_state.get("frames"):
        frames   = st.session_state.frames
        speed_ms = st.session_state.get("speed_ms", 750)

        # Summary metrics (across all frames)
        final_info = frames[-1]["info"]
        first_info = frames[0]["info"]
        max_flooded = int(max(np.sum(f["flood_depth"] > 0.05) for f in frames))
        total_vic   = len(frames[0]["incidents"])
        rescued     = sum(1 for inc in frames[-1]["incidents"] if inc[3])

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("🕐 Total Steps",    f"{len(frames)}")
        c2.metric("🌊 Peak Flood",     f"{max_flooded:,} cells")
        c3.metric("👥 Total Victims",  f"{total_vic}")
        c4.metric("🚑 Rescued",        f"{rescued}/{total_vic}")
        c5.metric("🏆 Score",          f"{round(final_info['total_reward'], 1)}")

        # Build & display the Plotly animated figure
        with st.spinner("🗺️ Building animated map…"):
            fig = build_plotly_animation(terrain, frames, speed_ms=speed_ms)

        st.plotly_chart(fig, use_container_width=True)

        # Result banner
        if rescued == total_vic:
            st.success(
                f"🎉 **ALL {total_vic} VICTIMS RESCUED!** "
                f"Total rescue score: {round(final_info['total_reward'], 1)}"
            )
        else:
            st.info(
                f"📍 **{rescued}/{total_vic}** victims rescued by end of simulation · "
                f"Use the ▶ Play button in the map above to watch the disaster unfold."
            )

    else:
        # ── Landing state ─────────────────────────────────────────────────
        st.markdown("---")
        col_a, col_b = st.columns([2, 3])
        with col_a:
            st.markdown("""
            ### ✨ Smooth Animation Mode

            This version pre-builds **all animation frames** as a single Plotly
            figure and sends them to your browser at once.

            **Controls embedded in the map:**
            - **▶ Play** / **⏸ Pause** buttons (bottom-left of map)
            - **Scrubber slider** to jump to any frame
            - **Hover** over any victim or unit for details
            - **Scroll** to zoom, **drag** to pan

            No page refresh, no flicker — pure client-side JavaScript animation.

            ---

            ### How It Works
            1. **🌊 Flood Simulation** — Dijkstra priority-queue water physics
            2. **🚧 Road Blocking** — Roads turn red when flooded
            3. **🧠 AI Dispatch** — Hungarian algorithm assigns units to victims
            4. **🔄 Dynamic Rerouting** — Units re-path around new flood edges

            > 👈 **Click "LAUNCH SIMULATION"** in the sidebar to begin
            """)
        with col_b:
            center_lat = (terrain.min_lat + terrain.max_lat) / 2
            center_lon = (terrain.min_lon + terrain.max_lon) / 2
            preview = go.Figure(go.Scattermapbox())
            preview.update_layout(
                mapbox={
                    "style": "white-bg",
                    "center": {"lat": center_lat, "lon": center_lon},
                    "zoom": 13,
                    "layers": [{
                        "below": "traces",
                        "sourcetype": "raster",
                        "sourceattribution": "Esri",
                        "source": [
                            "https://server.arcgisonline.com/ArcGIS/rest/services/"
                            "World_Imagery/MapServer/tile/{z}/{y}/{x}"
                        ],
                    }],
                },
                height=450,
                margin={"r": 0, "t": 0, "l": 0, "b": 0},
                paper_bgcolor="#0d1b2a",
            )
            st.plotly_chart(preview, use_container_width=True)


if __name__ == "__main__":
    main()
