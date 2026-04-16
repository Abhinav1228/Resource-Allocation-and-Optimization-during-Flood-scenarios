import streamlit as st
import numpy as np
import os
import time
import rasterio
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
from scipy.optimize import linear_sum_assignment

from env.terrain_loader import TerrainLoader
from env.data_loader import DataLoader
from env.hazard_injection import HazardInjector
from env.hazard_propagation import HazardPropagation
from env.environment import DisasterEnvironment

st.set_page_config(
    page_title="DisasterAI — Command Center",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────── CUSTOM CSS ───────────────────────────

st.markdown("""
<style>
    /* Dark gradient header */
    .main .block-container { padding-top: 1rem; }
    
    /* Metric cards */
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
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1b2a 0%, #1b2838 100%);
    }
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #e2e8f0 !important;
    }
    
    /* Buttons */
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
    
    /* Slider */
    div[data-testid="stSlider"] {
        padding: 0.5rem 0;
    }
    
    /* Info/Success boxes */
    div[data-testid="stAlert"] {
        border-radius: 10px !important;
    }
    
    /* Title */
    h1 { 
        background: linear-gradient(90deg, #00d2ff, #3a7bd5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.2rem !important;
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
    # Try live API first
    loader = DataLoader()
    flood_events = loader.load_flood_events()
    injector = HazardInjector(_terrain.transform, _rem.shape)
    source_pixels = injector.inject_from_events(flood_events)
    
    # If no API sources landed in our bbox, use realistic coastal injection
    if not source_pixels:
        print("Using coastal low-elevation flood injection points...")
        source_pixels = HazardInjector.find_coastal_sources(_rem, num_sources=4)
    return source_pixels

# ─────────────────────────── HELPERS ───────────────────────────

def rc_to_latlon(transform, r, c):
    x, y = rasterio.transform.xy(transform, int(r), int(c))
    return y, x

def flood_to_heatmap_points(flood_depth, transform):
    points = []
    rows, cols = np.where(flood_depth > 0.05)
    for r, c in zip(rows, cols):
        lat, lon = rc_to_latlon(transform, r, c)
        intensity = min(float(flood_depth[r, c]), 15.0) / 15.0
        points.append([lat, lon, intensity])
    return points

def classify_roads(terrain, flood_depth, threshold=0.15):
    blocked, open_roads = [], []
    if not terrain.road_graph:
        return blocked, open_roads
    for u, v, data in terrain.road_graph.edges(data=True):
        if u in terrain.node_to_rc and v in terrain.node_to_rc:
            r1, c1 = terrain.node_to_rc[u]
            r2, c2 = terrain.node_to_rc[v]
            lat1, lon1 = rc_to_latlon(terrain.transform, r1, c1)
            lat2, lon2 = rc_to_latlon(terrain.transform, r2, c2)
            if flood_depth[r1, c1] > threshold or flood_depth[r2, c2] > threshold:
                blocked.append([[lat1, lon1], [lat2, lon2]])
            else:
                open_roads.append([[lat1, lon1], [lat2, lon2]])
    return blocked, open_roads

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

def build_map(terrain, frame_data):
    center_lat = (terrain.min_lat + terrain.max_lat) / 2
    center_lon = (terrain.min_lon + terrain.max_lon) / 2

    m = folium.Map(location=[center_lat, center_lon], zoom_start=15, tiles=None)
    m.fit_bounds([[terrain.min_lat, terrain.min_lon], [terrain.max_lat, terrain.max_lon]])

    # Map layers
    folium.TileLayer('OpenStreetMap', name='Street Map').add_to(m)
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri', name='🛰️ Satellite', overlay=False
    ).add_to(m)

    flood_depth = frame_data['flood_depth']

    # Roads: open = subtle white, blocked = bright red
    blocked, open_roads = classify_roads(terrain, flood_depth)
    for seg in open_roads:
        folium.PolyLine(seg, color='#ffffff', weight=1, opacity=0.15).add_to(m)
    for seg in blocked:
        folium.PolyLine(seg, color='#ff1744', weight=3.5, opacity=0.9).add_to(m)

    # Flood heatmap
    heat_points = flood_to_heatmap_points(flood_depth, terrain.transform)
    if heat_points:
        HeatMap(
            heat_points, radius=20, blur=14, max_zoom=18,
            gradient={0.15: '#1a237e', 0.3: '#0d47a1', 0.5: '#1565c0', 0.7: '#1e88e5', 0.85: '#42a5f5', 1.0: '#90caf9'},
            name='🌊 Flood Extent'
        ).add_to(m)

    # Incident markers
    for inc_r, inc_c, risk, resolved, inc_id in frame_data['incidents']:
        lat, lon = rc_to_latlon(terrain.transform, inc_r, inc_c)
        if resolved:
            color, icon_str = 'green', 'check'
            label = 'RESCUED ✓'
        elif risk > 0.5:
            color, icon_str = 'red', 'exclamation-triangle'
            label = f'CRITICAL — Risk {risk:.0%}'
        else:
            color, icon_str = 'orange', 'exclamation-circle'
            label = f'Active — Risk {risk:.0%}'
        folium.Marker(
            [lat, lon],
            icon=folium.Icon(color=color, icon=icon_str, prefix='fa'),
            popup=folium.Popup(f"<b>Victim #{inc_id}</b><br>{label}", max_width=200),
            tooltip=f"Victim #{inc_id}: {label}"
        ).add_to(m)

    # Rescue units + paths
    for u_r, u_c, u_status, u_id, u_path, u_target in frame_data['units']:
        lat, lon = rc_to_latlon(terrain.transform, u_r, u_c)
        status_label = '🟢 IDLE' if u_status == 'idle' else '🔵 EN-ROUTE' if u_status == 'en-route' else '🟠 BUSY'
        target_str = f" → Victim #{u_target}" if u_target is not None else ""
        folium.Marker(
            [lat, lon],
            icon=folium.Icon(color='darkgreen', icon='ambulance', prefix='fa'),
            popup=folium.Popup(f"<b>Rescue Unit #{u_id}</b><br>{status_label}{target_str}", max_width=200),
            tooltip=f"Unit #{u_id}: {status_label}"
        ).add_to(m)
        if u_status == "en-route" and u_path:
            path_coords = [[lat, lon]]
            for pr, pc in u_path:
                plat, plon = rc_to_latlon(terrain.transform, pr, pc)
                path_coords.append([plat, plon])
            folium.PolyLine(path_coords, color='#00e676', weight=4, opacity=0.9, dash_array='8').add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    return m

# ─────────────────────────── MAIN ───────────────────────────

def main():
    st.title("🛰️ DisasterAI Command Center")

    terrain, rem = load_terrain_and_roads()
    source_pixels = load_flood_sources(terrain, rem)
    sources_rc = [(r, c) for r, c, _ in source_pixels]

    # ── Sidebar ──
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/flood.png", width=60)
        st.markdown("### Simulation Controls")
        st.markdown("Configure and run the disaster scenario below.")
        st.markdown("---")
        
        num_steps = st.slider("⏱ Duration (steps)", 5, 60, 30)
        num_victims = st.slider("👥 Number of Victims", 5, 20, 12)
        num_units = st.slider("🚑 Rescue Units", 3, 10, 5)
        
        st.markdown("---")
        
        playback_mode = st.radio(
            "🎬 Playback Mode",
            ["📊 Timeline (Manual Slider)", "🎬 Animation (Auto-Play)"],
            help="Timeline lets you scrub freely. Animation auto-plays like a video."
        )
        
        # Default defined here so anim_speed is always in scope
        anim_speed = "Normal"
        if playback_mode == "🎬 Animation (Auto-Play)":
            anim_speed = st.select_slider(
                "Animation Speed",
                options=["Slow", "Normal", "Fast"],
                value="Normal"
            )
        
        st.markdown("---")
        
        run_btn = st.button("▶ LAUNCH SIMULATION", type="primary", use_container_width=True)
        
        # Animation controls (only visible after simulation)
        if st.session_state.get('frames') and playback_mode == "🎬 Animation (Auto-Play)":
            col_play, col_stop = st.columns(2)
            with col_play:
                play_btn = st.button("⏵ Play", use_container_width=True)
            with col_stop:
                stop_btn = st.button("⏹ Stop", use_container_width=True)
        else:
            play_btn = False
            stop_btn = False
        
        reset_btn = st.button("🔄 Reset", use_container_width=True)
        
        st.markdown("---")
        st.markdown("### 🗺️ Map Legend")
        st.markdown("""
        - 🔵 **Blue gradient** — Flood water extent  
        - 🔴 **Red lines** — Blocked/flooded roads  
        - ⚪ **White lines** — Open roads  
        - 🟢 **Green dashes** — Active rescue routes  
        - 🟠 **Orange marker** — Active victim  
        - 🔴 **Red marker** — Critical victim  
        - 🟢 **Green marker** — Rescued victim  
        """)

    # ── Reset ──
    if reset_btn:
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

    # ── Stop Animation ──
    if stop_btn:
        st.session_state.animating = False

    # ── Play Animation ──
    if play_btn:
        st.session_state.animating = True
        st.session_state.anim_frame = 0
        st.rerun()

    # ── Run Simulation ──
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
        
        progress = st.progress(0, text="🌊 Simulating disaster scenario...")

        for step in range(num_steps):
            env.flood_depth = propagator.propagate(env.flood_depth, sources_rc, continuous_inflow_volume=20.0)
            actions = heuristic_dispatch(env)
            state, reward, done, info = env.step(actions=actions)

            frames.append({
                'flood_depth': env.flood_depth.copy(),
                'units': [
                    (u.r, u.c, u.status, u.id,
                     [terrain.node_to_rc[n] for n in u.path_nodes] if u.path_nodes else [],
                     u.target_incident.id if u.target_incident else None)
                    for u in env.units
                ],
                'incidents': [
                    (inc.r, inc.c, inc.risk_level, inc.is_resolved, inc.id)
                    for inc in env.incident_manager.incidents
                ],
                'info': info.copy()
            })
            
            flooded_cells = int(np.sum(env.flood_depth > 0.05))
            progress.progress(
                (step + 1) / num_steps,
                text=f"Step {step+1}/{num_steps} — 🌊 {flooded_cells} cells flooded · 🆘 {info['active_incidents']} active incidents"
            )
            if done:
                break

        progress.empty()
        st.session_state.frames = frames
        st.session_state.animating = False
        st.session_state.anim_frame = 0
        st.rerun()

    # ── Display Results ──
    if st.session_state.get('frames'):
        frames = st.session_state.frames
        is_animating = st.session_state.get('animating', False)

        # Determine which frame to show
        if playback_mode == "🎬 Animation (Auto-Play)" and is_animating:
            frame_idx = st.session_state.get('anim_frame', 0)
            st.markdown(f"**🎬 Playing Animation — Frame {frame_idx + 1}/{len(frames)}**")
        elif playback_mode == "📊 Timeline (Manual Slider)":
            frame_idx = st.slider(
                "🕐 Disaster Timeline — Drag to Watch the Flood Spread",
                min_value=0, max_value=len(frames) - 1,
                value=0,
                help="Slide from left → right to watch the flood progressively spread and rescue units respond"
            )
        else:
            # Animation mode but not playing — show slider too
            frame_idx = st.slider(
                "🕐 Disaster Timeline",
                min_value=0, max_value=len(frames) - 1,
                value=st.session_state.get('anim_frame', 0)
            )

        frame = frames[frame_idx]
        info = frame['info']
        blocked_roads, _ = classify_roads(terrain, frame['flood_depth'])

        # Metrics row
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("⏱ Time Step", f"{info['time_step']}")
        c2.metric("🌊 Flooded Cells", f"{int(np.sum(frame['flood_depth'] > 0.05)):,}")
        c3.metric("🚧 Roads Blocked", f"{len(blocked_roads)}")
        c4.metric("🆘 Active Victims", f"{info['active_incidents']}")
        c5.metric("🚑 Units Moving", f"{info['units_busy']}")

        # Map — always use the full folium satellite view
        m = build_map(terrain, frame)
        st_folium(m, width=1200, height=680, returned_objects=[])

        # Bottom info bar
        resolved = sum(1 for inc in frame['incidents'] if inc[3])
        total = len(frame['incidents'])
        
        if info['active_incidents'] == 0:
            st.success(f"🎉 **ALL {total} VICTIMS RESCUED!** Total rescue score: {round(info['total_reward'], 1)}")
        else:
            st.info(
                f"📍 **{resolved}/{total}** victims rescued · "
                f"**{len(blocked_roads)}** road segments blocked by flooding · "
                f"{'Animation playing...' if is_animating else 'Drag the timeline slider ↑ to watch the disaster unfold.'}"
            )

        # Auto-advance animation
        if is_animating and frame_idx < len(frames) - 1:
            speed_map = {"Slow": 1.5, "Normal": 0.8, "Fast": 0.3}
            delay = speed_map.get(anim_speed, 0.8)
            time.sleep(delay)
            st.session_state.anim_frame = frame_idx + 1
            st.rerun()
        elif is_animating and frame_idx >= len(frames) - 1:
            st.session_state.animating = False
            st.toast("✅ Animation complete!", icon="🎬")

    else:
        # Landing state
        st.markdown("---")
        col_a, col_b = st.columns([2, 3])
        with col_a:
            st.markdown("""
            ### How It Works
            1. **🌊 Flood Simulation** — Real-time water propagation using Dijkstra priority-queue physics over Mumbai's coastal terrain (Bandra-Kurla / Mithi River basin)
            2. **🚧 Road Blocking** — Roads dynamically turn **red** when floodwater reaches them
            3. **🧠 AI Dispatch** — Hungarian algorithm optimally assigns rescue units to victims
            4. **🔄 Dynamic Rerouting** — When a road floods mid-route, units automatically find alternate paths
            
            > 👈 **Click "LAUNCH SIMULATION" in the sidebar** to begin
            """)
        with col_b:
            center_lat = (terrain.min_lat + terrain.max_lat) / 2
            center_lon = (terrain.min_lon + terrain.max_lon) / 2
            m = folium.Map(location=[center_lat, center_lon], zoom_start=15, tiles=None)
            m.fit_bounds([[terrain.min_lat, terrain.min_lon], [terrain.max_lat, terrain.max_lon]])
            folium.TileLayer(
                tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
                attr='Esri', name='Satellite'
            ).add_to(m)
            st_folium(m, width=750, height=450, returned_objects=[])

if __name__ == "__main__":
    main()
