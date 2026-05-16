"""Serve a live Leaflet fire-tracker map backed by fire_state.json."""

from __future__ import annotations

import json
import os
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REFRESH_MS = 1000

# Edit these to change the location-jump buttons in the GUI.
# Sites picked for southwestern-Türkiye Mediterranean fire belt (2021 megafires).
LOCATIONS = [
    {"name": "DEÜ Buca", "lat": 38.3700, "lon": 27.2050},
    {"name": "Antalya",  "lat": 36.8969, "lon": 30.7133},
    {"name": "Manavgat", "lat": 36.7869, "lon": 31.4433},
    {"name": "Marmaris", "lat": 36.8550, "lon": 28.2750},
    {"name": "Datça",    "lat": 36.7257, "lon": 27.6850},
]

# Sidecar file: serve_map.py writes here, LustraApp polls it and re-anchors
# its map origin (which shifts every drone/fire/prediction lat-lon on the map).
ORIGIN_COMMAND_PATH = Path(__file__).resolve().parent / "origin_command.json"
_origin_lock = threading.Lock()

# __REFRESH_MS__ is replaced below — avoids doubling every {{ }} in CSS/JS
_INDEX_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Lustro</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <script src="https://unpkg.com/@turf/turf@7.2.0/turf.min.js"></script>
  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body {
      background: #111;
      font-family: 'Segoe UI', system-ui, sans-serif;
      display: flex; flex-direction: column; height: 100vh;
    }
    header {
      background: #1a1a1a;
      border-bottom: 2px solid #ff6b35;
      padding: 8px 20px;
      display: flex; align-items: center; gap: 14px;
      flex-shrink: 0; color: #eee;
    }
    h1 { font-size: 0.95rem; color: #ff6b35; letter-spacing: 2px; text-transform: uppercase; }
    .badge {
      font-size: 0.7rem; color: #999;
      background: #222; border: 1px solid #333;
      padding: 2px 9px; border-radius: 99px;
    }
    #last-update { margin-left: auto; font-size: 0.72rem; color: #555; }
    #map { flex: 1; position: relative; }
    #wind-widget {
      position: absolute; top: 14px; right: 14px;
      z-index: 1000;
      background: rgba(17, 17, 17, 0.88); color: #eee;
      border: 1px solid #333; border-radius: 8px;
      padding: 10px 14px;
      font-family: 'Segoe UI', system-ui, sans-serif;
      font-size: 0.72rem;
      min-width: 220px;
      box-shadow: 0 4px 18px rgba(0,0,0,0.5);
    }
    #wind-widget .title {
      color: #ff6b35; font-size: 0.65rem; letter-spacing: 1.5px;
      text-transform: uppercase; margin-bottom: 6px;
    }
    #wind-widget .row {
      display: flex; justify-content: space-between; align-items: center;
      margin: 3px 0;
    }
    #wind-widget .label { color: #888; }
    #wind-widget .val { color: #eee; font-weight: 600; font-variant-numeric: tabular-nums; }
    #wind-widget .head { color: #ffd23f; }
    #wind-widget .no-spread {
      color: #b87878; font-style: italic; font-size: 0.7rem;
      margin-top: 4px; line-height: 1.4;
    }
    #loc-widget {
      position: absolute; top: 14px; left: 60px;
      z-index: 1000;
      background: rgba(17, 17, 17, 0.88); color: #eee;
      border: 1px solid #333; border-radius: 8px;
      padding: 10px 14px;
      font-family: 'Segoe UI', system-ui, sans-serif;
      font-size: 0.72rem;
      box-shadow: 0 4px 18px rgba(0,0,0,0.5);
    }
    #loc-widget .title {
      color: #ff6b35; font-size: 0.65rem; letter-spacing: 1.5px;
      text-transform: uppercase; margin-bottom: 6px;
    }
    #loc-widget .btns { display: flex; flex-direction: column; gap: 4px; }
    #loc-widget button {
      background: #222; color: #eee; border: 1px solid #333;
      padding: 5px 10px; border-radius: 4px; cursor: pointer;
      font-size: 0.72rem; font-family: inherit; text-align: left;
      transition: background 0.15s, border-color 0.15s;
    }
    #loc-widget button:hover { background: #2a2a2a; border-color: #ff6b35; }
    #loc-widget button.active {
      background: #3a2418; border-color: #ff6b35; color: #ff6b35;
    }
  </style>
</head>
<body>
  <header>
    <h1>Lustro</h1>
    <span class="badge" id="fire-count">— fires</span>
    <span id="last-update">—</span>
  </header>
  <div id="map">
    <div id="wind-widget" style="display: none;">
      <div class="title">Weather</div>
      <div class="row"><span class="label">Wind</span><span class="val" id="wind-val">— m/s</span></div>
      <div class="row"><span class="label">From</span><span class="val" id="wind-dir-val">—</span></div>
      <div class="row"><span class="label">Temp</span><span class="val" id="temp-val">—</span></div>
      <div class="row"><span class="label">RH</span><span class="val" id="rh-val">—</span></div>
      <div class="row"><span class="label">1-hr FM</span><span class="val" id="fm-val">—</span></div>
      <div class="row"><span class="label head">Head ROS</span><span class="val head" id="ros-val">—</span></div>
      <div class="no-spread" id="no-spread-msg" style="display: none;">
        Weather does not support spread<br>(fuel moisture &gt; extinction)
      </div>
    </div>
    <div id="loc-widget">
      <div class="title">Locations</div>
      <div class="btns" id="loc-btns"></div>
    </div>
  </div>
  <script>
    const REFRESH_MS = __REFRESH_MS__;

    const map = L.map('map').setView([33.45, -112.07], 13);
    L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
      maxZoom: 19,
      referrerPolicy: 'origin',
    }).addTo(map);

    // Fill opacity scales with detector confidence but stays within a readable band
    // so low-confidence zones don't vanish and high-confidence ones don't fully hide the map.
    const confFill = (c, lo, hi) => {
      const x = Math.max(0, Math.min(1, Number.isFinite(c) ? c : 0.5));
      return lo + (hi - lo) * x;
    };
    const dryStyle = (feat) => ({
      color: '#8a6638', weight: 1,
      fillColor: '#c9a274',
      fillOpacity: confFill(feat?.properties?.confidence, 0.25, 0.55),
    });
    const fireStyle = (feat) => ({
      color: '#ff4500', weight: 2,
      fillColor: '#ff4500',
      fillOpacity: confFill(feat?.properties?.confidence, 0.25, 0.65),
    });
    // Predicted spread: graduated yellow, lighter at longer horizons, dashed outline
    const predFill = { 15: 0.32, 30: 0.22, 60: 0.14 };
    const predColor = { 15: '#ffd23f', 30: '#ffe57a', 60: '#fff1ad' };
    const predStyle = (feat) => {
      const h = Math.round(feat?.properties?.horizon_min ?? 60);
      return {
        color: '#a17c00', weight: 1.5, dashArray: '4 4',
        fillColor: predColor[h] ?? '#ffd23f',
        fillOpacity: predFill[h] ?? 0.20,
      };
    };

    // Layer ordering: predicted at the back, dry above, fire on top
    map.createPane('predPane');
    map.getPane('predPane').style.zIndex = 398;
    map.createPane('dryPane');
    map.getPane('dryPane').style.zIndex = 399;
    map.createPane('firePane');
    map.getPane('firePane').style.zIndex = 401;

    const clusterTitle = (kind, p) =>
      (p.merged_count ?? 1) > 1
        ? `${kind} cluster (${p.merged_count} tracks: #${p.track_id})`
        : `${kind} #${p.track_id ?? '?'}`;

    const dryLayer = L.geoJSON(null, {
      pane: 'dryPane',
      style: dryStyle,
      onEachFeature(feature, layer) {
        const p = feature.properties || {};
        layer.bindPopup(
          `<b>${clusterTitle('Dry zone', p)}</b><br>` +
          `Confidence: ${((p.confidence ?? 0) * 100).toFixed(1)} %<br>` +
          `Detections: ${p.hits ?? 0}`
        );
      },
    }).addTo(map);

    const fireLayer = L.geoJSON(null, {
      pane: 'firePane',
      style: fireStyle,
      onEachFeature(feature, layer) {
        const p = feature.properties || {};
        layer.bindPopup(
          `<b>${clusterTitle('Fire', p)}</b><br>` +
          `Confidence: ${((p.confidence ?? 0) * 100).toFixed(1)} %<br>` +
          `Detections: ${p.hits ?? 0}`
        );
      },
    }).addTo(map);

    const predLayer = L.geoJSON(null, {
      pane: 'predPane',
      style: predStyle,
      onEachFeature(feature, layer) {
        const p = feature.properties || {};
        layer.bindPopup(
          `<b>Predicted spread (t+${p.horizon_min ?? '?'} min)</b><br>` +
          `Track: #${p.track_id ?? '?'}<br>` +
          `Area: ${(p.area_m2 ?? 0).toFixed(0)} m&sup2;`
        );
      },
    }).addTo(map);

    // Merge polygons that overlap or touch into a single cluster feature.
    // Aggregated properties: comma-joined track IDs, summed hits, max confidence.
    function mergeTouching(geojson) {
      const features = (geojson?.features ?? []).filter(f => f && f.geometry);
      if (features.length === 0) return { type: 'FeatureCollection', features: [] };
      if (features.length === 1 || typeof turf === 'undefined') return geojson;

      let unioned;
      try {
        unioned = turf.union(turf.featureCollection(features));
      } catch (e) {
        return geojson;
      }
      if (!unioned) return geojson;

      const polys = [];
      if (unioned.geometry.type === 'Polygon') {
        polys.push(unioned.geometry.coordinates);
      } else if (unioned.geometry.type === 'MultiPolygon') {
        for (const c of unioned.geometry.coordinates) polys.push(c);
      } else {
        return geojson;
      }

      const out = [];
      for (const coords of polys) {
        const polyFeat = { type: 'Feature', geometry: { type: 'Polygon', coordinates: coords }, properties: {} };
        const members = [];
        for (const src of features) {
          try {
            const c = turf.centroid(src);
            if (turf.booleanPointInPolygon(c, polyFeat)) members.push(src);
          } catch (e) { /* skip bad geometry */ }
        }
        const pool = members.length ? members : features;
        const ids = pool.map(m => m.properties?.track_id).filter(v => v !== undefined && v !== null);
        const confs = pool.map(m => Number(m.properties?.confidence ?? 0));
        const hits = pool.reduce((s, m) => s + Number(m.properties?.hits ?? 0), 0);
        polyFeat.properties = {
          track_id: ids.join(','),
          confidence: confs.length ? Math.max(...confs) : 0,
          hits,
          merged_count: pool.length,
        };
        out.push(polyFeat);
      }
      return { type: 'FeatureCollection', features: out };
    }

    // Subtract every fire polygon from each dry polygon so the brown
    // is truly invisible (not just hidden) inside fire zones.
    function subtractFires(dryGeojson, fireGeojson) {
      const dryFeatures  = dryGeojson?.features  ?? [];
      const fireFeatures = fireGeojson?.features ?? [];
      if (!dryFeatures.length) return { type: 'FeatureCollection', features: [] };
      if (!fireFeatures.length || typeof turf === 'undefined') {
        return { type: 'FeatureCollection', features: dryFeatures };
      }
      const out = [];
      for (const dry of dryFeatures) {
        let remaining = dry;
        try {
          for (const fire of fireFeatures) {
            if (!remaining) break;
            const diff = turf.difference(turf.featureCollection([remaining, fire]));
            if (!diff) { remaining = null; break; }
            remaining = { ...diff, properties: dry.properties };
          }
        } catch (e) {
          remaining = dry;
        }
        if (remaining) out.push(remaining);
      }
      return { type: 'FeatureCollection', features: out };
    }

    const droneIcon = L.divIcon({
      className: '',
      html: `<svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" viewBox="0 0 32 32">
        <defs>
          <filter id="shadow" x="-40%" y="-40%" width="180%" height="180%">
            <feDropShadow dx="0" dy="0" stdDeviation="2" flood-color="#000" flood-opacity="0.9"/>
          </filter>
        </defs>
        <g filter="url(#shadow)">
          <line x1="16" y1="16" x2="4"  y2="4"  stroke="#fff" stroke-width="2.2"/>
          <line x1="16" y1="16" x2="28" y2="4"  stroke="#fff" stroke-width="2.2"/>
          <line x1="16" y1="16" x2="4"  y2="28" stroke="#fff" stroke-width="2.2"/>
          <line x1="16" y1="16" x2="28" y2="28" stroke="#fff" stroke-width="2.2"/>
          <circle cx="4"  cy="4"  r="4" fill="none" stroke="#ff3322" stroke-width="2.5"/>
          <circle cx="28" cy="4"  r="4" fill="none" stroke="#ff3322" stroke-width="2.5"/>
          <circle cx="4"  cy="28" r="4" fill="none" stroke="#ff3322" stroke-width="2.5"/>
          <circle cx="28" cy="28" r="4" fill="none" stroke="#ff3322" stroke-width="2.5"/>
          <rect x="12.5" y="12.5" width="7" height="7" rx="1.5" fill="#ff3322"/>
        </g>
      </svg>`,
      iconSize: [32, 32],
      iconAnchor: [16, 16],
    });
    let droneMarker = null;
    let firstFit = true;

    const fireCount  = document.getElementById('fire-count');
    const lastUpdate = document.getElementById('last-update');
    const windWidget = document.getElementById('wind-widget');
    const windVal    = document.getElementById('wind-val');
    const windDirVal = document.getElementById('wind-dir-val');
    const tempVal    = document.getElementById('temp-val');
    const rhVal      = document.getElementById('rh-val');
    const fmVal      = document.getElementById('fm-val');
    const rosVal     = document.getElementById('ros-val');
    const noSpreadMsg = document.getElementById('no-spread-msg');
    const locBtns    = document.getElementById('loc-btns');

    const LOCATIONS = __LOCATIONS_JSON__;
    let activeLoc = null;

    async function relocate(loc, btn) {
      // Tell LustraApp to re-anchor the simulator origin (this shifts the
      // drone marker and every fire/predicted polygon on the map).
      try {
        await fetch('/origin', {
          method: 'POST', headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({lat: loc.lat, lon: loc.lon, name: loc.name}),
        });
      } catch (_) { /* network errors are non-fatal; the map fly-to still runs */ }
      // map.getMaxZoom() returns 19 (set by the OSM tile layer above) — the
      // tightest zoom the basemap supports for the picked location.
      map.flyTo([loc.lat, loc.lon], map.getMaxZoom(), { duration: 1.2 });
      firstFit = false;  // user took control of the view; don't auto-fit again
      activeLoc = loc.name;
      for (const b of locBtns.children) {
        b.classList.toggle('active', b.dataset.name === loc.name);
      }
      // Trigger a refresh so the drone marker moves immediately once the
      // app picks up the new origin on its next tick.
      setTimeout(update, 1200);
    }

    for (const loc of LOCATIONS) {
      const btn = document.createElement('button');
      btn.textContent = loc.name;
      btn.dataset.name = loc.name;
      btn.addEventListener('click', () => relocate(loc, btn));
      locBtns.appendChild(btn);
    }

    function cardinalName(deg) {
      const names = ['N','NNE','NE','ENE','E','ESE','SE','SSE','S','SSW','SW','WSW','W','WNW','NW','NNW'];
      return names[Math.round(((deg % 360) + 360) % 360 / 22.5) % 16];
    }

    function updateWindWidget(w) {
      if (!w) { windWidget.style.display = 'none'; return; }
      windWidget.style.display = 'block';
      const from = w.wind_direction_10m_deg ?? 0;
      windVal.textContent = (w.wind_speed_10m_ms ?? 0).toFixed(2) + ' m/s';
      windDirVal.textContent = `${cardinalName(from)} (${Math.round(from)}°)`;
      tempVal.textContent = (w.temperature_c ?? 0).toFixed(1) + ' °C';
      rhVal.textContent = Math.round(w.relative_humidity_pct ?? 0) + ' %';
      fmVal.textContent = (w.one_hour_fuel_moisture_pct ?? 0).toFixed(1) + ' %';
      const ros = w.rate_of_spread_m_per_min ?? 0;
      if (w.spread_supported) {
        rosVal.textContent = ros.toFixed(2) + ' m/min';
        noSpreadMsg.style.display = 'none';
      } else {
        rosVal.textContent = '0 m/min';
        noSpreadMsg.style.display = 'block';
      }
    }

    async function update() {
      try {
        const res = await fetch('/state?_t=' + Date.now());
        if (!res.ok) throw new Error('not ready');
        const state = await res.json();

        fireLayer.clearLayers();
        dryLayer.clearLayers();
        predLayer.clearLayers();

        const rawFire = state.fire_geojson ?? state.geojson ?? { type: 'FeatureCollection', features: [] };
        const rawDry  = state.dry_geojson  ?? { type: 'FeatureCollection', features: [] };
        const rawPred = state.predicted_geojson ?? { type: 'FeatureCollection', features: [] };

        // Dissolve touching/overlapping zones of the same kind into a single polygon.
        const fireGeojson = mergeTouching(rawFire);
        const dryGeojson  = mergeTouching(rawDry);

        const fireFeatures = fireGeojson.features ?? [];
        const dryFeatures  = dryGeojson.features  ?? [];
        const predFeatures = rawPred.features ?? [];

        if (predFeatures.length > 0) predLayer.addData(rawPred);

        if (fireFeatures.length > 0) fireLayer.addData(fireGeojson);

        const dryClipped = subtractFires(dryGeojson, fireGeojson);
        if (dryClipped.features.length > 0) dryLayer.addData(dryClipped);

        if (firstFit && (fireFeatures.length > 0 || dryFeatures.length > 0)) {
          const group = L.featureGroup([fireLayer, dryLayer]);
          const bounds = group.getBounds();
          if (bounds.isValid()) {
            map.fitBounds(bounds, { padding: [60, 60] });
            firstFit = false;
          }
        }
        const fireLabel = fireFeatures.length + (fireFeatures.length === 1 ? ' fire' : ' fires');
        const dryLabel  = dryFeatures.length  + (dryFeatures.length  === 1 ? ' dry'  : ' dry');
        const predLabel = predFeatures.length > 0 ? ` · ${predFeatures.length} pred` : '';
        fireCount.textContent = `${fireLabel} · ${dryLabel}${predLabel}`;

        if (state.drone) {
          const ll = [state.drone.lat, state.drone.lon];
          if (!droneMarker) {
            droneMarker = L.marker(ll, { icon: droneIcon, zIndexOffset: 1000 })
              .bindTooltip('Drone', { permanent: false })
              .addTo(map);
          } else {
            droneMarker.setLatLng(ll);
          }
        }

        updateWindWidget(state.weather);

        lastUpdate.textContent = 'Updated ' + new Date().toLocaleTimeString();
      } catch (_) {
        lastUpdate.textContent = 'Waiting for app…';
      }
    }

    update();
    setInterval(update, REFRESH_MS);
  </script>
</body>
</html>
"""

_INDEX_HTML: bytes = (
    _INDEX_TEMPLATE
    .replace("__REFRESH_MS__", str(REFRESH_MS))
    .replace("__LOCATIONS_JSON__", json.dumps(LOCATIONS))
    .encode()
)
_EMPTY_STATE: bytes = (
    b'{"geojson":{"type":"FeatureCollection","features":[]},'
    b'"fire_geojson":{"type":"FeatureCollection","features":[]},'
    b'"dry_geojson":{"type":"FeatureCollection","features":[]},'
    b'"predicted_geojson":{"type":"FeatureCollection","features":[]},'
    b'"weather":null,'
    b'"drone":null}'
)


class FireMapHandler(SimpleHTTPRequestHandler):
    def do_GET(self) -> None:
        path = self.path.split("?")[0]

        if path in ("/", "/index.html"):
            self._send(200, "text/html; charset=utf-8", _INDEX_HTML)

        elif path == "/state":
            state_file = Path("fire_state.json")
            body = state_file.read_bytes() if state_file.exists() else _EMPTY_STATE
            self._send(200, "application/json", body)

        else:
            super().do_GET()

    def do_POST(self) -> None:
        path = self.path.split("?")[0]
        if path == "/origin":
            length = int(self.headers.get("Content-Length", 0))
            try:
                payload = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
                lat = float(payload["lat"])
                lon = float(payload["lon"])
                name = str(payload.get("name", ""))
            except Exception:
                self._send(400, "application/json", b'{"error":"bad request"}')
                return
            with _origin_lock:
                # Monotonic ns timestamp as the sequence: always increases
                # across serve_map.py restarts, so a stale file from an
                # earlier session can never out-rank a fresh click.
                cmd = {"seq": time.time_ns(), "lat": lat, "lon": lon, "name": name}
                tmp = ORIGIN_COMMAND_PATH.with_suffix(".json.tmp")
                tmp.write_text(json.dumps(cmd))
                tmp.replace(ORIGIN_COMMAND_PATH)
            print(f"[origin] wrote {ORIGIN_COMMAND_PATH.name} -> {name} ({lat:.4f}, {lon:.4f})", flush=True)
            self._send(200, "application/json", json.dumps(cmd).encode())
        else:
            self._send(404, "application/json", b'{"error":"not found"}')

    def _send(self, code: int, content_type: str, body: bytes) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def end_headers(self) -> None:
        self.send_header("Referrer-Policy", "origin")
        super().end_headers()

    def log_message(self, fmt: str, *args: object) -> None:
        pass  # silence per-request access logs


def main() -> None:
    root = Path(__file__).resolve().parent
    os.chdir(root)
    server = ThreadingHTTPServer(("127.0.0.1", 8000), FireMapHandler)
    print(f"Serving {root} at http://127.0.0.1:8000")
    print("Open http://127.0.0.1:8000  (live map, updates every 1 s)")
    server.serve_forever()


if __name__ == "__main__":
    main()
