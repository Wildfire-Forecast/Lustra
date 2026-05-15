"""Serve a live Leaflet fire-tracker map backed by fire_state.json."""

from __future__ import annotations

import os
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REFRESH_MS = 1000

# __REFRESH_MS__ is replaced below — avoids doubling every {{ }} in CSS/JS
_INDEX_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Lustra — Fire Tracker</title>
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
    .dot {
      width: 9px; height: 9px; border-radius: 50%;
      background: #555;
      transition: background 0.4s;
      animation: pulse 1.5s ease-in-out infinite;
    }
    @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.2; } }
    .badge {
      font-size: 0.7rem; color: #999;
      background: #222; border: 1px solid #333;
      padding: 2px 9px; border-radius: 99px;
    }
    #status { font-size: 0.72rem; color: #888; }
    #last-update { margin-left: auto; font-size: 0.72rem; color: #555; }
    #map { flex: 1; }
  </style>
</head>
<body>
  <header>
    <div class="dot" id="dot"></div>
    <h1>Lustro — Fire Tracker</h1>
    <span class="badge" id="fire-count">— fires</span>
    <span id="status">connecting…</span>
    <span id="last-update">—</span>
  </header>
  <div id="map"></div>
  <script>
    const REFRESH_MS = __REFRESH_MS__;

    const map = L.map('map').setView([39.0, 35.0], 13);
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

    // Dry pane sits below fire pane so red always renders on top
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

    const dot        = document.getElementById('dot');
    const fireCount  = document.getElementById('fire-count');
    const statusEl   = document.getElementById('status');
    const lastUpdate = document.getElementById('last-update');

    async function update() {
      try {
        const res = await fetch('/state?_t=' + Date.now());
        if (!res.ok) throw new Error('not ready');
        const state = await res.json();

        fireLayer.clearLayers();
        dryLayer.clearLayers();

        const rawFire = state.fire_geojson ?? state.geojson ?? { type: 'FeatureCollection', features: [] };
        const rawDry  = state.dry_geojson  ?? { type: 'FeatureCollection', features: [] };

        // Dissolve touching/overlapping zones of the same kind into a single polygon.
        const fireGeojson = mergeTouching(rawFire);
        const dryGeojson  = mergeTouching(rawDry);

        const fireFeatures = fireGeojson.features ?? [];
        const dryFeatures  = dryGeojson.features  ?? [];

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
        fireCount.textContent = `${fireLabel} · ${dryLabel}`;

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

        dot.style.background = '#4caf50';
        statusEl.textContent = 'live';
        lastUpdate.textContent = 'Updated ' + new Date().toLocaleTimeString();
      } catch (_) {
        dot.style.background = '#e53935';
        statusEl.textContent = 'waiting for app…';
      }
    }

    update();
    setInterval(update, REFRESH_MS);
  </script>
</body>
</html>
"""

_INDEX_HTML: bytes = _INDEX_TEMPLATE.replace("__REFRESH_MS__", str(REFRESH_MS)).encode()
_EMPTY_STATE: bytes = (
    b'{"geojson":{"type":"FeatureCollection","features":[]},'
    b'"fire_geojson":{"type":"FeatureCollection","features":[]},'
    b'"dry_geojson":{"type":"FeatureCollection","features":[]},'
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
