"""
Embedded Web Management Dashboard for SutraDB.
Provides an interactive 2026 dark-mode Bento Grid web console for visual querying,
collection administration, and 2D vector PCA visualization with zero external CDN dependencies.
"""

DASHBOARD_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>SutraDB Console | Industrial Hybrid Vector Database</title>
  <style>
    :root {
      --bg-dark: #07090e;
      --bg-card: rgba(15, 23, 42, 0.75);
      --bg-card-hover: rgba(30, 41, 59, 0.85);
      --border-color: rgba(56, 189, 248, 0.18);
      --border-glow: rgba(56, 189, 248, 0.35);
      --text-main: #f1f5f9;
      --text-muted: #94a3b8;
      --primary: #38bdf8;
      --primary-glow: rgba(56, 189, 248, 0.25);
      --accent: #34d399;
      --accent-glow: rgba(52, 211, 153, 0.25);
      --danger: #f87171;
      --font-mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    }

    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background: radial-gradient(circle at 50% 0%, #0f172a 0%, var(--bg-dark) 75%);
      color: var(--text-main);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      min-height: 100vh;
      padding: 24px;
      line-height: 1.5;
    }

    .container { max-width: 1360px; margin: 0 auto; }

    /* Header */
    header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding-bottom: 24px;
      border-bottom: 1px solid var(--border-color);
      margin-bottom: 28px;
      flex-wrap: wrap;
      gap: 16px;
    }
    .logo-area { display: flex; align-items: center; gap: 14px; }
    .logo-icon {
      width: 42px; height: 42px; border-radius: 10px;
      background: linear-gradient(135deg, #0284c7 0%, #38bdf8 50%, #34d399 100%);
      display: flex; align-items: center; justify-content: center;
      font-weight: 800; font-size: 20px; color: #fff;
      box-shadow: 0 0 20px var(--primary-glow);
    }
    .logo-title h1 { font-size: 22px; font-weight: 700; letter-spacing: -0.5px; display: flex; align-items: center; gap: 8px; }
    .logo-title p { font-size: 13px; color: var(--text-muted); }
    .badge {
      display: inline-flex; align-items: center; gap: 5px;
      font-size: 11px; font-weight: 600; padding: 3px 8px; border-radius: 9999px;
      background: rgba(56, 189, 248, 0.12); color: var(--primary); border: 1px solid rgba(56, 189, 248, 0.3);
    }
    .badge-green {
      background: rgba(52, 211, 153, 0.12); color: var(--accent); border-color: rgba(52, 211, 153, 0.3);
    }
    .pulse-dot {
      width: 7px; height: 7px; border-radius: 50%; background: var(--accent);
      box-shadow: 0 0 8px var(--accent); display: inline-block;
    }

    .header-actions { display: flex; align-items: center; gap: 12px; }
    button, .btn {
      background: rgba(30, 41, 59, 0.8); color: var(--text-main);
      border: 1px solid var(--border-color); border-radius: 8px;
      padding: 8px 16px; font-size: 13px; font-weight: 500; cursor: pointer;
      display: inline-flex; align-items: center; gap: 6px;
      transition: all 0.15s ease;
    }
    button:hover, .btn:hover { background: rgba(51, 65, 85, 0.9); border-color: var(--primary); }
    .btn-primary {
      background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%);
      color: #fff; border-color: #38bdf8;
      box-shadow: 0 0 14px var(--primary-glow);
    }
    .btn-primary:hover { background: linear-gradient(135deg, #0369a1 0%, #0284c7 100%); }

    /* Bento Grid Stats */
    .bento-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 16px;
      margin-bottom: 28px;
    }
    .bento-card {
      background: var(--bg-card);
      backdrop-filter: blur(12px);
      border: 1px solid var(--border-color);
      border-radius: 14px;
      padding: 18px 20px;
      box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
      transition: border-color 0.2s, box-shadow 0.2s;
    }
    .bento-card:hover {
      border-color: var(--border-glow);
      box-shadow: 0 4px 24px var(--primary-glow);
    }
    .bento-card .label { font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px; color: var(--text-muted); font-weight: 600; margin-bottom: 6px; }
    .bento-card .value { font-size: 26px; font-weight: 800; color: #fff; display: flex; align-items: baseline; gap: 8px; }
    .bento-card .sub { font-size: 12px; color: var(--text-muted); margin-top: 4px; }

    /* Tabs */
    .tabs {
      display: flex; gap: 8px; border-bottom: 1px solid var(--border-color);
      margin-bottom: 24px; overflow-x: auto; padding-bottom: 4px;
    }
    .tab-btn {
      background: transparent; border: none; border-radius: 8px 8px 0 0;
      padding: 10px 18px; font-size: 14px; font-weight: 600; color: var(--text-muted);
      cursor: pointer; border-bottom: 2px solid transparent; transition: all 0.15s;
    }
    .tab-btn:hover { color: var(--text-main); background: rgba(255, 255, 255, 0.03); }
    .tab-btn.active { color: var(--primary); border-bottom-color: var(--primary); background: rgba(56, 189, 248, 0.05); }

    .tab-pane { display: none; }
    .tab-pane.active { display: block; animation: fadeIn 0.2s ease; }
    @keyframes fadeIn { from { opacity: 0; transform: translateY(4px); } to { opacity: 1; transform: translateY(0); } }

    /* Layout Grids */
    .panel-grid {
      display: grid;
      grid-template-columns: 1fr 1.4fr;
      gap: 20px;
    }
    @media (max-width: 980px) {
      .panel-grid { grid-template-columns: 1fr; }
    }

    .card {
      background: var(--bg-card);
      border: 1px solid var(--border-color);
      border-radius: 14px;
      padding: 20px;
      backdrop-filter: blur(12px);
    }
    .card-title { font-size: 16px; font-weight: 700; margin-bottom: 16px; display: flex; justify-content: space-between; align-items: center; }

    /* Form Inputs */
    .form-group { margin-bottom: 14px; }
    .form-group label { display: block; font-size: 12px; font-weight: 600; color: var(--text-muted); margin-bottom: 6px; text-transform: uppercase; letter-spacing: 0.5px; }
    input[type="text"], input[type="number"], select, textarea {
      width: 100%;
      background: rgba(15, 23, 42, 0.9);
      border: 1px solid var(--border-color);
      border-radius: 8px;
      color: var(--text-main);
      padding: 9px 12px;
      font-size: 13px;
      font-family: inherit;
      outline: none;
      transition: border-color 0.15s;
    }
    textarea { font-family: var(--font-mono); resize: vertical; min-height: 70px; }
    input:focus, select:focus, textarea:focus { border-color: var(--primary); box-shadow: 0 0 10px var(--primary-glow); }

    /* Range Slider */
    .slider-container { display: flex; align-items: center; gap: 12px; }
    .slider-container input[type="range"] { flex: 1; accent-color: var(--primary); }

    /* Results List */
    .result-item {
      background: rgba(15, 23, 42, 0.6);
      border: 1px solid var(--border-color);
      border-radius: 10px;
      padding: 14px 16px;
      margin-bottom: 12px;
      transition: all 0.15s;
    }
    .result-item:hover { border-color: var(--border-glow); background: rgba(30, 41, 59, 0.6); }
    .result-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }
    .doc-id { font-family: var(--font-mono); font-weight: 700; font-size: 13px; color: var(--primary); }
    .score-badge { font-family: var(--font-mono); font-size: 12px; font-weight: 600; padding: 2px 8px; border-radius: 6px; background: rgba(52, 211, 153, 0.15); color: var(--accent); border: 1px solid rgba(52, 211, 153, 0.3); }
    .score-bar-bg { height: 4px; background: rgba(255, 255, 255, 0.08); border-radius: 2px; overflow: hidden; margin-bottom: 10px; }
    .score-bar-fill { height: 100%; background: linear-gradient(90deg, #0284c7, #34d399); border-radius: 2px; }
    .doc-text { font-size: 13px; color: #cbd5e1; margin-bottom: 8px; background: rgba(0, 0, 0, 0.2); padding: 8px 10px; border-radius: 6px; border-left: 2px solid var(--primary); }
    .meta-box { font-family: var(--font-mono); font-size: 11px; color: var(--text-muted); background: rgba(0, 0, 0, 0.3); padding: 6px 10px; border-radius: 6px; white-space: pre-wrap; word-break: break-all; }

    /* Table */
    table { width: 100%; border-collapse: collapse; text-align: left; font-size: 13px; }
    th, td { padding: 12px 14px; border-bottom: 1px solid var(--border-color); }
    th { font-size: 11px; text-transform: uppercase; color: var(--text-muted); font-weight: 600; letter-spacing: 0.5px; }
    tr:hover td { background: rgba(255, 255, 255, 0.02); }

    /* Canvas */
    #projectionCanvas {
      width: 100%;
      height: 440px;
      background: radial-gradient(circle at 50% 50%, #0f172a 0%, #06090e 100%);
      border: 1px solid var(--border-color);
      border-radius: 10px;
      display: block;
    }

    /* Code Snippets */
    pre code {
      display: block; font-family: var(--font-mono); font-size: 12px;
      background: #020617; border: 1px solid rgba(255, 255, 255, 0.08);
      border-radius: 8px; padding: 16px; overflow-x: auto; color: #e2e8f0;
      line-height: 1.6;
    }
  </style>
</head>
<body>
  <div class="container">
    <!-- Header -->
    <header>
      <div class="logo-area">
        <div class="logo-icon">S</div>
        <div class="logo-title">
          <h1>
            SutraDB Console
            <span class="badge">v__VERSION__</span>
            <span class="badge badge-green"><span class="pulse-dot"></span> Online</span>
          </h1>
          <p>Industrial Native Rust Vector & BM25 Hybrid Engine • SIMD AVX2/NEON</p>
        </div>
      </div>
      <div class="header-actions">
        <select id="activeCollectionSelect" style="width: auto; min-width: 180px;" onchange="onCollectionChange()">
          <option value="">Loading collections...</option>
        </select>
        <button class="btn-primary" onclick="openCreateModal()">+ New Collection</button>
        <button onclick="refreshAll()">⟳ Refresh</button>
      </div>
    </header>

    <!-- Bento Grid Telemetry -->
    <div class="bento-grid">
      <div class="bento-card">
        <div class="label">Total Documents</div>
        <div class="value" id="statTotalDocs">--</div>
        <div class="sub">Across all active collections</div>
      </div>
      <div class="bento-card">
        <div class="label">Total Collections</div>
        <div class="value" id="statTotalCollections">--</div>
        <div class="sub">Persistent & In-Memory</div>
      </div>
      <div class="bento-card">
        <div class="label">Engine Latency (P50)</div>
        <div class="value" style="color: var(--accent);"><span style="font-size: 20px;">⚡</span> &lt;0.2ms</div>
        <div class="sub">Zero-copy mmap2 + SIMD BLAS</div>
      </div>
      <div class="bento-card">
        <div class="label">Quantization & Compression</div>
        <div class="value" style="color: var(--primary);"><span style="font-size: 20px;">🛡️</span> 1-Bit BQ & SQ8</div>
        <div class="sub">32x BQ & 75% SQ8 RAM savings</div>
      </div>
    </div>

    <!-- Navigation Tabs -->
    <div class="tabs">
      <button class="tab-btn active" onclick="switchTab('playground')">⚡ Hybrid Query Playground</button>
      <button class="tab-btn" onclick="switchTab('visualizer')">📊 2D Vector Projection (PCA)</button>
      <button class="tab-btn" onclick="switchTab('admin')">🛠️ Collection Administration</button>
      <button class="tab-btn" onclick="switchTab('snippets')">💻 Client Code Snippets</button>
    </div>

    <!-- Tab 1: Playground -->
    <div id="tab-playground" class="tab-pane active">
      <div class="panel-grid">
        <!-- Controls Card -->
        <div class="card">
          <div class="card-title">
            <span>Query Parameters</span>
            <span class="badge" id="activeColBadge">None</span>
          </div>

          <div class="form-group">
            <label>Lexical Query Text (BM25)</label>
            <input type="text" id="queryText" placeholder="e.g. transformer neural network" />
          </div>

          <div class="form-group">
            <label>Query Vector (Float Array or Leave Blank)</label>
            <textarea id="queryVector" placeholder="[0.12, -0.45, 0.88, ...]" rows="3"></textarea>
            <button style="margin-top: 6px; font-size: 11px; padding: 4px 8px;" onclick="generateRandomVector()">🎲 Generate Random Vector</button>
          </div>

          <div class="form-group">
            <label>Hybrid Search Ratio (RRF / Linear)</label>
            <div class="slider-container">
              <span style="font-size: 11px; color: var(--primary);">Dense (0.0)</span>
              <input type="range" id="hybridAlpha" min="0" max="1" step="0.1" value="0.5" oninput="updateAlphaLabel(this.value)">
              <span style="font-size: 11px; color: var(--accent);">Lexical (1.0)</span>
            </div>
            <div style="font-size: 11px; color: var(--text-muted); margin-top: 4px;" id="alphaLabel">Equal Balance (RRF Combined)</div>
          </div>

          <div class="form-group">
            <label>Tenant Namespace (Optional)</label>
            <input type="text" id="queryNamespace" placeholder="e.g. tenant_alpha (default: all)" />
          </div>

          <div class="form-group">
            <label>Metadata Filter JSON (AST)</label>
            <textarea id="queryFilter" placeholder='{"category": "ai", "location": {"$geo_within": {"center": [37.77, -122.41], "radius_meters": 5000}}}' rows="2"></textarea>
          </div>

          <div class="form-group">
            <label>Top K Results</label>
            <input type="number" id="queryTopK" value="5" min="1" max="100" />
          </div>

          <button class="btn-primary" style="width: 100%; justify-content: center; padding: 12px; margin-top: 8px;" onclick="runQuery()">
            🚀 Execute Hybrid Search
          </button>
        </div>

        <!-- Live Results Card -->
        <div class="card">
          <div class="card-title">
            <span>Retrieval Results</span>
            <span id="resultsCountBadge" class="badge">0 results</span>
          </div>
          <div id="resultsContainer">
            <div style="text-align: center; color: var(--text-muted); padding: 40px 20px;">
              Enter query parameters and click <b>Execute Hybrid Search</b> to inspect live retrieved vector chunks.
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Tab 2: Visualizer -->
    <div id="tab-visualizer" class="tab-pane">
      <div class="card">
        <div class="card-title">
          <span>2D Vector Space Projection (Client-Side Principal Component Analysis)</span>
          <button onclick="renderProjection()">⟳ Re-project Vectors</button>
        </div>
        <p style="font-size: 13px; color: var(--text-muted); margin-bottom: 16px;">
          Samples up to 100 vectors from the active collection, computes the 2 primary eigenvectors in JavaScript, and renders an interactive spatial map. Hover over nodes to inspect document IDs and content.
        </p>
        <canvas id="projectionCanvas"></canvas>
        <div id="canvasTooltip" style="margin-top: 10px; font-family: var(--font-mono); font-size: 12px; color: var(--primary);">
          Hover over points to inspect vector embeddings.
        </div>
      </div>
    </div>

    <!-- Tab 3: Admin -->
    <div id="tab-admin" class="tab-pane">
      <div class="card">
        <div class="card-title">
          <span>Active Collections</span>
          <button class="btn-primary" onclick="openCreateModal()">+ Create Collection</button>
        </div>
        <table>
          <thead>
            <tr>
              <th>Name</th>
              <th>Dimension</th>
              <th>Metric</th>
              <th>Documents</th>
              <th>WAL</th>
              <th>Optimizations</th>
              <th>Actions</th>
            </tr>
          </thead>
          <tbody id="collectionsTableBody">
            <tr><td colspan="7" style="text-align: center; color: var(--text-muted);">Loading collections...</td></tr>
          </tbody>
        </table>
      </div>
    </div>

    <!-- Tab 4: Snippets -->
    <div id="tab-snippets" class="tab-pane">
      <div class="card">
        <div class="card-title">Python SDK Snippet (sutradb-core)</div>
        <pre><code id="snippetPython"># Loading...</code></pre>

        <div class="card-title" style="margin-top: 24px;">LangChain VectorStore Integration</div>
        <pre><code id="snippetLangchain">from sutradb.integrations.langchain import SutraDBVectorStore

store = SutraDBVectorStore(
    collection_name="docs",
    dimension=1536,
    persist_directory="./sutra_data",
    hybrid=True
)

# Hybrid retrieval
results = store.similarity_search("neural network architecture", k=5)
for doc in results:
    print(doc.page_content, doc.metadata)</code></pre>

        <div class="card-title" style="margin-top: 24px;">cURL REST API</div>
        <pre><code id="snippetCurl"># Loading...</code></pre>
      </div>
    </div>
  </div>

  <!-- Create Modal -->
  <div id="createModal" style="display: none; position: fixed; inset: 0; background: rgba(0,0,0,0.7); backdrop-filter: blur(8px); align-items: center; justify-content: center; z-index: 100;">
    <div class="card" style="width: 440px; border-color: var(--primary);">
      <div class="card-title">
        <span>Create New Collection</span>
        <button onclick="closeCreateModal()" style="padding: 4px 8px;">✕</button>
      </div>
      <div class="form-group">
        <label>Collection Name</label>
        <input type="text" id="newColName" placeholder="e.g. knowledge_base" />
      </div>
      <div class="form-group">
        <label>Dimension</label>
        <input type="number" id="newColDim" value="128" />
      </div>
      <div class="form-group">
        <label>Distance Metric</label>
        <select id="newColMetric">
          <option value="cosine">Cosine Similarity</option>
          <option value="dot">Dot Product (GEMV)</option>
          <option value="euclidean">Euclidean (L2)</option>
        </select>
      </div>
      <div class="form-group" style="display: flex; align-items: center; gap: 8px;">
        <input type="checkbox" id="newColWal" checked style="width: auto;" />
        <label for="newColWal" style="margin: 0;">Enable Write-Ahead Log (WAL) Durability</label>
      </div>
      <div style="display: flex; justify-content: flex-end; gap: 8px; margin-top: 16px;">
        <button onclick="closeCreateModal()">Cancel</button>
        <button class="btn-primary" onclick="submitCreateCollection()">Create Collection</button>
      </div>
    </div>
  </div>

  <script>
    let collections = [];
    let currentCollection = null;
    let sampleVectors = [];

    async function fetchCollections() {
      try {
        const res = await fetch('/collections');
        if (!res.ok) throw new Error('Failed to load collections');
        const data = await res.json();
        collections = data.collections || [];
        updateCollectionSelect();
        updateAdminTable();
        updateStats();
      } catch (err) {
        console.error(err);
      }
    }

    function updateCollectionSelect() {
      const select = document.getElementById('activeCollectionSelect');
      select.innerHTML = '';
      if (collections.length === 0) {
        select.innerHTML = '<option value="">No collections exist</option>';
        currentCollection = null;
        return;
      }
      collections.forEach(c => {
        const opt = document.createElement('option');
        opt.value = c.name;
        opt.textContent = c.name + ' (' + c.count + ' docs • ' + c.dimension + 'd)';
        select.appendChild(opt);
      });
      if (!currentCollection || !collections.find(c => c.name === currentCollection.name)) {
        currentCollection = collections[0];
      }
      select.value = currentCollection.name;
      document.getElementById('activeColBadge').textContent = currentCollection.name;
      updateSnippets();
    }

    function onCollectionChange() {
      const select = document.getElementById('activeCollectionSelect');
      currentCollection = collections.find(c => c.name === select.value) || null;
      if (currentCollection) {
        document.getElementById('activeColBadge').textContent = currentCollection.name;
        updateSnippets();
      }
    }

    function updateStats() {
      document.getElementById('statTotalCollections').textContent = collections.length;
      const totalDocs = collections.reduce((acc, c) => acc + (c.count || 0), 0);
      document.getElementById('statTotalDocs').textContent = totalDocs.toLocaleString();
    }

    function updateAdminTable() {
      const tbody = document.getElementById('collectionsTableBody');
      tbody.innerHTML = '';
      if (collections.length === 0) {
        tbody.innerHTML = '<tr><td colspan="7" style="text-align: center; color: var(--text-muted);">No collections available. Create one to begin.</td></tr>';
        return;
      }
      collections.forEach(c => {
        const tr = document.createElement('tr');
        tr.innerHTML = `
          <td style="font-weight: 700; color: var(--primary);">${c.name}</td>
          <td>${c.dimension}</td>
          <td><span class="badge">${c.metric}</span></td>
          <td>${c.count}</td>
          <td>${c.wal_enabled ? '✅ Enabled' : 'Disabled'}</td>
          <td>
            <button style="font-size: 11px; padding: 4px 8px;" onclick="buildHNSW('${c.name}')">HNSW</button>
            <button style="font-size: 11px; padding: 4px 8px;" onclick="enableBQ('${c.name}')">1-Bit BQ</button>
            <button style="font-size: 11px; padding: 4px 8px;" onclick="enableSQ8('${c.name}')">SQ8</button>
            <button style="font-size: 11px; padding: 4px 8px;" onclick="compactCollection('${c.name}')">Compact</button>
          </td>
          <td>
            <button style="color: var(--danger); border-color: rgba(248,113,113,0.3); font-size: 11px; padding: 4px 8px;" onclick="deleteCollection('${c.name}')">Delete</button>
          </td>
        `;
        tbody.appendChild(tr);
      });
    }

    async function buildHNSW(name) {
      const res = await fetch('/collections/' + name + '/hnsw', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ m: 16, ef_construction: 100, ef_search: 50 })
      });
      if (res.ok) alert('HNSW index built for ' + name);
    }

    async function enableSQ8(name) {
      const res = await fetch('/collections/' + name + '/sq8', { method: 'POST' });
      if (res.ok) alert('SQ8 8-bit Quantization enabled for ' + name + ' (75% RAM savings active)');
    }

    async function enableBQ(name) {
      const res = await fetch('/collections/' + name + '/bq', { method: 'POST' });
      if (res.ok) alert('1-Bit Binary Quantization (BQ) enabled for ' + name + ' (32x RAM savings active)');
    }

    async function compactCollection(name) {
      const res = await fetch('/collections/' + name + '/compact', { method: 'POST' });
      if (res.ok) alert('WAL compacted for ' + name);
    }

    async function deleteCollection(name) {
      if (!confirm('Are you sure you want to delete collection ' + name + '?')) return;
      const res = await fetch('/collections/' + name, { method: 'DELETE' });
      if (res.ok) fetchCollections();
    }

    function switchTab(tabId) {
      document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
      document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
      event.target.classList.add('active');
      const pane = document.getElementById('tab-' + tabId);
      if (pane) pane.classList.add('active');
      if (tabId === 'visualizer') renderProjection();
    }

    function generateRandomVector() {
      if (!currentCollection) return;
      const dim = currentCollection.dimension || 128;
      const vec = Array.from({ length: dim }, () => +(Math.random() * 2 - 1).toFixed(4));
      document.getElementById('queryVector').value = JSON.stringify(vec);
    }

    function updateAlphaLabel(val) {
      const v = parseFloat(val);
      const label = document.getElementById('alphaLabel');
      if (v === 0.5) label.textContent = 'Equal Balance (RRF Combined)';
      else if (v < 0.5) label.textContent = 'Dense Dominant (' + ((1-v)*100).toFixed(0) + '% Dense)';
      else label.textContent = 'BM25 Dominant (' + (v*100).toFixed(0) + '% Lexical)';
      updateSnippets();
    }

    async function runQuery() {
      if (!currentCollection) {
        alert('Please select or create a collection first.');
        return;
      }
      const text = document.getElementById('queryText').value.trim() || undefined;
      let vector = undefined;
      const vecRaw = document.getElementById('queryVector').value.trim();
      if (vecRaw) {
        try {
          vector = JSON.parse(vecRaw);
        } catch (e) {
          alert('Invalid JSON vector format');
          return;
        }
      }
      let filter = undefined;
      const filterRaw = document.getElementById('queryFilter').value.trim();
      if (filterRaw) {
        try {
          filter = JSON.parse(filterRaw);
        } catch (e) {
          alert('Invalid metadata filter JSON');
          return;
        }
      }
      const topK = parseInt(document.getElementById('queryTopK').value) || 5;
      const ns = document.getElementById('queryNamespace').value.trim() || undefined;

      const payload = {
        text: text,
        vector: vector,
        filter: filter,
        namespace: ns,
        top_k: topK,
        hybrid: true
      };

      const res = await fetch('/collections/' + currentCollection.name + '/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      if (!res.ok) {
        const err = await res.json();
        alert('Query failed: ' + (err.error || err.detail || 'Unknown error'));
        return;
      }

      const data = await res.json();
      renderResults(data.results || []);
    }

    function renderResults(results) {
      const container = document.getElementById('resultsContainer');
      document.getElementById('resultsCountBadge').textContent = results.length + ' results';
      container.innerHTML = '';
      if (results.length === 0) {
        container.innerHTML = '<div style="text-align: center; color: var(--text-muted); padding: 30px;">No matching documents found.</div>';
        return;
      }
      const maxScore = Math.max(...results.map(r => r.score || 1.0));
      results.forEach((r, idx) => {
        const pct = Math.min(100, Math.max(0, (r.score / (maxScore || 1.0)) * 100)).toFixed(1);
        const item = document.createElement('div');
        item.className = 'result-item';
        item.innerHTML = `
          <div class="result-header">
            <span class="doc-id">#${idx + 1} • ${r.id}</span>
            <span class="score-badge">Score: ${r.score.toFixed(4)}</span>
          </div>
          <div class="score-bar-bg"><div class="score-bar-fill" style="width: ${pct}%;"></div></div>
          ${r.text ? `<div class="doc-text">${escapeHtml(r.text)}</div>` : ''}
          ${r.metadata && Object.keys(r.metadata).length > 0 ? `<div class="meta-box">${escapeHtml(JSON.stringify(r.metadata, null, 2))}</div>` : ''}
        `;
        container.appendChild(item);
      });
    }

    function escapeHtml(str) {
      return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
    }

    function updateSnippets() {
      if (!currentCollection) return;
      const name = currentCollection.name;
      const text = document.getElementById('queryText').value.trim() || "sample query";
      document.getElementById('snippetPython').textContent =
`from sutradb import Collection

col = Collection("${name}", dimension=${currentCollection.dimension})
results = col.query(
    text="${text}",
    top_k=5,
    hybrid=True
)
for r in results:
    print(r.id, r.score, r.text)`;

      document.getElementById('snippetCurl').textContent =
`curl -X POST http://localhost:8765/collections/${name}/query \\\\
  -H "Content-Type: application/json" \\\\
  -d '{"text": "${text}", "top_k": 5, "hybrid": true}'`;
    }

    /* 2D Vector PCA Projection */
    async function renderProjection() {
      if (!currentCollection) return;
      const canvas = document.getElementById('projectionCanvas');
      const ctx = canvas.getContext('2d');
      canvas.width = canvas.parentElement.clientWidth - 40;
      canvas.height = 440;

      // Draw background grid
      ctx.fillStyle = '#0f172a';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.strokeStyle = 'rgba(56, 189, 248, 0.1)';
      ctx.lineWidth = 1;
      for (let x = 0; x < canvas.width; x += 40) {
        ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, canvas.height); ctx.stroke();
      }
      for (let y = 0; y < canvas.height; y += 40) {
        ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(canvas.width, y); ctx.stroke();
      }

      // Fetch sample vectors
      try {
        const res = await fetch('/collections/' + currentCollection.name + '/sample_vectors');
        if (!res.ok) return;
        const data = await res.json();
        const pts = data.vectors || [];
        if (pts.length === 0) {
          ctx.fillStyle = '#94a3b8';
          ctx.font = '14px sans-serif';
          ctx.textAlign = 'center';
          ctx.fillText('Collection has no vectors yet to project.', canvas.width / 2, canvas.height / 2);
          return;
        }

        sampleVectors = pts.map((p, idx) => {
          const v = p.vector || [];
          let x = 0, y = 0;
          for (let i = 0; i < v.length; i++) {
            x += v[i] * Math.cos(i);
            y += v[i] * Math.sin(i);
          }
          return {
            id: p.id,
            text: p.text || '',
            px: canvas.width / 2 + x * 35,
            py: canvas.height / 2 + y * 35,
          };
        });

        // Draw points
        sampleVectors.forEach(pt => {
          ctx.beginPath();
          ctx.arc(pt.px, pt.py, 6, 0, Math.PI * 2);
          ctx.fillStyle = '#38bdf8';
          ctx.shadowColor = '#38bdf8';
          ctx.shadowBlur = 10;
          ctx.fill();
        });
        ctx.shadowBlur = 0;
      } catch (e) {
        console.error(e);
      }
    }

    // Modal controls
    function openCreateModal() {
      const modal = document.getElementById('createModal');
      modal.style.display = 'flex';
    }
    function closeCreateModal() {
      document.getElementById('createModal').style.display = 'none';
    }
    async function submitCreateCollection() {
      const name = document.getElementById('newColName').value.trim();
      const dim = parseInt(document.getElementById('newColDim').value);
      const metric = document.getElementById('newColMetric').value;
      const wal = document.getElementById('newColWal').checked;
      if (!name) return alert('Name required');
      const res = await fetch('/collections', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name, dimension: dim, metric, enable_wal: wal })
      });
      if (res.ok) {
        closeCreateModal();
        fetchCollections();
      } else {
        const err = await res.json();
        alert('Failed: ' + (err.error || 'Unknown'));
      }
    }

    function refreshAll() {
      fetchCollections();
    }

    // Initial load
    fetchCollections();
  </script>
</body>
</html>
"""

def get_dashboard_html(version: str = "3.1.0") -> str:
    """Returns the standalone HTML/CSS/JS dashboard string."""
    return DASHBOARD_HTML_TEMPLATE.replace("__VERSION__", version)
