"""
Embedded Developer Workbench for SutraDB.
Provides an interactive 2026 dark-mode Bento Grid console for visual querying,
batch data ingestion, document exploration, and 2D vector PCA visualization
with zero external CDN dependencies.
"""

DASHBOARD_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>SutraDB Console | Developer Workbench</title>
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
      --warning: #f59e0b;
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

    .container { max-width: 1400px; margin: 0 auto; }

    /* Header */
    header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding-bottom: 24px;
      border-bottom: 1px solid var(--border-color);
      margin-bottom: 24px;
      flex-wrap: wrap;
      gap: 16px;
    }
    .logo-area { display: flex; align-items: center; gap: 14px; }
    .logo-icon {
      width: 44px; height: 44px; border-radius: 12px;
      background: linear-gradient(135deg, #0284c7 0%, #38bdf8 50%, #34d399 100%);
      display: flex; align-items: center; justify-content: center;
      font-weight: 900; font-size: 22px; color: #fff;
      box-shadow: 0 0 24px var(--primary-glow);
    }
    .logo-title h1 { font-size: 22px; font-weight: 800; letter-spacing: -0.5px; display: flex; align-items: center; gap: 8px; }
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
      padding: 8px 16px; font-size: 13px; font-weight: 600; cursor: pointer;
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
    .btn-danger { background: rgba(248, 113, 113, 0.15); color: var(--danger); border-color: rgba(248, 113, 113, 0.3); }
    .btn-danger:hover { background: rgba(248, 113, 113, 0.3); }

    /* Bento Grid Telemetry */
    .bento-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 16px;
      margin-bottom: 24px;
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

    /* Tables */
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

    /* Drag & Drop File Zone */
    .drop-zone {
      border: 2px dashed var(--border-color);
      border-radius: 10px;
      padding: 24px;
      text-align: center;
      color: var(--text-muted);
      cursor: pointer;
      transition: all 0.2s;
      background: rgba(15, 23, 42, 0.4);
    }
    .drop-zone:hover, .drop-zone.dragover {
      border-color: var(--primary);
      background: rgba(56, 189, 248, 0.05);
      color: var(--text-main);
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
            <span class="badge" style="background: rgba(56, 189, 248, 0.2); color: #fff;">Workbench</span>
            <span class="badge">v__VERSION__</span>
            <span class="badge badge-green"><span class="pulse-dot"></span> Online</span>
          </h1>
          <p>Industrial Native Rust Vector & BM25 Hybrid Engine • Dual gRPC & REST Architecture</p>
        </div>
      </div>
      <div class="header-actions">
        <span class="badge" title="Binary gRPC Daemon Port">⚡ gRPC: 50051</span>
        <span class="badge badge-green" title="REST & Workbench Port">🌐 REST: 8765</span>
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
        <div class="label">Quantization Modes</div>
        <div class="value" style="color: var(--primary);"><span style="font-size: 20px;">🛡️</span> 1-Bit BQ & SQ8</div>
        <div class="sub">32x BQ & 75% SQ8 RAM savings</div>
      </div>
    </div>

    <!-- Navigation Tabs -->
    <div class="tabs">
      <button class="tab-btn active" onclick="switchTab('playground', event)">⚡ Hybrid Query Lab</button>
      <button class="tab-btn" onclick="switchTab('ingest', event)">📥 Ingestion & Batch Upsert</button>
      <button class="tab-btn" onclick="switchTab('explorer', event)">🗂️ Document Explorer</button>
      <button class="tab-btn" onclick="switchTab('visualizer', event)">📊 2D Vector Projection (PCA)</button>
      <button class="tab-btn" onclick="switchTab('admin', event)">🛠️ Collection Administration</button>
      <button class="tab-btn" onclick="switchTab('snippets', event)">💻 Client Code Snippets</button>
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
            <label>Query Vector (Dense Embedding)</label>
            <textarea id="queryVector" placeholder="[0.12, -0.45, 0.88, ...]" rows="2"></textarea>
            <button style="margin-top: 6px; font-size: 11px; padding: 4px 8px;" onclick="generateRandomVector()">🎲 Generate Random Vector</button>
          </div>

          <div class="form-group">
            <label>Hybrid Search Ratio (RRF / Linear)</label>
            <div class="slider-container">
              <span style="font-size: 11px; color: var(--accent);">Lexical (0.0)</span>
              <input type="range" id="hybridAlpha" min="0" max="1" step="0.1" value="0.5" oninput="updateAlphaLabel(this.value)">
              <span style="font-size: 11px; color: var(--primary);">Dense (1.0)</span>
            </div>
            <div style="font-size: 11px; color: var(--text-muted); margin-top: 4px;" id="alphaLabel">Equal Balance (RRF Combined)</div>
          </div>

          <div class="form-group">
            <label>Tenant Namespace (Optional)</label>
            <input type="text" id="queryNamespace" placeholder="e.g. tenant_alpha (default: all)" />
          </div>

          <div class="form-group">
            <label>Metadata Filter JSON (AST)</label>
            <textarea id="queryFilter" placeholder='{"category": "ai"}' rows="2"></textarea>
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
            <div style="display: flex; gap: 8px; align-items: center;">
              <span id="queryLatencyBadge" class="badge" style="display: none; background: rgba(52, 211, 153, 0.15); color: var(--accent);">⚡ 0.00ms</span>
              <span id="resultsCountBadge" class="badge">0 results</span>
            </div>
          </div>
          <div id="resultsContainer">
            <div style="text-align: center; color: var(--text-muted); padding: 40px 20px;">
              Enter query parameters and click <b>Execute Hybrid Search</b> to inspect live retrieved vector chunks.
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Tab 2: Ingest Workbench -->
    <div id="tab-ingest" class="tab-pane">
      <div class="panel-grid">
        <div class="card">
          <div class="card-title">
            <span>Single Document Upsert</span>
            <span class="badge">Instant Embed & Save</span>
          </div>
          <div class="form-group">
            <label>Document ID (Optional)</label>
            <input type="text" id="singleDocId" placeholder="e.g. doc_101 (auto-generates UUID if blank)" />
          </div>
          <div class="form-group">
            <label>Document Text (BM25 & Semantic)</label>
            <textarea id="singleDocText" placeholder="Enter raw text content here..." rows="3"></textarea>
          </div>
          <div class="form-group">
            <label>Dense Vector (Optional, comma-separated)</label>
            <textarea id="singleDocVector" placeholder="[0.1, 0.2, 0.3...] or leave blank to auto-embed" rows="2"></textarea>
            <button style="margin-top: 6px; font-size: 11px; padding: 4px 8px;" onclick="generateIngestRandomVector()">🎲 Generate Random Matching Vector</button>
          </div>
          <div class="form-group">
            <label>Metadata JSON</label>
            <textarea id="singleDocMeta" placeholder='{"source": "documentation", "author": "Samarth"}' rows="2">{"source": "workbench"}</textarea>
          </div>
          <div class="form-group">
            <label>Namespace (Optional)</label>
            <input type="text" id="singleDocNamespace" placeholder="e.g. tenant_1" />
          </div>
          <button class="btn-primary" style="width: 100%; justify-content: center; padding: 12px;" onclick="submitSingleDoc()">
            📥 Upsert Single Document
          </button>
        </div>

        <div class="card">
          <div class="card-title">
            <span>Batch JSON / JSONL Ingestion</span>
            <span class="badge">Bulk Ingest</span>
          </div>
          <p style="font-size: 12px; color: var(--text-muted); margin-bottom: 12px;">
            Paste an array of document objects or drag-and-drop a <code>.json</code> / <code>.jsonl</code> file directly into the workbench:
          </p>
          <div class="form-group">
            <textarea id="batchJsonInput" rows="9" style="font-size: 12px;">[
  {
    "id": "doc_01",
    "text": "SutraDB: High-throughput embedded hybrid vector engine in pure Rust.",
    "metadata": {"category": "database", "tier": "production"}
  },
  {
    "id": "doc_02",
    "text": "Sub-0.4ms query round-trip latency powered by native binary gRPC.",
    "metadata": {"category": "network", "tier": "production"}
  }
]</textarea>
          </div>
          <div class="drop-zone" id="fileDropZone" onclick="document.getElementById('fileUploadInput').click()">
            📄 Click to select or drop a <b>.json</b> or <b>.jsonl</b> file here
            <input type="file" id="fileUploadInput" accept=".json,.jsonl" style="display: none;" onchange="handleFileUpload(event)" />
          </div>
          <button class="btn-primary" style="width: 100%; justify-content: center; padding: 12px; margin-top: 14px;" onclick="submitBatchJson()">
            ⚡ Ingest Document Batch
          </button>
        </div>
      </div>
    </div>

    <!-- Tab 3: Document Explorer -->
    <div id="tab-explorer" class="tab-pane">
      <div class="card">
        <div class="card-title">
          <div style="display: flex; align-items: center; gap: 12px;">
            <span>Document Explorer</span>
            <span id="explorerTotalBadge" class="badge">0 total</span>
          </div>
          <div style="display: flex; gap: 8px;">
            <input type="text" id="explorerSearchInput" placeholder="Filter by ID or keyword..." style="width: 240px; padding: 6px 10px;" oninput="filterExplorerDocs()" />
            <button onclick="fetchExplorerDocs()">⟳ Refresh Docs</button>
          </div>
        </div>
        <table>
          <thead>
            <tr>
              <th style="width: 150px;">Document ID</th>
              <th style="width: 110px;">Namespace</th>
              <th>Text Preview</th>
              <th>Metadata</th>
              <th style="width: 140px;">Actions</th>
            </tr>
          </thead>
          <tbody id="explorerTableBody">
            <tr><td colspan="5" style="text-align: center; color: var(--text-muted); padding: 30px;">Select a collection to view documents.</td></tr>
          </tbody>
        </table>
      </div>
    </div>

    <!-- Tab 4: Visualizer -->
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

    <!-- Tab 5: Admin -->
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

    <!-- Tab 6: Snippets -->
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

        <div class="card-title" style="margin-top: 24px;">High-Throughput Binary gRPC Streaming (Tonic & Protobuf)</div>
        <pre><code>// Rust client streaming over HTTP/2 gRPC port 50051
use _sutradb::grpc::proto::sutra_service_client::SutraServiceClient;
use _sutradb::grpc::proto::QueryRequest;

let mut client = SutraServiceClient::connect("http://127.0.0.1:50051").await?;
let response = client.query(QueryRequest {
    collection_name: "docs".into(),
    vector: vec![0.1, 0.2, 0.3, 0.4],
    top_k: 10,
    text: "hybrid search".into(),
    ..Default::default()
}).await?;
println!("Retrieved {} documents with <0.4ms round-trip latency!", response.into_inner().results.len());</code></pre>
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

  <!-- Patch Metadata Modal -->
  <div id="patchModal" style="display: none; position: fixed; inset: 0; background: rgba(0,0,0,0.7); backdrop-filter: blur(8px); align-items: center; justify-content: center; z-index: 100;">
    <div class="card" style="width: 440px; border-color: var(--accent);">
      <div class="card-title">
        <span>Patch Document Metadata</span>
        <button onclick="closePatchModal()" style="padding: 4px 8px;">✕</button>
      </div>
      <div class="form-group">
        <label>Document ID</label>
        <input type="text" id="patchDocId" readonly style="opacity: 0.8;" />
      </div>
      <div class="form-group">
        <label>New Metadata JSON</label>
        <textarea id="patchDocMeta" rows="4"></textarea>
      </div>
      <div style="display: flex; justify-content: flex-end; gap: 8px; margin-top: 16px;">
        <button onclick="closePatchModal()">Cancel</button>
        <button class="btn-primary" onclick="submitPatchMetadata()">Save Changes</button>
      </div>
    </div>
  </div>

  <script>
    let collections = [];
    let currentCollection = null;
    let sampleVectors = [];
    let loadedExplorerDocs = [];

    async function fetchCollections() {
      try {
        const res = await fetch('/collections');
        if (!res.ok) return;
        const data = await res.json();
        collections = data.collections || [];

        const select = document.getElementById('activeCollectionSelect');
        select.innerHTML = '';

        let totalDocs = 0;
        collections.forEach(col => {
          totalDocs += (col.count || 0);
          const opt = document.createElement('option');
          opt.value = col.name;
          opt.textContent = `${col.name} (${col.count} docs, ${col.dimension}d, ${col.metric})`;
          select.appendChild(opt);
        });

        document.getElementById('statTotalDocs').textContent = totalDocs.toLocaleString();
        document.getElementById('statTotalCollections').textContent = collections.length;

        if (collections.length > 0) {
          if (!currentCollection || !collections.find(c => c.name === currentCollection.name)) {
            currentCollection = collections[0];
          } else {
            currentCollection = collections.find(c => c.name === currentCollection.name);
          }
          select.value = currentCollection.name;
          document.getElementById('activeColBadge').textContent = currentCollection.name;
        } else {
          currentCollection = null;
          select.innerHTML = '<option value="">No collections created</option>';
          document.getElementById('activeColBadge').textContent = 'None';
        }

        renderCollectionsTable();
        updateCodeSnippets();
      } catch (e) {
        console.error('Failed to fetch collections:', e);
      }
    }

    function onCollectionChange() {
      const select = document.getElementById('activeCollectionSelect');
      const selected = collections.find(c => c.name === select.value);
      if (selected) {
        currentCollection = selected;
        document.getElementById('activeColBadge').textContent = currentCollection.name;
        updateCodeSnippets();
        if (document.getElementById('tab-explorer').classList.contains('active')) {
          fetchExplorerDocs();
        }
      }
    }

    function switchTab(tabId, evt) {
      document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
      document.querySelectorAll('.tab-pane').forEach(pane => pane.classList.remove('active'));
      if (evt && evt.target) evt.target.classList.add('active');
      const activePane = document.getElementById('tab-' + tabId);
      if (activePane) activePane.classList.add('active');

      if (tabId === 'visualizer') {
        renderProjection();
      } else if (tabId === 'explorer') {
        fetchExplorerDocs();
      }
    }

    function updateAlphaLabel(val) {
      const label = document.getElementById('alphaLabel');
      const v = parseFloat(val);
      if (v === 0.0) label.textContent = '100% BM25 Lexical Score (Pure Keyword)';
      else if (v === 1.0) label.textContent = '100% Dense Similarity (Pure Vector)';
      else label.textContent = `Hybrid Linear Fusion (Alpha: ${v})`;
    }

    function generateRandomVector() {
      if (!currentCollection) return alert('Create or select a collection first');
      const dim = currentCollection.dimension || 128;
      const vec = Array.from({ length: dim }, () => parseFloat((Math.random() * 2 - 1).toFixed(4)));
      document.getElementById('queryVector').value = JSON.stringify(vec);
    }

    function generateIngestRandomVector() {
      if (!currentCollection) return alert('Create or select a collection first');
      const dim = currentCollection.dimension || 128;
      const vec = Array.from({ length: dim }, () => parseFloat((Math.random() * 2 - 1).toFixed(4)));
      document.getElementById('singleDocVector').value = JSON.stringify(vec);
    }

    async function runQuery() {
      if (!currentCollection) return alert('Please create or select a collection first.');
      const text = document.getElementById('queryText').value.trim();
      const vecStr = document.getElementById('queryVector').value.trim();
      const alpha = parseFloat(document.getElementById('hybridAlpha').value);
      const ns = document.getElementById('queryNamespace').value.trim();
      const filterStr = document.getElementById('queryFilter').value.trim();
      const topK = parseInt(document.getElementById('queryTopK').value) || 5;

      let vector = null;
      if (vecStr) {
        try {
          vector = JSON.parse(vecStr);
        } catch (e) {
          return alert('Invalid vector JSON array');
        }
      }

      let filter = null;
      if (filterStr) {
        try {
          filter = JSON.parse(filterStr);
        } catch (e) {
          return alert('Invalid filter JSON syntax');
        }
      }

      const payload = {
        top_k: topK,
        hybrid: true,
        alpha: alpha,
        text: text || null,
        vector: vector,
        namespace: ns || null,
        filter: filter
      };

      const startTime = performance.now();
      try {
        const res = await fetch(`/collections/${currentCollection.name}/query`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });

        const elapsed = (performance.now() - startTime).toFixed(2);
        const latencyBadge = document.getElementById('queryLatencyBadge');
        latencyBadge.textContent = `⚡ ${elapsed}ms`;
        latencyBadge.style.display = 'inline-flex';

        if (!res.ok) {
          const err = await res.json();
          alert('Query failed: ' + (err.detail || err.error || 'Server error'));
          return;
        }

        const data = await res.json();
        renderResults(data.results || []);
      } catch (e) {
        alert('Query network error: ' + e);
      }
    }

    function renderResults(results) {
      const container = document.getElementById('resultsContainer');
      document.getElementById('resultsCountBadge').textContent = `${results.length} results`;
      container.replaceChildren();

      if (results.length === 0) {
        const empty = document.createElement('div');
        empty.style.cssText = 'text-align: center; color: var(--text-muted); padding: 40px;';
        empty.textContent = 'No matching documents found.';
        container.appendChild(empty);
        return;
      }

      results.forEach((r, idx) => {
        const score = typeof r.score === 'number' ? r.score.toFixed(4) : String(r.score ?? '');
        const widthPct = Math.min(100, Math.max(10, (Number(r.score) || 0.5) * 100));

        const item = document.createElement('div');
        item.className = 'result-item';

        const header = document.createElement('div');
        header.className = 'result-header';

        const docIdSpan = document.createElement('span');
        docIdSpan.className = 'doc-id';
        docIdSpan.textContent = `#${idx + 1} • ${String(r.id ?? '')}`;

        const scoreBadge = document.createElement('span');
        scoreBadge.className = 'score-badge';
        scoreBadge.textContent = `Score: ${score}`;

        header.append(docIdSpan, scoreBadge);

        const barBg = document.createElement('div');
        barBg.className = 'score-bar-bg';
        const barFill = document.createElement('div');
        barFill.className = 'score-bar-fill';
        barFill.style.width = `${widthPct}%`;
        barBg.appendChild(barFill);

        const docText = document.createElement('div');
        docText.className = 'doc-text';
        docText.textContent = r.text || '(No text content)';

        const metaBox = document.createElement('div');
        metaBox.className = 'meta-box';
        metaBox.textContent = JSON.stringify(r.metadata || {}, null, 2);

        item.append(header, barBg, docText, metaBox);
        container.appendChild(item);
      });
    }

    /* Ingestion Workbench Functions */
    async function submitSingleDoc() {
      if (!currentCollection) return alert('Select or create a collection first');
      const id = document.getElementById('singleDocId').value.trim() || undefined;
      const text = document.getElementById('singleDocText').value.trim();
      const vecStr = document.getElementById('singleDocVector').value.trim();
      const metaStr = document.getElementById('singleDocMeta').value.trim();
      const ns = document.getElementById('singleDocNamespace').value.trim() || undefined;

      let vector = undefined;
      if (vecStr) {
        try { vector = JSON.parse(vecStr); } catch (e) { return alert('Invalid vector array JSON'); }
      }
      let metadata = {};
      if (metaStr) {
        try { metadata = JSON.parse(metaStr); } catch (e) { return alert('Invalid metadata JSON'); }
      }

      const doc = { id, text, vector, metadata, namespace: ns };
      try {
        const res = await fetch(`/collections/${currentCollection.name}/upsert`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify([doc])
        });
        if (res.ok) {
          alert('Document successfully upserted!');
          document.getElementById('singleDocText').value = '';
          fetchCollections();
        } else {
          const err = await res.json();
          alert('Upsert failed: ' + (err.detail || err.error));
        }
      } catch (e) {
        alert('Network error: ' + e);
      }
    }

    async function submitBatchJson() {
      if (!currentCollection) return alert('Select or create a collection first');
      const raw = document.getElementById('batchJsonInput').value.trim();
      let docs = [];
      try {
        docs = JSON.parse(raw);
        if (!Array.isArray(docs)) docs = [docs];
      } catch (e) {
        return alert('Invalid JSON format: ' + e.message);
      }

      try {
        const res = await fetch(`/collections/${currentCollection.name}/upsert`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(docs)
        });
        if (res.ok) {
          const resp = await res.json();
          alert(`Successfully ingested ${docs.length} documents! Total in collection: ${resp.total_count}`);
          fetchCollections();
        } else {
          const err = await res.json();
          alert('Batch upsert failed: ' + (err.detail || err.error));
        }
      } catch (e) {
        alert('Network error: ' + e);
      }
    }

    function handleFileUpload(evt) {
      const file = evt.target.files[0];
      if (!file) return;
      const reader = new FileReader();
      reader.onload = function(e) {
        const content = e.target.result;
        try {
          if (file.name.endsWith('.jsonl')) {
            const lines = content.trim().split('\\n').map(l => JSON.parse(l));
            document.getElementById('batchJsonInput').value = JSON.stringify(lines, null, 2);
          } else {
            document.getElementById('batchJsonInput').value = content;
          }
        } catch (err) {
          alert('Error parsing uploaded file: ' + err.message);
        }
      };
      reader.readAsText(file);
    }

    /* Document Explorer Functions */
    async function fetchExplorerDocs() {
      if (!currentCollection) return;
      const tbody = document.getElementById('explorerTableBody');
      tbody.innerHTML = '<tr><td colspan="5" style="text-align: center; padding: 20px;">Fetching documents...</td></tr>';
      try {
        const res = await fetch(`/collections/${currentCollection.name}/documents?limit=100`);
        if (!res.ok) {
          tbody.innerHTML = '<tr><td colspan="5" style="text-align: center; color: var(--danger);">Failed to load documents</td></tr>';
          return;
        }
        const data = await res.json();
        loadedExplorerDocs = data.documents || [];
        document.getElementById('explorerTotalBadge').textContent = `${data.total || loadedExplorerDocs.length} total`;
        renderExplorerTable(loadedExplorerDocs);
      } catch (e) {
        tbody.innerHTML = '<tr><td colspan="5" style="text-align: center; color: var(--danger);">Network error</td></tr>';
      }
    }

    function renderExplorerTable(docs) {
      const tbody = document.getElementById('explorerTableBody');
      tbody.replaceChildren();
      if (docs.length === 0) {
        const tr = document.createElement('tr');
        const td = document.createElement('td');
        td.colSpan = 5;
        td.style.cssText = 'text-align: center; color: var(--text-muted); padding: 30px;';
        td.textContent = 'No documents in this collection.';
        tr.appendChild(td);
        tbody.appendChild(tr);
        return;
      }
      docs.forEach(doc => {
        const tr = document.createElement('tr');

        const tdId = document.createElement('td');
        const codeId = document.createElement('code');
        codeId.style.cssText = 'color: var(--primary); font-family: var(--font-mono); font-size: 11px;';
        codeId.textContent = String(doc.id ?? '');
        tdId.appendChild(codeId);

        const tdNs = document.createElement('td');
        const nsBadge = document.createElement('span');
        nsBadge.className = 'badge';
        nsBadge.style.fontSize = '10px';
        nsBadge.textContent = doc.namespace || 'default';
        tdNs.appendChild(nsBadge);

        const tdText = document.createElement('td');
        const textDiv = document.createElement('div');
        textDiv.style.cssText = 'max-width: 320px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; font-size: 12px;';
        textDiv.textContent = doc.text || '(empty)';
        tdText.appendChild(textDiv);

        const tdMeta = document.createElement('td');
        const metaPre = document.createElement('pre');
        metaPre.style.cssText = 'margin: 0; font-size: 11px; max-width: 250px; overflow-x: auto; color: var(--text-muted);';
        metaPre.textContent = JSON.stringify(doc.metadata || {});
        tdMeta.appendChild(metaPre);

        const tdActions = document.createElement('td');
        const btnGroup = document.createElement('div');
        btnGroup.style.cssText = 'display: flex; gap: 6px;';

        const editBtn = document.createElement('button');
        editBtn.style.cssText = 'padding: 4px 8px; font-size: 11px;';
        editBtn.textContent = '✏️ Edit';
        editBtn.addEventListener('click', () => openPatchModal(String(doc.id ?? ''), JSON.stringify(doc.metadata || {})));

        const delBtn = document.createElement('button');
        delBtn.className = 'btn-danger';
        delBtn.style.cssText = 'padding: 4px 8px; font-size: 11px;';
        delBtn.textContent = '🗑️';
        delBtn.addEventListener('click', () => deleteDoc(String(doc.id ?? '')));

        btnGroup.append(editBtn, delBtn);
        tdActions.appendChild(btnGroup);

        tr.append(tdId, tdNs, tdText, tdMeta, tdActions);
        tbody.appendChild(tr);
      });
    }

    function filterExplorerDocs() {
      const q = document.getElementById('explorerSearchInput').value.toLowerCase();
      const filtered = loadedExplorerDocs.filter(d => 
        (d.id && d.id.toLowerCase().includes(q)) || 
        (d.text && d.text.toLowerCase().includes(q)) ||
        JSON.stringify(d.metadata || {}).toLowerCase().includes(q)
      );
      renderExplorerTable(filtered);
    }

    async function deleteDoc(id) {
      if (!confirm(`Delete document "${id}"?`)) return;
      try {
        const res = await fetch(`/collections/${currentCollection.name}/documents/${id}`, { method: 'DELETE' });
        if (res.ok) {
          fetchExplorerDocs();
          fetchCollections();
        } else {
          alert('Delete failed');
        }
      } catch (e) {
        alert('Network error: ' + e);
      }
    }

    function openPatchModal(id, metaJson) {
      document.getElementById('patchDocId').value = id;
      try {
        const parsed = JSON.parse(metaJson);
        document.getElementById('patchDocMeta').value = JSON.stringify(parsed, null, 2);
      } catch (e) {
        document.getElementById('patchDocMeta').value = metaJson;
      }
      document.getElementById('patchModal').style.display = 'flex';
    }

    function closePatchModal() {
      document.getElementById('patchModal').style.display = 'none';
    }

    async function submitPatchMetadata() {
      const id = document.getElementById('patchDocId').value;
      const raw = document.getElementById('patchDocMeta').value;
      let metadata = {};
      try {
        metadata = JSON.parse(raw);
      } catch (e) {
        return alert('Invalid JSON metadata');
      }

      try {
        const res = await fetch(`/collections/${currentCollection.name}/documents/${id}/metadata`, {
          method: 'PATCH',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ metadata })
        });
        if (res.ok) {
          closePatchModal();
          fetchExplorerDocs();
        } else {
          alert('Failed to update metadata');
        }
      } catch (e) {
        alert('Error: ' + e);
      }
    }

    /* Admin Tab Actions */
    function renderCollectionsTable() {
      const tbody = document.getElementById('collectionsTableBody');
      tbody.replaceChildren();
      if (collections.length === 0) {
        const tr = document.createElement('tr');
        const td = document.createElement('td');
        td.colSpan = 7;
        td.style.cssText = 'text-align: center; color: var(--text-muted); padding: 30px;';
        td.textContent = 'No collections found. Click "+ Create Collection" above.';
        tr.appendChild(td);
        tbody.appendChild(tr);
        return;
      }

      collections.forEach(col => {
        const tr = document.createElement('tr');

        const tdName = document.createElement('td');
        const bName = document.createElement('b');
        bName.textContent = String(col.name ?? '');
        tdName.appendChild(bName);

        const tdDim = document.createElement('td');
        const cDim = document.createElement('code');
        cDim.textContent = String(col.dimension ?? '');
        tdDim.appendChild(cDim);

        const tdMetric = document.createElement('td');
        const cMetric = document.createElement('code');
        cMetric.textContent = String(col.metric ?? '');
        tdMetric.appendChild(cMetric);

        const tdCount = document.createElement('td');
        tdCount.textContent = Number(col.count || 0).toLocaleString();

        const tdWal = document.createElement('td');
        tdWal.textContent = col.enable_wal ? '✅ Active' : '❌ Disabled';

        const tdOpt = document.createElement('td');
        const addBadge = (text, cls, color) => {
          const span = document.createElement('span');
          span.className = cls;
          if (color) span.style.color = color;
          span.textContent = text;
          tdOpt.appendChild(span);
          tdOpt.appendChild(document.createTextNode(' '));
        };
        let hasOpt = false;
        if (col.has_hnsw) { addBadge('HNSW Graph', 'badge badge-green', null); hasOpt = true; }
        if (col.quantized_sq8) { addBadge('SQ8 (75%)', 'badge', '#60a5fa'); hasOpt = true; }
        if (col.quantized_bq) { addBadge('1-Bit BQ (32x)', 'badge', '#c084fc'); hasOpt = true; }
        if (col.quantized_pq) { addBadge('PQ (64x)', 'badge', '#f472b6'); hasOpt = true; }
        if (!hasOpt) {
          const span = document.createElement('span');
          span.style.cssText = 'color: var(--text-muted); font-size: 11px;';
          span.textContent = 'FP32 Dense';
          tdOpt.appendChild(span);
        }

        const tdActions = document.createElement('td');
        const btnWrap = document.createElement('div');
        btnWrap.style.cssText = 'display: flex; gap: 6px; flex-wrap: wrap;';
        const colName = String(col.name ?? '');
        const mkBtn = (label, handler, isDanger = false) => {
          const btn = document.createElement('button');
          if (isDanger) btn.className = 'btn-danger';
          btn.style.cssText = 'padding: 4px 8px; font-size: 11px;';
          btn.textContent = label;
          btn.addEventListener('click', () => handler(colName));
          return btn;
        };
        btnWrap.append(
          mkBtn('🚀 HNSW', triggerHNSW),
          mkBtn('📦 SQ8', triggerSQ8),
          mkBtn('🛡️ BQ', triggerBQ),
          mkBtn('🧹 Compact', triggerCompaction),
          mkBtn('🗑️ Drop', dropCollection, true)
        );
        tdActions.appendChild(btnWrap);

        tr.append(tdName, tdDim, tdMetric, tdCount, tdWal, tdOpt, tdActions);
        tbody.appendChild(tr);
      });
    }

    async function triggerHNSW(name) {
      if (!confirm(`Build HNSW hierarchical graph for collection '${name}'?`)) return;
      const res = await fetch(`/collections/${name}/hnsw`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ m: 16, ef_construction: 64, ef_search: 32 })
      });
      if (res.ok) { alert('HNSW graph successfully built!'); fetchCollections(); }
      else alert('Failed to build HNSW graph');
    }

    async function triggerSQ8(name) {
      const res = await fetch(`/collections/${name}/sq8`, { method: 'POST' });
      if (res.ok) { alert('SQ8 quantization applied (75% RAM reduction)!'); fetchCollections(); }
      else alert('Failed to apply SQ8');
    }

    async function triggerBQ(name) {
      const res = await fetch(`/collections/${name}/bq`, { method: 'POST' });
      if (res.ok) { alert('1-Bit Binary Quantization enabled (32x RAM reduction)!'); fetchCollections(); }
      else alert('Failed to apply BQ');
    }

    async function triggerCompaction(name) {
      const res = await fetch(`/collections/${name}/compact`, { method: 'POST' });
      if (res.ok) { alert('Write-Ahead Log compacted successfully!'); fetchCollections(); }
      else alert('Compaction failed');
    }

    async function dropCollection(name) {
      if (!confirm(`Permanently drop collection '${name}' and delete all vectors?`)) return;
      const res = await fetch(`/collections/${name}`, { method: 'DELETE' });
      if (res.ok) { fetchCollections(); }
      else alert('Failed to drop collection');
    }

    /* 2D Projection PCA */
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

        const dim = (pts[0].vector || []).length;
        const nPts = pts.length;
        // Center vectors for true 2D PCA via Gram matrix (N x N) power iteration
        const mean = new Array(dim).fill(0);
        for (let idx = 0; idx < nPts; idx++) {
          const v = pts[idx].vector || [];
          for (let d = 0; d < dim; d++) mean[d] += (v[d] || 0);
        }
        for (let d = 0; d < dim; d++) mean[d] /= nPts;

        const centered = pts.map(p => {
          const v = p.vector || [];
          const row = new Array(dim);
          for (let d = 0; d < dim; d++) row[d] = (v[d] || 0) - mean[d];
          return row;
        });

        const gram = Array.from({ length: nPts }, () => new Array(nPts).fill(0));
        for (let r = 0; r < nPts; r++) {
          for (let c = r; c < nPts; c++) {
            let dot = 0;
            for (let d = 0; d < dim; d++) dot += centered[r][d] * centered[c][d];
            gram[r][c] = dot;
            gram[c][r] = dot;
          }
        }

        function topEigenvector(mat, size, ortho) {
          let vec = Array.from({ length: size }, (_, k) => ((k % 5) + 1) / size);
          const normInit = Math.hypot(...vec) || 1;
          vec = vec.map(x => x / normInit);
          for (let iter = 0; iter < 30; iter++) {
            const next = new Array(size).fill(0);
            for (let r = 0; r < size; r++) {
              let sum = 0;
              for (let c = 0; c < size; c++) sum += mat[r][c] * vec[c];
              next[r] = sum;
            }
            if (ortho) {
              let proj = 0;
              for (let r = 0; r < size; r++) proj += next[r] * ortho[r];
              for (let r = 0; r < size; r++) next[r] -= proj * ortho[r];
            }
            const norm = Math.hypot(...next);
            if (norm < 1e-12) break;
            for (let r = 0; r < size; r++) vec[r] = next[r] / norm;
          }
          let rayleigh = 0;
          for (let r = 0; r < size; r++) {
            let rowDot = 0;
            for (let c = 0; c < size; c++) rowDot += mat[r][c] * vec[c];
            rayleigh += vec[r] * rowDot;
          }
          const scale = Math.sqrt(Math.max(0, rayleigh));
          return { dir: vec, coords: vec.map(x => x * scale) };
        }

        const pc1 = topEigenvector(gram, nPts, null);
        const pc2 = topEigenvector(gram, nPts, pc1.dir);
        let maxAbs = 1e-6;
        for (let idx = 0; idx < nPts; idx++) {
          maxAbs = Math.max(maxAbs, Math.abs(pc1.coords[idx]), Math.abs(pc2.coords[idx]));
        }
        const plotScale = (Math.min(canvas.width, canvas.height) * 0.38) / maxAbs;

        sampleVectors = pts.map((p, idx) => ({
          id: p.id,
          text: p.text || '',
          px: canvas.width / 2 + pc1.coords[idx] * plotScale,
          py: canvas.height / 2 + pc2.coords[idx] * plotScale,
        }));

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

    function updateCodeSnippets() {
      const colName = currentCollection ? currentCollection.name : 'knowledge_base';
      const dim = currentCollection ? currentCollection.dimension : 128;

      document.getElementById('snippetPython').textContent = 
`from sutradb import SutraDB

db = SutraDB(persist_directory="./sutra_data")
col = db.get_collection("${colName}")

# Hybrid Query (Dense Vector + BM25 Lexical + RRF)
results = col.search(
    query="transformer architecture",
    top_k=5,
    hybrid=True
)

for r in results:
    print(f"[{r.score:.4f}] {r.text} | Metadata: {r.metadata}")`;

      document.getElementById('snippetCurl').textContent =
`# High-Throughput REST Search
curl -X POST http://localhost:8765/collections/${colName}/query \\
  -H "Content-Type: application/json" \\
  -d '{
    "text": "transformer architecture",
    "top_k": 5,
    "hybrid": true
  }'`;
    }

    function openCreateModal() {
      document.getElementById('createModal').style.display = 'flex';
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

    function escapeHtml(str) {
      if (!str) return '';
      return String(str)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');
    }

    fetchCollections();
  </script>
</body>
</html>
"""

def get_dashboard_html(version: str = "5.5.0") -> str:
    """Returns the standalone HTML/CSS/JS developer workbench string."""
    return DASHBOARD_HTML_TEMPLATE.replace("__VERSION__", version)
