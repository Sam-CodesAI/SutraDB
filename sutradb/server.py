"""
Zero-dependency HTTP REST micro-server for SutraDB.
Enables running SutraDB as a standalone network microservice using Python's built-in http.server.
"""

from http.server import HTTPServer, BaseHTTPRequestHandler
import json
from pathlib import Path
import sys
from typing import Any, Dict, Optional
from urllib.parse import urlparse

from sutradb.core import SutraDB, Collection


class SutraHTTPHandler(BaseHTTPRequestHandler):
    """Handles REST HTTP requests for SutraDB."""

    db: SutraDB = None  # Injected on server startup

    def _send_json(self, status: int, payload: Dict[str, Any]) -> None:
        """Sends a JSON response with CORS headers."""
        data = json.dumps(payload, separators=(',', ':')).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.end_headers()
        self.wfile.write(data)

    def _read_json(self) -> Optional[Dict[str, Any]]:
        """Reads and parses JSON payload from request body."""
        content_length = int(self.headers.get("Content-Length", 0))
        if content_length == 0:
            return None
        body = self.rfile.read(content_length)
        return json.loads(body.decode("utf-8"))

    def do_OPTIONS(self) -> None:
        """Handles CORS pre-flight requests."""
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.end_headers()

    def do_GET(self) -> None:
        """Routes GET requests."""
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")

        if path == "" or path == "/health":
            self._send_json(200, {
                "status": "healthy",
                "version": "2.0.1",
                "collections": self.db.list_collections()
            })
            return

        if path == "/collections":
            cols = []
            for name in self.db.list_collections():
                col = self.db.get_collection(name)
                cols.append(col.stats())
            self._send_json(200, {"collections": cols})
            return

        # /collections/<name>/stats
        parts = path.split("/")
        if len(parts) == 4 and parts[1] == "collections" and parts[3] == "stats":
            col_name = parts[2]
            try:
                col = self.db.get_collection(col_name)
                self._send_json(200, col.stats())
            except KeyError:
                self._send_json(404, {"error": f"Collection '{col_name}' not found"})
            return

        self._send_json(404, {"error": "Endpoint not found"})

    def do_POST(self) -> None:
        """Routes POST requests."""
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")
        parts = path.split("/")

        try:
            body = self._read_json() or {}
        except Exception as e:
            self._send_json(400, {"error": f"Invalid JSON body: {str(e)}"})
            return

        # POST /collections -> Create collection
        if path == "/collections":
            name = body.get("name")
            dimension = body.get("dimension")
            metric = body.get("metric", "cosine")
            if not name or not dimension:
                self._send_json(400, {"error": "Fields 'name' and 'dimension' are required"})
                return
            try:
                col = self.db.create_collection(name=name, dimension=int(dimension), metric=metric)
                self._send_json(201, {"message": "Collection created", "stats": col.stats()})
            except Exception as e:
                self._send_json(400, {"error": str(e)})
            return

        # POST /collections/<name>/insert
        if len(parts) == 4 and parts[1] == "collections" and parts[3] == "insert":
            col_name = parts[2]
            try:
                col = self.db.get_collection(col_name)
                docs = body.get("documents", [])
                if not docs:
                    self._send_json(400, {"error": "Missing 'documents' list"})
                    return
                count = col.insert(docs)
                self._send_json(200, {"inserted": len(docs), "total_count": count})
            except KeyError:
                self._send_json(404, {"error": f"Collection '{col_name}' not found"})
            except Exception as e:
                self._send_json(400, {"error": str(e)})
            return

        # POST /collections/<name>/query
        if len(parts) == 4 and parts[1] == "collections" and parts[3] == "query":
            col_name = parts[2]
            try:
                col = self.db.get_collection(col_name)
                vector = body.get("vector")
                text = body.get("text")
                filter_spec = body.get("filter")
                top_k = int(body.get("top_k", 10))
                hybrid = bool(body.get("hybrid", True))
                alpha = body.get("alpha")

                results = col.query(
                    vector=vector,
                    text=text,
                    filter=filter_spec,
                    top_k=top_k,
                    hybrid=hybrid,
                    alpha=float(alpha) if alpha is not None else None
                )
                self._send_json(200, {
                    "count": len(results),
                    "results": [r.to_dict() for r in results]
                })
            except KeyError:
                self._send_json(404, {"error": f"Collection '{col_name}' not found"})
            except Exception as e:
                self._send_json(400, {"error": str(e)})
            return

        # POST /collections/<name>/save
        if len(parts) == 4 and parts[1] == "collections" and parts[3] == "save":
            col_name = parts[2]
            try:
                col = self.db.get_collection(col_name)
                bytes_written = col.save()
                self._send_json(200, {"message": "Snapshot saved", "bytes": bytes_written})
            except KeyError:
                self._send_json(404, {"error": f"Collection '{col_name}' not found"})
            except Exception as e:
                self._send_json(500, {"error": str(e)})
            return

        self._send_json(404, {"error": "Endpoint not found"})

    def do_DELETE(self) -> None:
        """Routes DELETE requests."""
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")
        parts = path.split("/")

        # DELETE /collections/<name>
        if len(parts) == 3 and parts[1] == "collections":
            col_name = parts[2]
            dropped = self.db.drop_collection(col_name)
            if dropped:
                self._send_json(200, {"message": f"Collection '{col_name}' dropped successfully"})
            else:
                self._send_json(404, {"error": f"Collection '{col_name}' not found"})
            return

        self._send_json(404, {"error": "Endpoint not found"})

    def log_message(self, format: str, *args: Any) -> None:
        """Mutes noisy default HTTP access logs."""
        pass


def run_server(host: str = "0.0.0.0", port: int = 8765, data_dir: str = "./sutra_data") -> None:
    """Starts the SutraDB HTTP server."""
    db = SutraDB(persist_directory=data_dir)
    SutraHTTPHandler.db = db
    server = HTTPServer((host, port), SutraHTTPHandler)
    print(f"⚡ SutraDB v2.0 REST server running at http://{host}:{port}")
    print(f"📁 Persistence directory: {Path(data_dir).resolve()}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down SutraDB server gracefully...")
        server.server_close()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    run_server(port=port)
