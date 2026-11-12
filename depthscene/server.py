"""HTTP 服务：/frame /state + 静态页面（纯标准库，无 Web 框架依赖）。"""

from __future__ import annotations

import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from .pipeline import DepthScene

WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'web')
MAX_FRAME_BYTES = 6 * 1024 * 1024

STATIC = {
    '/': ('live.html', 'text/html; charset=utf-8'),
    '/live.js': ('live.js', 'text/javascript; charset=utf-8'),
    '/styles.css': ('styles.css', 'text/css; charset=utf-8'),
}


class NumpyEncoder(json.JSONEncoder):
    """兜底：任何漏网的 numpy 标量都能序列化。"""

    def default(self, o):
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


def _handler(scene: DepthScene):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'

        def log_message(self, fmt, *args):
            if '/frame' in (fmt % args) or '/state' in (fmt % args):
                return
            sys.stderr.write('%s - %s\n' % (self.address_string(), fmt % args))

        def _json(self, payload: dict, status: int = 200):
            body = json.dumps(payload, ensure_ascii=False, cls=NumpyEncoder).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(body)

        def _file(self, name: str, ctype: str):
            path = os.path.join(WEB_DIR, name)
            if not os.path.exists(path):
                self.send_error(404, 'Not Found')
                return
            with open(path, 'rb') as f:
                body = f.read()
            self.send_response(200)
            self.send_header('Content-Type', ctype)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802
            path = self.path.split('?')[0]
            if path == '/state':
                self._json(scene.last_state or {'ready': False})
                return

            entry = STATIC.get(path)
            if entry:
                self._file(*entry)
            else:
                self.send_error(404, 'Not Found')

        def do_POST(self):  # noqa: N802
            if self.path.split('?')[0] != '/frame':
                self.send_error(404, 'Not Found')
                return

            try:
                length = int(self.headers.get('Content-Length') or 0)
            except ValueError:
                length = 0

            if length <= 0 or length > MAX_FRAME_BYTES:
                self._json({'ready': False, 'error': 'bad frame'}, 400)
                return

            self._json(scene.process(self.rfile.read(length)))

    return Handler


def serve(scene: DepthScene, host: str = '0.0.0.0', port: int = 8000) -> None:
    httpd = ThreadingHTTPServer((host, port), _handler(scene))
    print(f'就绪 →  http://localhost:{port}')
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
