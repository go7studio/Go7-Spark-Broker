from __future__ import annotations

import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from spark_broker.client import BrokerClient, ClientError


class RedirectHandler(BaseHTTPRequestHandler):
    target = ""

    def log_message(self, fmt, *args):
        pass

    def do_GET(self):
        self.send_response(302)
        self.send_header("Location", self.target)
        self.send_header("Content-Length", "0")
        self.end_headers()


class CredentialSinkHandler(BaseHTTPRequestHandler):
    called = False
    authorization = None

    def log_message(self, fmt, *args):
        pass

    def do_GET(self):
        type(self).called = True
        type(self).authorization = self.headers.get("Authorization")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", "2")
        self.end_headers()
        self.wfile.write(b"{}")


class ClientSecurityTests(unittest.TestCase):
    def test_remote_plaintext_and_url_credentials_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "HTTPS"):
            BrokerClient("http://example.invalid", "t" * 32)
        with self.assertRaisesRegex(ValueError, "credentials"):
            BrokerClient("https://user:password@example.invalid", "t" * 32)
        BrokerClient("https://example.invalid", "t" * 32)
        BrokerClient("http://127.0.0.1:8790", "t" * 32)

    def test_redirect_is_not_followed_with_bearer_token(self) -> None:
        sink = ThreadingHTTPServer(("127.0.0.1", 0), CredentialSinkHandler)
        redirect = ThreadingHTTPServer(("127.0.0.1", 0), RedirectHandler)
        CredentialSinkHandler.called = False
        CredentialSinkHandler.authorization = None
        RedirectHandler.target = f"http://127.0.0.1:{sink.server_address[1]}/capture"
        threads = [
            threading.Thread(target=sink.serve_forever, daemon=True),
            threading.Thread(target=redirect.serve_forever, daemon=True),
        ]
        for thread in threads:
            thread.start()
        try:
            client = BrokerClient(f"http://127.0.0.1:{redirect.server_address[1]}", "secret-token-value" * 2)
            with self.assertRaises(ClientError):
                client.capabilities()
            self.assertFalse(CredentialSinkHandler.called)
            self.assertIsNone(CredentialSinkHandler.authorization)
        finally:
            redirect.shutdown()
            sink.shutdown()
            redirect.server_close()
            sink.server_close()
            for thread in threads:
                thread.join(5)


if __name__ == "__main__":
    unittest.main()
