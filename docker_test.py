import http.server
import socketserver

# Define the server address and port
HOST = "0.0.0.0"
PORT = 3000


# Define a request handler class by extending the BaseHTTPRequestHandler
class MyRequestHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        self.wfile.write(b"Hello from a container!")


# Create a server object with the defined handler
with socketserver.TCPServer((HOST, PORT), MyRequestHandler) as server:
    print(f"Server started at {HOST}:{PORT}")
    # Start the server and keep it running until interrupted
    server.serve_forever()
