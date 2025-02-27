import logging
import threading
from flask import Flask, send_file, render_template_string, jsonify

# To disable server log messages in terminal
logging.getLogger('werkzeug').setLevel(logging.ERROR)

# Global metadata state used by the server
global_state = {
    "iteration": 0,
    "loss": 0.0
}

app = Flask(__name__)

@app.route("/")
def index():
    return render_template_string('''
        <html>
            <head>
                <title>Live Style Transfer</title>
                <script>
                    function refreshImage(){
                        var img = document.getElementById("live-image");
                        img.src = "/current_image?" + new Date().getTime();
                    }
                    function refreshMetadata(){
                        fetch('/metadata')
                            .then(response => response.json())
                            .then(data => {
                                document.getElementById("iteration").innerText = "Iteration: " + data.iteration;
                                document.getElementById("loss").innerText = "Loss: " + data.loss;
                            });
                    }
                    setInterval(refreshImage, 1000);
                    setInterval(refreshMetadata, 1000);
                </script>
            </head>
            <body>
                <h1>Live Style Transfer</h1>
                <img id="live-image" src="/current_image" alt="Live Style Transfer" style="max-width: 512px;">
                <div id="metadata">
                    <p id="iteration">Iteration: 0</p>
                    <p id="loss">Loss: 0.0</p>
                </div>
            </body>
        </html>
    ''')

@app.route("/metadata")
def metadata():
    return jsonify(global_state)

@app.route("/current_image")
def current_image():
    return send_file("current.jpg", mimetype='image/jpeg')

def run_flask():
    app.run(port=5000, debug=False, use_reloader=False)

def start_server_in_background():
    flask_thread = threading.Thread(target=run_flask)
    flask_thread.daemon = True
    flask_thread.start()
