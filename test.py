from flask import Flask
from pyngrok import ngrok  # Correct Ngrok module
import time

app = Flask(__name__)
ngrok.set_auth_token("2N1uyElcHqbXEsvE6616QFzSn4W_6rZ1Ek8vBJNGsKXyRhZ3P")  # Replace with your actual token

@app.route('/')
def hello_world():
    return 'Hello, World!'

if __name__ == '__main__':
    # Open a tunnel on port 5050
    public_url = ngrok.connect(5050)
    print(' * Ngrok tunnel "http://{}" -> "http://127.0.0.1:5050"'.format(public_url))
    
    # Run the Flask app on port 5050
    app.run(port=5050)
    
    # Keep the tunnel open
    time.sleep(10)
