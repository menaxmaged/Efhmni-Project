from app import create_app
# from app.middleware import SlashFixerMiddleware
from app.routes import start_ngrok

app = create_app()
# app.wsgi_app = SlashFixerMiddleware(app.wsgi_app)

if __name__ == "__main__":
    start_ngrok(5050)
    app.run(port=5050, host="0.0.0.0")
