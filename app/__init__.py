from flask import Flask

def create_app():
    app = Flask(__name__)
    app.config['JSON_AS_ASCII'] = False
    app.config['MAX_CONTENT_LENGTH'] = 300 * 1024 * 1024  # 100 MB

    from .routes import register_routes
    register_routes(app)

    return app
