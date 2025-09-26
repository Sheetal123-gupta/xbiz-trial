from flask import Flask, jsonify, request
import os, uuid
from datetime import timedelta
from pdf2image import convert_from_path
from werkzeug.security import generate_password_hash, check_password_hash
from flask_jwt_extended import (
    JWTManager, create_access_token, create_refresh_token,
    jwt_required, get_jwt_identity
)
from main import process_document, BASE_FOLDER
import config

app = Flask(__name__)

# JWT config
app.config["JWT_SECRET_KEY"] = config.JWT_SECRET_KEY
app.config["JWT_ACCESS_TOKEN_EXPIRES"] = timedelta(minutes=15)   # access token life
app.config["JWT_REFRESH_TOKEN_EXPIRES"] = timedelta(days=7)     # refresh token life
jwt = JWTManager(app)

# Hash the configured password in memory (you may store a hash in config instead)
USERNAME = config.USERNAME
PASSWORD_HASH = generate_password_hash(config.PASSWORD)


# ---------- AUTH: /login and /refresh ----------
@app.route("/login", methods=["POST"])
def login():
    data = request.get_json(force=True, silent=True)
    if not data:
        return jsonify({"error": "missing JSON body"}), 400

    username = data.get("username")
    password = data.get("password")

    if username != USERNAME or not check_password_hash(PASSWORD_HASH, str(password)):
        return jsonify({"error": "invalid credentials"}), 401

    access_token = create_access_token(identity=username)
    refresh_token = create_refresh_token(identity=username)
    return jsonify({
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer",
        "expires_in": app.config["JWT_ACCESS_TOKEN_EXPIRES"].total_seconds()
    }), 200


@app.route("/refresh", methods=["POST"])
@jwt_required(refresh=True)
def refresh():
    """Send a fresh access token using a refresh token in Authorization header."""
    current_user = get_jwt_identity()
    new_access = create_access_token(identity=current_user)
    return jsonify({"access_token": new_access}), 200


# ---------- FILE PROCESSING ENDPOINT(s) ----------
# Example unified endpoint that accepts either an image (file) or a pdf (pdf)
@app.route("/process", methods=["POST"])
@jwt_required()   # requires Authorization: Bearer <access_token>
def process_file():
    # Accept either form field "file" or "pdf"
    uploaded = request.files.get("file") or request.files.get("pdf")
    if not uploaded:
        return jsonify({"error": "no file uploaded; use form-data key 'file' or 'pdf'"}), 400

    filename = f"{uuid.uuid4().hex}_{uploaded.filename}"
    temp_path = os.path.join(BASE_FOLDER, filename)
    uploaded.save(temp_path)

    txn_id = f"TXN{uuid.uuid4().hex[:6]}"
    results = []

    # If PDF, split into pages
    if uploaded.filename.lower().endswith(".pdf"):
        pages = convert_from_path(temp_path)
        for page_num, page_image in enumerate(pages, start=1):
            temp_img_path = os.path.join(BASE_FOLDER, f"{txn_id}_page{page_num}.png")
            page_image.save(temp_img_path, "PNG")
            result = process_document(temp_img_path, f"{txn_id}_page{page_num}")
            result["page_num"] = page_num
            results.append(result)
    else:
        # image or other: process directly
        result = process_document(temp_path, txn_id)
        results.append(result)

    return jsonify({"txn_id": txn_id, "results": results}), 200


if __name__ == "__main__":
    app.run(debug=True, port=8080)
