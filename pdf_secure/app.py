from flask import Flask, jsonify, request
from functools import wraps
from werkzeug.security import generate_password_hash, check_password_hash
from pdf2image import convert_from_path
import os, uuid, config
import logging
from main import process_document, BASE_FOLDER

app = Flask(__name__)

USERNAME = config.USERNAME
PASSWORD_HASH = generate_password_hash(config.PASSWORD)


def requires_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth = request.authorization
        if not auth or not (auth.username == USERNAME and check_password_hash(PASSWORD_HASH, auth.password)):
            return jsonify({"error": "Authentication required"}), 401
        return f(*args, **kwargs)
    return decorated


@app.route("/process", methods=["POST"])
@requires_auth
def process_file():
    file = request.files.get("file") or request.files.get("pdf")
    if not file:
        return jsonify({"error": "no file uploaded"}), 400

    filename = f"{uuid.uuid4().hex}_{file.filename}"
    temp_path = os.path.join(BASE_FOLDER, filename)
    file.save(temp_path)

    txn_id = f"TXN{uuid.uuid4().hex[:2]}"
    results = []

    if file.filename.lower().endswith(".pdf"):
        for page_num, page in enumerate(convert_from_path(temp_path), start=1):
            page_path = os.path.join(BASE_FOLDER, f"{txn_id}_page{page_num}.png")
            page.save(page_path, "PNG")
            result = process_document(page_path, f"{txn_id}_page{page_num}")
            result["page_num"] = page_num
            results.append(result)
    else:
        results.append(process_document(temp_path, txn_id))

    return jsonify({"txn_id": txn_id, "results": results})


if __name__ == "__main__":
    app.run(debug=True, port=8081)
