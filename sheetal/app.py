'''
from flask import Flask, request, render_template, Response
import os, uuid
from functools import wraps
from pdf2image import convert_from_path
from main import process_document, BASE_FOLDER

app = Flask(__name__)

# --- Basic Auth ---
USERNAME = "sheetal456"
PASSWORD = "9681"

def check_auth(username, password):
    return username == USERNAME and password == PASSWORD

def authenticate():
    return Response("Authentication required", 401,
                    {"WWW-Authenticate": 'Basic realm="Login Required"'})

def requires_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth = request.authorization
        if not auth or not check_auth(auth.username, auth.password):
            return authenticate()
        return f(*args, **kwargs)
    return decorated


@app.route("/", methods=["GET", "POST"])
@requires_auth
def index():
    if request.method == "POST":
        file = request.files["file"]
        if file:
            filename = f"{uuid.uuid4().hex}_{file.filename}"
            temp_path = os.path.join(BASE_FOLDER, filename)
            file.save(temp_path)
            txn_id = f"TXN{uuid.uuid4().hex[:6]}"
            result = process_document(temp_path, txn_id)
            return render_template("index.html", result=result)
    return render_template("index.html", result=None)


@app.route("/pdf")
@requires_auth
def pdf():
    return render_template("pdf.html")


@app.route("/process_pdf", methods=["POST"])
@requires_auth
def process_pdf():
    file = request.files.get("pdf")
    if not file:
        return "No PDF uploaded", 400

    filename = f"{uuid.uuid4().hex}_{file.filename}"
    temp_pdf_path = os.path.join(BASE_FOLDER, filename)
    file.save(temp_pdf_path)
    txn_id = f"TXN{uuid.uuid4().hex[:6]}"

    pages = convert_from_path(temp_pdf_path)
    results = []
    for page_num, page_image in enumerate(pages, start=1):
        temp_img_path = os.path.join(BASE_FOLDER, f"{txn_id}_page{page_num}.png")
        page_image.save(temp_img_path, "PNG")
        result = process_document(temp_img_path, f"{txn_id}_page{page_num}")
        result["page_num"] = page_num
        results.append(result)

    return render_template("pdf.html", results=results)


if __name__ == "__main__":
    app.run(debug=True)
'''

from flask import Flask, jsonify, request, Response
import os, uuid
from functools import wraps
from pdf2image import convert_from_path
from werkzeug.security import generate_password_hash, check_password_hash
from main import process_document, BASE_FOLDER
import config

app = Flask(__name__)

USERNAME = config.USERNAME
PASSWORD_HASH = generate_password_hash(config.PASSWORD)


def check_auth(username, password):
    return username == USERNAME and check_password_hash(PASSWORD_HASH, password)


def authenticate():
    return jsonify({"error": "Authentication required"}), 401


def requires_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth = request.authorization
        if not auth or not check_auth(auth.username, auth.password):
            return authenticate()
        return f(*args, **kwargs)
    return decorated


@app.route("/", methods=["POST"])
@requires_auth
def index():
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "no file uploaded"}), 400

    filename = f"{uuid.uuid4().hex}_{file.filename}"
    temp_path = os.path.join(BASE_FOLDER, filename)
    file.save(temp_path)

    txn_id = f"TXN{uuid.uuid4().hex[:4]}"
    result = process_document(temp_path, txn_id)

    return jsonify({
        "txn_id": txn_id,
        "result": result
    })


@app.route("/process_pdf", methods=["POST"])
@requires_auth
def process_pdf():
    file = request.files.get("pdf")
    if not file:
        return jsonify({"error": "no PDF uploaded"}), 400

    filename = f"{uuid.uuid4().hex}_{file.filename}"
    temp_pdf_path = os.path.join(BASE_FOLDER, filename)
    file.save(temp_pdf_path)

    txn_id = f"TXN{uuid.uuid4().hex[:6]}"

    pages = convert_from_path(temp_pdf_path)
    results = []
    for page_num, page_image in enumerate(pages, start=1):
        temp_img_path = os.path.join(BASE_FOLDER, f"{txn_id}_page{page_num}.png")
        page_image.save(temp_img_path, "PNG")
        result = process_document(temp_img_path, f"{txn_id}_page{page_num}")
        result["page_num"] = page_num
        results.append(result)

    return jsonify({
        "txn_id": txn_id,
        "results": results
    })


if __name__ == "__main__":
    app.run(debug=True)
