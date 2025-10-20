from flask import Flask, request, jsonify
import requests

app = Flask(__name__)

@app.route('/operation', methods=['POST'])
def operation():
    data = request.get_json()
    data1 = data.get('data1')
    data2 = data.get('data2')
    perform = data.get('perform')
    merge = data.get('merge', False)

    # Perform operation locally (for reference, optional)
    local_result = None
    if perform == '+':
        local_result = data1 + data2
    elif perform == '-':
        local_result = data1 - data2
    elif perform == '*':
        local_result = data1 * data2
    elif perform == '/':
        local_result = data1 / data2
    else:
        return jsonify({"error": "invalid operation"}), 400

    response = {}

    if merge:
        # Call API1
        api1_res = requests.post('http://api1:5000/calculate', json={"data1": data1, "data2": data2}).json()
        response["api2_result"] = api1_res          # full response from API1
        response["api1_result"] = api1_res[perform]  # only requested operation
    else:
        response["api1_result"] = local_result
        response["api2_result"] = None

    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
