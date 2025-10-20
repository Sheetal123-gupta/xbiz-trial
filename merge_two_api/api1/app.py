from flask import Flask,request,jsonify
app=Flask(__name__)
@app.route('/calculate' ,methods=['POST'])
def calculate():
  data=request.get_json()
  data1=data.get('data1')
  data2=data.get('data2')

  result={
    "+":data1+data2,
    "-":data1-data2,
    "*":data1*data2,
    "/":data1/data2
  }
  return jsonify(result)

if __name__=='__main__':
  app.run(host='0.0.0.0',port=5000)
