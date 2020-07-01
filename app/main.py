from flask import Flask
from ADGAN.run_test import run

app = Flask(__name__)

@app.route('/')
def hello_world():
  return 'DeepFashion.MIPT! (by sgoldyaev for DLS2020)'

@app.route('/test')
def test_model():
  run()
  return 'success'

if __name__ == '__main__':
  app.run()
