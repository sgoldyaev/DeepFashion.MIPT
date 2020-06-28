from flask import Flask
app = Flask(__name__)

@app.route('/')
def hello_world():
  return 'DeepFashion.MIPT! (by sgoldyaev for DLS2020)'

if __name__ == '__main__':
  app.run()
