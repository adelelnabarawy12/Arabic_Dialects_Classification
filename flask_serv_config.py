from flask import Flask, request, render_template
from project_utils import get_label, RNNModel

app = Flask(__name__, static_folder='assets')

@app.route('/')
def bot_page():
    return render_template('/index.html')
# End Func

@app.route('/get_resposne', methods=['GET', 'POST'])
def get_resposne():
    return get_label(request.json['text'])
# End Func

if __name__ == '__main__':
    app.run()
# End if
