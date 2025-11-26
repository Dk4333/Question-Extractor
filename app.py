from flask import Flask, render_template, request
from tripfactory import get_answer  

app = Flask(__name__)     # IMPORTANT: must be before all @app.route decorators

@app.route('/', methods=['GET', 'POST'])
def index():
    answer = None
    question = ""

    if request.method == 'POST':
        question = request.form.get('question')

        if question:
            # (RAG uses itinerary.txt)
            answer = get_answer("", question)
        else:
            answer = "Please enter a question."

    return render_template('index.html', answer=answer, question=question)

if __name__ == '__main__':
    app.run(debug=True)