
from flask import Flask, render_template, request

app = Flask(__name__)


# @app.route("/")
# def home():
#     return "Hello! My Flask application is working. Great!!!"


# @app.route("/about")
# def about():
#     return "This is the About Page."

# @app.route("/contact")
# def contact():
#     return "Contact page"

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")


@app.route("/greet", methods=["GET", "POST"])
def greet():

    # If the user opens /greet directly
    if request.method == "GET":
        return render_template("greeting.html")

    # If the user submits the form using POST
    name = request.form.get("student_name", "").strip()
    s_id = request.form.get("student_id", "").strip()

    if not name:
        return "Please enter your name.", 400

    if not s_id:
        return "Please enter your student ID.", 400

    greeting = f"Hello, {name}! Your student ID is {s_id}. Welcome to Flask."

    return greeting


if __name__ == "__main__":
    app.run(debug=True)