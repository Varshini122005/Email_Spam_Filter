from flask import Flask, render_template, request
from spam_filter import SpamFilter

app = Flask(__name__)
spam_filter = SpamFilter()
spam_filter.train("data/training_data.csv")  # Train the model

@app.route("/")
def upload_page():
    return render_template("upload.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Check if a file was uploaded
    email_file = request.files.get("email_file")
    email_content = ""

    if email_file:
        # Read file content and decode it
        email_content = email_file.read().decode('utf-8')
    else:
        # Fall back to text input if no file is uploaded
        email_content = request.form.get("email_content", "")

    if not email_content.strip():
        return "No email content provided! Please enter text or upload a file."

    # Predict spam or not
    result = spam_filter.predict(email_content)
    prediction = "Not Spam" if result == "ham" else "Spam"

    return render_template("results.html", prediction=prediction, email=email_content)

if __name__ == "__main__":
    app.run(debug=True)
