from flask import Flask, render_template, request, redirect, url_for
import os

app = Flask(__name__, template_folder="templates", static_folder='static')

# لتخزين الفيديوهات المرفوعة
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# الصفحة الرئيسية
@app.route('/')
def home():
    return render_template('home.html', title="Home", custom_css='home')


# صفحة رفع الفيديو
@app.route('/upload', methods=['GET', 'POST'])
def upload_video():
    # if request.method == 'POST':
    #     file = request.files['video']
    #     if file:
    #         filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    #         file.save(filepath)
    #         return redirect(url_for('result', filename=file.filename))
    return render_template('upload.html', title="Upload Video", custom_css='upload')

# صفحة النتيجة
@app.route('/result')
def result():
    # filename = request.args.get('filename')
    # return render_template('result.html', title="Result", filename=filename)
    return render_template('result.html', title="Result", custom_css='result')

@app.route('/how-it-work')
def howItWork():
    return render_template('howItWork.html', title="How It Work", custom_css='how-it-work')

@app.route('/aboutus')
def aboutUs():
    return render_template('about.html', title="About Us", custom_css='about')

@app.route('/contact')
def contact():
    # filename = request.args.get('filename')
    # return render_template('result.html', title="Result", filename=filename)
    return render_template('contact.html', title="contact", custom_css='contact')


if __name__ == '__main__':
    app.run(debug=True)
