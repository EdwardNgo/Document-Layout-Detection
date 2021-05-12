from flask import Flask, render_template, request, redirect, url_for,abort,send_from_directory,flash
import logging
import os
from  werkzeug.utils import secure_filename
import imghdr
from infer import inference
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024
app.config['UPLOAD_EXTENSIONS'] = ['.jpg', '.png', '.gif']
app.config['UPLOAD_PATH'] = 'uploads'
app.config['ANNOTATION_PATH'] = 'annotations'

def validate_image(stream):
	header = stream.read(512)
	stream.seek(0)
	format = imghdr.what(None,header)
	if not format:
		return None
	return '.' + (format if format != 'jpeg' else 'jpg')

@app.route('/')
def index():
	files = os.listdir(app.config['UPLOAD_PATH'])
	return render_template('index.html',files=files)

@app.route('/', methods=['POST'])
def upload_files():
	uploaded_file = request.files['file']
	filename = secure_filename(uploaded_file.filename)
	try:
		os.mkdir(app.config['UPLOAD_PATH'])
	except:
		print("folder had already existed")
	if filename != '':
		file_ext = os.path.splitext(filename)[1]
		if file_ext not in app.config['UPLOAD_EXTENSIONS'] or file_ext != validate_image(uploaded_file.stream):
			flash('Not in right format')
			abort(400)
		dst = os.path.join(app.config['UPLOAD_PATH'], filename)
		uploaded_file.save(dst)
		inference(dst,app.config['ANNOTATION_PATH'])
		annotations = os.path.join(app.config['ANNOTATION_PATH'],filename)
		print(annotations)
		return render_template("upload.html",filename=filename,annotations=annotations)
		# flash('File(s) successfully uploaded')

	return redirect(url_for('index'))

@app.route('/uploads/<filename>')
def upload(filename):
	return send_from_directory(app.config['UPLOAD_PATH'],filename) 
@app.route('/annotations/<filename>')
def annotations(filename):
	return send_from_directory(app.config['ANNOTATION_PATH'],filename) 

if __name__ == "__main__":
	# set debug level
	logging.basicConfig(level=logging.DEBUG)
    
	# run app
	app.run(host='127.0.0.1', port=5000)