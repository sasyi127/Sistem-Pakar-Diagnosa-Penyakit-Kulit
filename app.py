# coding=utf-8
import os
import numpy as np

# Keras
from keras.models import load_model
from keras.utils import img_to_array
from keras.utils import load_img

# Flask utils
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
# from gevent.pywsgi import WSGIServer

# Mendefinisikan App Flask
app = Flask(__name__)

MODEL_PATH = 'models/SBPFixed_Model.h5'

# Memuat model yang sudah disimpan
model = load_model(MODEL_PATH)

# Membuat Fungsi Prediksi penyakit kulit


def model_predict(img_path, model):
    img = load_img(img_path, target_size=(150, 150))

    # Preprocessing the image
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)

    preds = model.predict(x)
    preds = np.argmax(preds, axis=1)
    if preds == 0:
        preds = """
        <h2>Ini adalah Cacar Air</h2>
        <h5>Cara Mengatasi Cacar Air</h5>
        <ul>
            <li>Konsumsi obat penghilang rasa sakit untuk membantu mengurangi <br> 
            demam tinggi dan rasa sakit ketika seseorang menderita cacar air.</li>
            <li>Minum banyak cairan untuk mencegah dehidrasi, <br>
             yang dapat menjadi komplikasi cacar air.</li>
            <li>Hindari makanan asin atau pedas</li>
            <li>Untuk menghindari gatal bisa menjadi parah, dengan memakai salep.</li>
            <li>Jangan menggaruk luka dan menjaga kuku tetap bersih</li>
        </ul>
        """
    elif preds == 1:
        preds = """
        <h2>Ini adalah Herpes</h2>
        <h5>Cara Mengatasi Herpes</h5>
        <ul>
            <li>Kompres menggunakan air hangat atau dingin pada bagian yang <br>
            sering muncul herpes untuk meredakan rasa sakit</li>
            <li>Aplikasikan tumbukan halus bawang putih dan minyak zaitun <br> 
            pada bagian tubuh yang terdampak virus herpes tiga kali sehari </li>
            <li>Oleskan Cuka Apel ke bagian tubuh yang terdampak virus. <br> 
            Cuka apel memiliki komponen anti inflamasi yang bisa membuat luka cepat kering</li>
            <li>Mengonsumsi suplemen seperti yogurt, vitamin B dan zinc dengan <br>
             takaran 30 mg per hari untuk mengatasi penyebaran virus </li>
            <li>Mengatur pola makan yang baik untuk mencegah penurunan daya tahan tubuh</li>
        </ul>
        """
    elif preds == 2:
        preds = """
        <h2>ini adalah Impetigo</h2>
        <h5>Cara Mengatasi Impetigo</h5>
        <ul>
            <li>Merendam luka dengan menggunakan air hangat</li>
            <li>Gunakan salep atau krim antibiotik</li>
            <li>Meminum obat seperti clindamycin atau obat antibiotik golongan sefalosporin</li>
        </ul>
        """
    elif preds == 3:
        preds = """
        <h2>Ini adalah Kurap</h2>
        <h5>Cara Mengatasi Kurap</h5>
        <ul>
            <li>Cuci sprei dan pakaian setiap hari untuk membantu membunuh jamur-jamur</li>
            <li>Keringkan area tubuh secara menyeluruh setelah mandi</li>
            <li>Gunakan pakaian longgar di daerah yang terkena kurap</li>
            <li>Obati semua area yang terinfeksi dengan produk yang mengandung <br> 
            clotrimazole, miconazole, terbinafine, atau bahan terkait lainnya</li>
        </ul>
        """
    elif preds == 4:
        preds = """
        <h2>Ini adalah Kutil</h2>
        <h5>Cara Mengatasi Kutil</h5>
        <ul>
            <li>Perawatan dengan nitrogen cair/cryotherapy</li>
            <li>Operasi pembedahan</li>
            <li>Perawatan laser</li>
        </ul>
        """
    elif preds == 5:
        preds = """
        <h2>Ini adalah Melanoma</h2>
        <h5>Cara Mengatasi Melanoma</h5>
        <ul>
            <li>Operasi atau pembedahan jadi pengobatan</li>
            <li>Terapi radiasi</li>
            <li>Kemoterapi</li>
        </ul>
        """
    elif preds == 6:
        preds = """
        <h2>Ini adalah Psoriasis</h2>
        <h5>Cara Mengatasi Psoriasis</h5>
        <ul>
            <li>Mengenal dan menjauhi faktor pemicu gejala psoriasis</li>
            <li>Membatasi waktu mandi</li>
            <li>Mengoleskan pelembap pada kulit</li>
            <li>Menjalani pola makan sehat</li>
            <li>Mengelola stres dengan baik</li>
            <li>Menggunakan bahan alami</li>
        </ul>
        """
    elif preds == 7:
        preds = """
        <h2>Ini adalah Vitiligo</h2>
        <h5>Cara Mengatasi Vitiligo</h5>
        <ul>
            <li>Obat yang mengontrol peradangan.</li>
            <li>Pengobatan yang mempengaruhi sistem kekebalan.</li>
            <li>Terapi cahaya seperti Fototerapi dengan ultraviolet B pita sempit (UVB)</li>
            <li>Operasi Cangkok kulit.</li>
            <li>Transplantasi suspensi seluler.</li>
        </ul>
        """

    return preds


@app.route('/', methods=['GET'])
def index():
    # Halaman Utama
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Mendapatkan file dari permintaan post
        f = request.files['file']

        # menyimpan file yang di upload ke folder images
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'images', secure_filename(f.filename))
        f.save(file_path)

        # Membuat prediksi
        preds = model_predict(file_path, model)
        result = preds
        return result
    return None


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8090)
