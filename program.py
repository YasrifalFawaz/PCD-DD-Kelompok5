import sys
import cv2
import numpy as np
import os
import glob
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from collections import Counter
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QFileDialog, QMessageBox, QDialog, QGridLayout, QScrollArea
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt

def cv_image_to_qpixmap(cv_image, target_size=(200, 200)):
    if cv_image is None or cv_image.size == 0:
        empty_pixmap = QPixmap(target_size[0], target_size[1])
        empty_pixmap.fill(Qt.lightGray)
        return empty_pixmap

    try:
        if len(cv_image.shape) == 2:
            h, w = cv_image.shape
            bytes_per_line = w
            qt_img = QImage(cv_image.data, w, h, bytes_per_line, QImage.Format_Grayscale8)
        elif len(cv_image.shape) == 3:
            image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            h, w, ch = image_rgb.shape
            bytes_per_line = ch * w
            qt_img = QImage(image_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        else:
            raise ValueError("Format gambar tidak didukung untuk konversi QPixmap")

        pixmap = QPixmap.fromImage(qt_img)
        return pixmap.scaled(target_size[0], target_size[1], Qt.KeepAspectRatio, Qt.SmoothTransformation)
    except Exception as e:
        print(f"Error converting CV image to QPixmap: {e}")
        empty_pixmap = QPixmap(target_size[0], target_size[1])
        empty_pixmap.fill(Qt.darkRed)
        return empty_pixmap

def segment_banana_raw_mask(image_hsv):
    lower_yellow_green = np.array([10, 60, 60])
    upper_orange_yellow = np.array([40, 255, 255])
    mask = cv2.inRange(image_hsv, lower_yellow_green, upper_orange_yellow)
    return mask

def preprocess_image_original_method(image_path):
    original_bgr = cv2.imread(image_path)
    if original_bgr is None:
        raise ValueError(f"Gagal membaca gambar dari path: {image_path}. File mungkin rusak atau bukan format gambar yang didukung.")

    resized_bgr = cv2.resize(original_bgr, (200, 200))

    hsv_image = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2HSV)

    raw_mask_before_morphology = segment_banana_raw_mask(hsv_image)

    kernel = np.ones((5, 5), np.uint8)
    mask_after_morphology = cv2.dilate(raw_mask_before_morphology, kernel, iterations=2)
    mask_after_morphology = cv2.erode(mask_after_morphology, kernel, iterations=2)

    masked_bgr_display = cv2.bitwise_and(resized_bgr, resized_bgr, mask=mask_after_morphology)

    h_channel, s_channel, v_channel = cv2.split(hsv_image)
    features = None
    if np.any(mask_after_morphology > 0):
        mean_h = np.mean(h_channel[mask_after_morphology > 0])
        mean_s = np.mean(s_channel[mask_after_morphology > 0])
        mean_v = np.mean(v_channel[mask_after_morphology > 0])
        features = [mean_h, mean_s, mean_v]

    processing_steps_data = {
        'original_image': original_bgr,
        'resized_image': resized_bgr,
        'hsv_converted': hsv_image,
        'raw_mask_hsv': raw_mask_before_morphology,
        'mask_after_morphology': mask_after_morphology,
        'masked_output': masked_bgr_display,
        'extracted_features': features
    }

    return features, resized_bgr, masked_bgr_display, processing_steps_data

def load_training_data_original_method(base_folder):
    features_list = []
    labels_list = []
    for label in ['belum_matang', 'matang', 'sangat_matang']:
        path_pattern = os.path.join(base_folder, label, '*.jpg')
        image_files = glob.glob(path_pattern)
        if not image_files:
            print(f"Peringatan: Tidak ada file *.jpg ditemukan di {os.path.join(base_folder, label)}")
            continue

        for file_path in image_files:
            try:
                feat, _, _, _ = preprocess_image_original_method(file_path)
                if feat is not None:
                    features_list.append(feat)
                    labels_list.append(label)
                else:
                    print(f"Peringatan: Gagal mengekstrak fitur dari {file_path} saat training.")
            except Exception as e:
                print(f"Error memproses file {file_path} saat training: {e}")

    return np.array(features_list), np.array(labels_list)

class PreprocessingStepsWindow(QDialog):
    def __init__(self, processing_steps_dict, parent=None):
        super().__init__(parent)
        self.processing_steps = processing_steps_dict
        self.setWindowTitle("Tahapan Preprocessing Citra Pisang")
        self.setGeometry(150, 150, 900, 700)
        self.setup_ui()

    def setup_ui(self):
        main_layout = QVBoxLayout(self)

        scroll = QScrollArea(self)
        scroll_widget = QWidget()
        scroll_layout = QGridLayout(scroll_widget)

        steps_info = [
            ('original_image', '1. Citra Asli'),
            ('resized_image', '2. Citra Resize (200x200)'),
            ('hsv_converted', '3. Konversi ke HSV'),
            ('raw_mask_hsv', '4. Masking HSV Awal'),
            ('mask_after_morphology', '5. Masking (Dilasi & Erosi)'),
            ('masked_output', '6. Citra Hasil Masking Final'),
        ]

        row, col = 0, 0
        for key, title_text in steps_info:
            if key in self.processing_steps:
                img_data = self.processing_steps[key]

                title_label = QLabel(title_text)
                title_label.setAlignment(Qt.AlignCenter)
                title_label.setStyleSheet("font-weight: bold; margin-bottom: 5px;")

                img_label = QLabel()
                img_label.setFixedSize(200, 200)
                img_label.setAlignment(Qt.AlignCenter)
                img_label.setStyleSheet("border: 1px solid grey;")

                pixmap = None
                if key == 'hsv_converted':
                    hsv_display = img_data.copy()
                    pixmap = cv_image_to_qpixmap(cv2.cvtColor(hsv_display, cv2.COLOR_HSV2BGR), (200,200))
                else:
                    pixmap = cv_image_to_qpixmap(img_data, (200,200))

                img_label.setPixmap(pixmap)

                step_v_layout = QVBoxLayout()
                step_v_layout.addWidget(title_label)
                step_v_layout.addWidget(img_label)

                scroll_layout.addLayout(step_v_layout, row, col)

                col += 1
                if col >= 3:
                    col = 0
                    row += 1
        
        scroll_widget.setLayout(scroll_layout)
        scroll.setWidgetResizable(True)
        scroll.setWidget(scroll_widget)
        main_layout.addWidget(scroll)

        close_button = QPushButton("Tutup")
        close_button.clicked.connect(self.accept)
        main_layout.addWidget(close_button, alignment=Qt.AlignCenter)

        self.setLayout(main_layout)

class EnhancedBananaRipenessApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Klasifikasi Kematangan Pisang (Metode Asli Enhanced GUI)")
        self.setGeometry(100, 100, 750, 650)

        self.current_processing_steps_dict = None
        self.y_train = None
        self.category_hsv_means = {}
        self.setup_ui()

        self.model_ready = False
        self.train_model()

    def setup_ui(self):
        main_layout = QVBoxLayout()

        app_title_label = QLabel("Klasifikasi Kematangan Pisang (Segmentasi HSV)")
        app_title_label.setAlignment(Qt.AlignCenter)
        app_title_label.setStyleSheet("font-size: 18px; font-weight: bold; margin: 10px;")
        main_layout.addWidget(app_title_label)

        image_display_layout = QHBoxLayout()

        original_v_layout = QVBoxLayout()
        self.title_original = QLabel("Citra Asli (Input):")
        self.title_original.setAlignment(Qt.AlignCenter)
        self.label_image_original = QLabel("Pilih gambar untuk ditampilkan")
        self.label_image_original.setFixedSize(250, 250)
        self.label_image_original.setAlignment(Qt.AlignCenter)
        self.label_image_original.setStyleSheet("border: 1px solid grey;")
        original_v_layout.addWidget(self.title_original)
        original_v_layout.addWidget(self.label_image_original)
        image_display_layout.addLayout(original_v_layout)

        masked_v_layout = QVBoxLayout()
        self.title_masked = QLabel("Citra Hasil Masking:")
        self.title_masked.setAlignment(Qt.AlignCenter)
        self.label_image_masked = QLabel("Hasil masking akan ditampilkan di sini")
        self.label_image_masked.setFixedSize(250, 250)
        self.label_image_masked.setAlignment(Qt.AlignCenter)
        self.label_image_masked.setStyleSheet("border: 1px solid grey;")
        masked_v_layout.addWidget(self.title_masked)
        masked_v_layout.addWidget(self.label_image_masked)
        image_display_layout.addLayout(masked_v_layout)
        main_layout.addLayout(image_display_layout)

        self.label_prediction = QLabel("Prediksi K-NN: -")
        self.label_prediction.setAlignment(Qt.AlignCenter)
        self.label_prediction.setStyleSheet("font-size: 16px; font-weight: bold; margin-top: 15px;")
        main_layout.addWidget(self.label_prediction)

        self.label_features = QLabel("Fitur HSV Citra Uji: H= -, S= -, V= -")
        self.label_features.setAlignment(Qt.AlignCenter)
        self.label_features.setStyleSheet("font-size: 12px;")
        main_layout.addWidget(self.label_features)

        self.label_hsv_similarity = QLabel("Kemiripan dengan Rata-rata HSV Kategori: -")
        self.label_hsv_similarity.setAlignment(Qt.AlignCenter)
        self.label_hsv_similarity.setStyleSheet("font-size: 12px; margin-top: 5px; margin-bottom:10px;")
        main_layout.addWidget(self.label_hsv_similarity)


        button_layout = QHBoxLayout()
        self.button_select = QPushButton("Pilih Gambar Pisang")
        self.button_select.clicked.connect(self.load_image)
        button_layout.addWidget(self.button_select)

        self.button_show_steps = QPushButton("Tampilkan Tahapan Proses")
        self.button_show_steps.clicked.connect(self.show_processing_steps_window)
        self.button_show_steps.setEnabled(False)
        button_layout.addWidget(self.button_show_steps)
        main_layout.addLayout(button_layout)

        self.setLayout(main_layout)


    def train_model(self):
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            train_folder = os.path.join(script_dir, 'train')

            if not os.path.isdir(train_folder):
                QMessageBox.critical(self, "Error Data Latih", f"Folder data latih 'train' tidak ditemukan di:\n{train_folder}\n" "Pastikan folder 'train' berisi subfolder 'belum_matang', 'matang', dan 'sangat_matang' dengan gambar *.jpg.")
                self.model_ready = False
                return

            X_train_features, y_train_labels = load_training_data_original_method(train_folder)
            self.y_train = y_train_labels

            if X_train_features.size == 0 or y_train_labels.size == 0:
                QMessageBox.critical(self, "Error Data Latih",
                                     "Tidak ada data latih yang berhasil dimuat. "
                                     "Pastikan folder 'train' dan subfoldernya berisi gambar *.jpg yang valid.")
                self.model_ready = False
                self.y_train = None
                self.category_hsv_means = {}
                return

            self.scaler = StandardScaler() #Inisialisasi K-NN
            X_scaled = self.scaler.fit_transform(X_train_features) # Scale data dilatih
            self.knn = KNeighborsClassifier(n_neighbors=5)  # Menentukan K
            self.knn.fit(X_scaled, y_train_labels) # Latih model K-NN
            self.model_ready = True

            self.category_hsv_means = {}
            if X_train_features.shape[0] > 0:
                unique_labels_for_hsv_avg = np.unique(y_train_labels)
                for label in unique_labels_for_hsv_avg:
                    class_specific_features = X_train_features[y_train_labels == label]
                    if class_specific_features.shape[0] > 0:
                        self.category_hsv_means[label] = np.mean(class_specific_features, axis=0)
                    else:
                        print(f"Peringatan: Tidak ada fitur untuk kelas {label} saat menghitung rata-rata HSV.")

        except Exception as e:
            QMessageBox.critical(self, "Error Pelatihan Model", f"Gagal melatih model: {e}")
            self.model_ready = False
            self.y_train = None
            self.category_hsv_means = {}


    def load_image(self):
        if not self.model_ready:
            QMessageBox.warning(self, "Model Belum Siap", "Model K-NN belum berhasil dilatih. Silakan periksa konsol untuk detail error data latih.")
            return

        file_path, _ = QFileDialog.getOpenFileName(self, "Pilih Gambar Pisang", "",
                                                   "Images (*.png *.jpg *.jpeg *.bmp)")
        if file_path:
            try:
                features, original_resized_bgr, masked_bgr_display, processing_steps_data = preprocess_image_original_method(file_path)

                self.current_processing_steps_dict = processing_steps_data
                self.button_show_steps.setEnabled(True)

                pixmap_original = cv_image_to_qpixmap(original_resized_bgr, (250, 250))
                self.label_image_original.setPixmap(pixmap_original)

                pixmap_masked = cv_image_to_qpixmap(masked_bgr_display, (250, 250))
                self.label_image_masked.setPixmap(pixmap_masked)

                if features is None:
                    QMessageBox.warning(self, "Error Fitur",
                                        "Tidak dapat mengekstrak fitur dari gambar (kemungkinan pisang tidak terdeteksi dengan baik). Prediksi tidak dapat dilakukan.")
                    self.label_prediction.setText("Prediksi K-NN: Gagal deteksi")
                    self.label_features.setText("Fitur HSV Citra Uji: H= -, S= -, V= -")
                    self.label_hsv_similarity.setText("Kemiripan dengan Rata-rata HSV Kategori: -")
                    return

                self.label_features.setText(f"Fitur HSV Citra Uji: H={features[0]:.2f}, S={features[1]:.2f}, V={features[2]:.2f}")

                scaled_feat = self.scaler.transform([features]) # Scale data di uji
                prediction = self.knn.predict(scaled_feat)[0] # Prediksi data
                prediction_text = prediction.replace("_", " ").title()
                self.label_prediction.setText(f"Prediksi K-NN: {prediction_text}") # Menampilkan prediksi

                if self.category_hsv_means:
                    test_hsv_features = np.array(features)
                    inverse_distances = {}
                    total_inverse_distance = 0.0

                    available_classes_for_hsv_sim = [cls for cls in self.knn.classes_ if cls in self.category_hsv_means]

                    for category_label in available_classes_for_hsv_sim:
                        mean_hsv_cat = self.category_hsv_means[category_label]
                        dist = np.linalg.norm(test_hsv_features - mean_hsv_cat)

                        inv_dist = 1.0 / (1.0 + dist)
                        inverse_distances[category_label] = inv_dist
                        total_inverse_distance += inv_dist

                    similarity_texts = ["Kemiripan dengan Rata-rata HSV Kategori:"]
                    if total_inverse_distance > 1e-9:
                        for category_label in self.knn.classes_:
                            if category_label in inverse_distances:
                                percentage = (inverse_distances[category_label] / total_inverse_distance) * 100
                                formatted_class_name = category_label.replace('_', ' ').title()
                                similarity_texts.append(f"  - {formatted_class_name}: {percentage:.2f}%")
                            else:
                                formatted_class_name = category_label.replace('_', ' ').title()
                                similarity_texts.append(f"  - {formatted_class_name}: N/A (rata-rata tidak ada)")

                    else:
                            for category_label in self.knn.classes_:
                                formatted_class_name = category_label.replace('_', ' ').title()
                                similarity_texts.append(f"  - {formatted_class_name}: N/A (jarak tidak terdefinisi)")

                    self.label_hsv_similarity.setText("\n".join(similarity_texts))
                else:
                    self.label_hsv_similarity.setText("Kemiripan dengan Rata-rata HSV Kategori: Data rata-rata tidak tersedia.")

            except ValueError as ve:
                QMessageBox.critical(self, "Error Baca Gambar", str(ve))
                self.reset_display_on_error()
            except Exception as e:
                QMessageBox.critical(self, "Error Proses Gambar", f"Terjadi kesalahan saat memproses gambar: {e}")
                self.reset_display_on_error()

    def show_processing_steps_window(self):
        if self.current_processing_steps_dict:
            steps_win = PreprocessingStepsWindow(self.current_processing_steps_dict, self)
            steps_win.exec_()
        else:
            QMessageBox.information(self, "Info", "Tidak ada data proses untuk ditampilkan. Silakan pilih dan proses gambar terlebih dahulu.")

    def reset_display_on_error(self):
        self.label_image_original.setPixmap(cv_image_to_qpixmap(None, (250,250)))
        self.label_image_masked.setPixmap(cv_image_to_qpixmap(None, (250,250)))
        self.label_image_original.setText("Gagal memuat gambar")
        self.label_image_masked.setText("Error proses gambar")
        self.label_prediction.setText("Prediksi K-NN: Error")
        self.label_features.setText("Fitur HSV Citra Uji: H= -, S= -, V= -")
        self.label_hsv_similarity.setText("Kemiripan dengan Rata-rata HSV Kategori: -")
        self.button_show_steps.setEnabled(False)
        self.current_processing_steps_dict = None

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = EnhancedBananaRipenessApp()
    window.show()
    sys.exit(app.exec_())