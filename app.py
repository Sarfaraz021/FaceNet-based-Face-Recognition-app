import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QFileDialog, QWidget
from PyQt5.QtGui import QPixmap, QImage, QPainter, QColor
from PyQt5.QtCore import Qt
import cv2
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch


class FaceMatchApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        self.setWindowTitle('Face Match App')
        self.setGeometry(100, 100, 800, 600)

        # Create widgets
        self.title_label = QLabel('AI Face Recongnition', self)
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setStyleSheet("font-size: 40px; font-weight: bold;")

        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setFixedSize(1600, 1200)  # border-radius:1000px
        self.image_label.setStyleSheet(
            "border-radius:1000px;border: 10px solid #6B728E;")

        self.result_label = QLabel(self)
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet(
            "font-size: 40px; font-weight: bold; color:#030637; padding-bottom:30px;")

        self.upload_button = QPushButton('Upload Image', self)
        self.upload_button.clicked.connect(self.uploadImage)
        self.upload_button.setFixedSize(400, 80)
        self.upload_button.setStyleSheet(
            "background-color: #030637; color: white; border-radius: 15px; font-size: 30px; font-weight: bold;")

        # Create layout
        title_layout = QVBoxLayout()
        title_layout.addWidget(self.title_label)

        button_layout = QHBoxLayout()
        button_layout.addStretch(1)
        button_layout.addWidget(self.upload_button)
        button_layout.addStretch(1)

        main_layout = QVBoxLayout()
        main_layout.addLayout(title_layout)
        main_layout.addWidget(self.image_label)
        main_layout.addWidget(self.result_label)
        main_layout.addLayout(button_layout)

        # Set central widget and layout
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        # Initialize face matching components
        self.mtcnn = MTCNN(image_size=240, margin=0, min_face_size=20)
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval()

    def uploadImage(self):
        # Open file dialog to select an image
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        filePath, _ = QFileDialog.getOpenFileName(
            self, "Open Image File", "", "Image Files (*.png *.jpg *.jpeg *.bmp);;All Files (*)", options=options)

        if filePath:
            # Perform face matching on the selected image
            result = self.face_match(filePath, 'data.pt')

            # Display the image and result in the UI
            self.displayImage(filePath)
            self.displayResult(result)

    def displayImage(self, img_path):
        # Display the selected image in the UI with a black border
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (1024, 1024))

        # Convert the image to a QImage
        height, width, channel = img.shape
        bytes_per_line = 3 * width
        q_image = QImage(img.data, width, height,
                         bytes_per_line, QImage.Format_RGB888)

        # Draw the border around the image
        painter = QPainter(q_image)
        painter.setPen(QColor(0, 0, 0))  # Black color for the border
        painter.drawRect(0, 0, width - 1, height - 1)
        painter.end()

        # Display the image with the border
        pixmap = QPixmap.fromImage(q_image)
        self.image_label.setPixmap(pixmap)

    def displayResult(self, result):
        # Display the face matching result in the UI
        text = f'Face matched with: {result[0]}'
        self.result_label.setText(text)

    def face_match(self, img_path, data_path):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (512, 512))

        face, prob = self.mtcnn(img, return_prob=True)

        # Check if a face is detected
        if face is None:
            result = ('No Face Detected', 0, 'No Match')
        else:
            emb = self.resnet(face.unsqueeze(0)).detach()

            saved_data = torch.load(data_path)
            embedding_list = saved_data[0]
            name_list = saved_data[1]
            dist_list = []

            for idx, emb_db in enumerate(embedding_list):
                dist = torch.dist(emb, emb_db).item()
                dist_list.append(dist)

            idx_min = dist_list.index(min(dist_list))

            threshold = 0.6
            if min(dist_list) < threshold:
                result = (name_list[idx_min], min(dist_list), 'Match')
            else:
                result = ('Unknown', min(dist_list), 'No Match')

        return result


if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWin = FaceMatchApp()
    mainWin.show()
    sys.exit(app.exec_())
