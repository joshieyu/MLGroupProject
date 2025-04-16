from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QSlider,
    QFileDialog, QMessageBox, QSizePolicy, QApplication
)
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput, QMediaDevices
from PyQt6.QtCore import QUrl, QTimer, Qt
from PyQt6.QtGui import QIcon
import os

import os
import tempfile
import torchaudio

def make_pcm16(src_path: str) -> str:
    waveform, sr = torchaudio.load(src_path)
    fd, out_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    torchaudio.save(out_path, waveform, sr, encoding="PCM_S", bits_per_sample=16)
    return out_path

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Audio Denoiser")
        self.setMinimumSize(520, 220)

        self.file_path = ""
        self.output_path = "denoised_output.wav"

        self.player = QMediaPlayer()

        # explicitly grab the default audio device
        default_audio_dev = QMediaDevices.defaultAudioOutput()
        self.audio_output = QAudioOutput(default_audio_dev)

        # volume is a float 0.0–1.0
        self.audio_output.setVolume(1)

        # wire up the player to it
        self.player.setAudioOutput(self.audio_output)

        # === Layouts ===
        layout = QVBoxLayout()
        top_row = QHBoxLayout()
        mid_row = QHBoxLayout()
        bottom_row = QHBoxLayout()

        # === Top Row ===
        self.select_btn = QPushButton("Select Audio")
        self.select_btn.clicked.connect(self.select_file)
        top_row.addWidget(self.select_btn)

        self.file_label = QLabel("No file selected")
        top_row.addWidget(self.file_label)

        # === Middle Row ===
        self.play_btn = QPushButton()
        self.play_btn.setIcon(QIcon.fromTheme("media-playback-start"))
        self.play_btn.clicked.connect(self.toggle_play)
        self.play_btn.setEnabled(False)
        mid_row.addWidget(self.play_btn)

        self.time_label = QLabel("0:00 / 0:00")
        self.time_label.setEnabled(False)
        mid_row.addWidget(self.time_label)

        self.progress = QSlider(Qt.Orientation.Horizontal)
        self.progress.sliderMoved.connect(self.seek_position)
        self.progress.setEnabled(False)
        mid_row.addWidget(self.progress)

        # === Bottom Row ===
        self.denoise_btn = QPushButton("Denoise")
        self.denoise_btn.setEnabled(False)
        self.denoise_btn.clicked.connect(self.denoise_audio)
        self.denoise_btn.setFixedHeight(40)
        self.denoise_btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        bottom_row.addWidget(self.denoise_btn)

        self.download_btn = QPushButton()
        self.download_btn.setIcon(QIcon.fromTheme("document-save"))
        self.download_btn.setIconSize(self.play_btn.iconSize())
        self.download_btn.setFixedSize(self.play_btn.sizeHint())
        self.download_btn.setEnabled(False)
        self.download_btn.clicked.connect(self.download_file)
        bottom_row.addWidget(self.download_btn)

        # === Status Label ===
        self.status_label = QLabel("")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setVisible(False)
        self.status_label.setObjectName("status")

        # Combine layouts
        layout.addLayout(top_row)
        layout.addLayout(mid_row)
        layout.addLayout(bottom_row)
        layout.addWidget(self.status_label)
        self.setLayout(layout)

        # === Media Events ===
        self.player.durationChanged.connect(self.set_duration)
        self.player.positionChanged.connect(self.update_ui)

        # === Timer for slider updates ===
        self.timer = QTimer()
        self.timer.setInterval(100)
        self.timer.timeout.connect(self.update_ui)
        self.timer.start()

        # === Dark Mode Stylesheet (with disabled states & status style) ===
        self.setStyleSheet("""
            QWidget {
                background-color: #1e1e1e;
                color: #e0e0e0;
                font-family: Arial, sans-serif;
            }
            QPushButton {
                background-color: #2e2e2e;
                border: none;
                padding: 8px 16px;
                border-radius: 16px;
                font-size: 14px;
                color: #e0e0e0;
            }
            QPushButton:hover {
                background-color: #3a3a3a;
            }
            QPushButton:disabled {
                background-color: #1a1a1a;
                color: #666666;
            }
            QSlider::groove:horizontal {
                height: 6px;
                background: #444;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #bbb;
                width: 12px;
                margin: -4px 0;
                border-radius: 6px;
            }
            QSlider::groove:disabled {
                background: #333;
            }
            QSlider::handle:disabled {
                background: #666;
            }
            QLabel {
                font-size: 13px;
            }
            QLabel:disabled {
                color: #666666;
            }
            QLabel#status {
                font-style: italic;
                color: #888888;
                padding-top: 6px;
            }
        """)

    def select_file(self):
        # Ask user
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Audio", os.path.expanduser("~/Desktop"), "Audio (*.wav *.mp3)"
        )
        if not path:
            return

        self.file_path = path
        # Transcode to 16‑bit
        self._temp_input = make_pcm16(path)
        self.set_media(self._temp_input)

        self.file_label.setText(os.path.basename(path))
        self.play_btn.setEnabled(True)
        self.progress.setEnabled(True)
        self.denoise_btn.setEnabled(True)
        self.time_label.setEnabled(True)
        self.progress.setValue(0)
        self.time_label.setText("0:00 / 0:00")

    def set_media(self, path):
        # Always point the player at a PCM16 file
        self.player.setSource(QUrl.fromLocalFile(path))
        self.play_btn.setIcon(QIcon.fromTheme("media-playback-start"))

    def toggle_play(self):
        if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.player.pause()
            self.play_btn.setIcon(QIcon.fromTheme("media-playback-start"))
        else:
            self.player.play()
            self.play_btn.setIcon(QIcon.fromTheme("media-playback-pause"))

    def set_duration(self, duration):
        self.progress.setRange(0, duration)

    def update_ui(self):
        pos = self.player.position()
        dur = self.player.duration()
        if dur > 0:
            self.progress.setValue(pos)
            self.time_label.setText(f"{self.ms_to_time(pos)} / {self.ms_to_time(dur)}")

    def seek_position(self, pos):
        self.player.setPosition(pos)

    def ms_to_time(self, ms):
        s = ms // 1000
        return f"{s//60}:{s%60:02}"

    def denoise_audio(self):
        from controller import denoise_audio  # This runs your model under the hood
        from ui_main import make_pcm16  # This converts the .wav to PCM16 for Qt

        self.status_label.setText("Denoising...")
        self.status_label.setVisible(True)
        QApplication.processEvents()

        # Call your model via controller.py (which saves to self.output_path)
        denoise_audio(self.file_path, self.output_path)

        # 🔧 Transcode the denoised output to 16-bit PCM so Qt can play it
        self._temp_output = make_pcm16(self.output_path)

        self.status_label.setVisible(False)
        QMessageBox.information(self, "Success", "Denoised successfully.")

        # Set this for playback
        self.set_media(self._temp_output)
        self.file_label.setText("(Denoised) " + os.path.basename(self.file_path))

        # Enable the download button
        self.download_btn.setEnabled(True)

    def download_file(self):
        if not os.path.exists(self.output_path):
            QMessageBox.warning(self, "No file", "You haven't denoised a file yet.")
            return

		# Extract original name, extension, and directory
        original_name = os.path.basename(self.file_path)
        name_no_ext, ext = os.path.splitext(original_name)
        suggested_name = f"(Denoised) {name_no_ext}{ext}"
        original_dir = os.path.dirname(self.file_path)
        default_path = os.path.join(original_dir, suggested_name)

		# Save dialog
        dest, _ = QFileDialog.getSaveFileName(self, "Save As", default_path)
        
        if dest:
            import shutil
            shutil.copy(self.output_path, dest)
            QMessageBox.information(self, "Saved", f"File saved to:\n{dest}")
