from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QLineEdit, QPushButton, QMessageBox, QFrame)
from api.security import APIKeyManager

class ApiKeyMixin:
    """Mixin to add API Key management to the MainWindow."""
    
    def create_api_key_page(self):
        """Creates the API Key management page."""
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(40, 40, 40, 40)
        layout.setSpacing(20)
        
        # Header
        header = QLabel("REST API Access Management")
        header.setStyleSheet("font-size: 24px; font-weight: bold; color: #00f2fe;")
        layout.addWidget(header)
        
        desc = QLabel("Generate and manage API keys to access DeVana's headless optimization engines programmatically.")
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #b0b0b0; font-size: 14px;")
        layout.addWidget(desc)
        
        # Key Display Area
        frame = QFrame()
        frame.setStyleSheet("background-color: #1e1e26; border-radius: 10px; border: 1px solid #3e3e4a;")
        frame_layout = QVBoxLayout(frame)
        
        self.key_display = QLineEdit()
        self.key_display.setPlaceholderText("No API Key generated yet...")
        self.key_display.setReadOnly(True)
        self.key_display.setStyleSheet("""
            QLineEdit {
                background-color: #0f0f14;
                color: #ffffff;
                border: none;
                padding: 15px;
                font-family: 'Courier New';
                font-size: 16px;
            }
        """)
        frame_layout.addWidget(self.key_display)
        layout.addWidget(frame)
        
        # Buttons
        btn_layout = QHBoxLayout()
        
        self.gen_btn = QPushButton("Generate New Key")
        self.gen_btn.setStyleSheet("""
            QPushButton {
                background-color: #00f2fe;
                color: #000000;
                font-weight: bold;
                padding: 12px 24px;
                border-radius: 5px;
            }
            QPushButton:hover { background-color: #4facfe; }
        """)
        self.gen_btn.clicked.connect(self.handle_generate_key)
        
        self.copy_btn = QPushButton("Copy to Clipboard")
        self.copy_btn.setStyleSheet("""
            QPushButton {
                background-color: #3e3e4a;
                color: #ffffff;
                padding: 12px 24px;
                border-radius: 5px;
            }
            QPushButton:hover { background-color: #4e4e5a; }
        """)
        self.copy_btn.clicked.connect(self.handle_copy_key)
        
        btn_layout.addWidget(self.gen_btn)
        btn_layout.addWidget(self.copy_btn)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)
        
        # Security Note
        note = QLabel("<b>Security Note:</b> Never share your API key. It grants full access to your local machine's computational resources via the REST server.")
        note.setStyleSheet("color: #ff4b2b; font-size: 12px;")
        layout.addWidget(note)
        
        layout.addStretch()
        
        # Load existing key if any
        keys = APIKeyManager.load_keys()
        if "default_user" in keys:
            self.key_display.setText(keys["default_user"])
            
        return page

    def handle_generate_key(self):
        """Generates and saves a new key."""
        new_key = APIKeyManager.generate_key()
        APIKeyManager.save_key("default_user", new_key)
        self.key_display.setText(new_key)
        QMessageBox.information(self, "Success", "New API Key generated and saved successfully.")

    def handle_copy_key(self):
        """Copies the key to the clipboard."""
        key = self.key_display.text()
        if key:
            from PyQt5.QtWidgets import QApplication
            QApplication.clipboard().setText(key)
            QMessageBox.information(self, "Copied", "API Key copied to clipboard.")
        else:
            QMessageBox.warning(self, "Error", "No key to copy!")
