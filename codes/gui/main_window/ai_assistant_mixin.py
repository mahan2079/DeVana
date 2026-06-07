from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QTextEdit, 
    QLineEdit, QPushButton, QComboBox, QGroupBox, QFormLayout,
    QTabWidget, QProgressBar,
    QMessageBox
)
from PyQt5.QtCore import QSettings, QThread, pyqtSignal
from PyQt5.QtGui import QFont

from workers.AIWorker import AIWorker, GENAI_AVAILABLE, RAGHelper

class AIAssistantMixin:
    """
    Mixin for the DeVana AI CoPilot (Assistant B).
    Handles chat interface, state bridge, and model settings.
    """
    def create_ai_assistant_drawer(self):
        """Create the sliding drawer or side panel for the AI assistant"""
        self.ai_panel = QWidget()
        self.ai_panel.setObjectName("ai-assistant-panel")
        self.ai_panel.setFixedWidth(400)
        
        # Ensure the panel has an opaque background
        self.ai_panel.setStyleSheet("""
            QWidget#ai-assistant-panel {
                background-color: #2D2D3A;
                border-left: 1px solid #3F3F4F;
            }
        """)
        self.settings = QSettings("DeVana", "AI_Assistant")
        
        panel_layout = QVBoxLayout(self.ai_panel)
        panel_layout.setContentsMargins(10, 10, 10, 10)
        panel_layout.setSpacing(10)

        # 1. Header
        header = QLabel("DeVana CoPilot")
        header.setFont(QFont("Segoe UI", 16, QFont.Bold))
        header.setStyleSheet("color: #00BFA5;")
        panel_layout.addWidget(header)

        # 2. Main Tabs (Chat vs Settings)
        self.ai_tabs = QTabWidget()
        
        # --- CHAT TAB ---
        chat_tab = QWidget()
        chat_layout = QVBoxLayout(chat_tab)
        chat_layout.setContentsMargins(0, 5, 0, 0)

        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setPlaceholderText("Welcome to the DVA CoPilot. How can I assist with your design today?")
        self.chat_display.setStyleSheet("background-color: #1A1A24; border-radius: 5px; padding: 5px;")
        chat_layout.addWidget(self.chat_display, 5)

        self.thinking_progress = QProgressBar()
        self.thinking_progress.setRange(0, 0) # Indeterminate
        self.thinking_progress.setVisible(False)
        chat_layout.addWidget(self.thinking_progress)

        input_container = QWidget()
        input_layout = QVBoxLayout(input_container)
        input_layout.setContentsMargins(0, 5, 0, 0)
        
        self.chat_input = QTextEdit()
        self.chat_input.setMaximumHeight(80)
        self.chat_input.setPlaceholderText("Ask about your system...")
        input_layout.addWidget(self.chat_input)

        btn_layout = QHBoxLayout()
        self.send_btn = QPushButton("Send Prompt")
        self.send_btn.setObjectName("primary-button")
        self.send_btn.clicked.connect(self.send_ai_query)
        
        self.clear_chat_btn = QPushButton("Clear")
        self.clear_chat_btn.clicked.connect(lambda: self.chat_display.clear())
        
        btn_layout.addWidget(self.send_btn)
        btn_layout.addWidget(self.clear_chat_btn)
        input_layout.addLayout(btn_layout)
        
        chat_layout.addWidget(input_container, 1)
        self.ai_tabs.addTab(chat_tab, "Assistant")

        # --- SETTINGS TAB (API KEYS) ---
        settings_tab = QWidget()
        settings_layout = QVBoxLayout(settings_tab)
        
        key_group = QGroupBox("Model Management")
        key_form = QFormLayout(key_group)
        
        self.ai_provider_combo = QComboBox()
        self.ai_provider_combo.addItems(["Google Gemini", "OpenAI (Future)", "Local (Ollama)"])
        key_form.addRow("Provider:", self.ai_provider_combo)

        self.api_key_input = QLineEdit()
        self.api_key_input.setEchoMode(QLineEdit.Password)
        # Load existing key from QSettings
        self.api_key_input.setText(self.settings.value("gemini_api_key", ""))
        key_form.addRow("API Key:", self.api_key_input)

        self.model_combo = QComboBox()
        self.model_combo.setEditable(True)
        # Add default models
        self.model_combo.addItems(["gemini-1.5-pro", "gemini-1.5-flash"])
        saved_model = self.settings.value("gemini_model", "gemini-1.5-pro")
        if self.model_combo.findText(saved_model) == -1:
            self.model_combo.addItem(saved_model)
        self.model_combo.setCurrentText(saved_model)
        key_form.addRow("Select Model:", self.model_combo)

        self.fetch_models_btn = QPushButton("Fetch Available Models")
        self.fetch_models_btn.clicked.connect(self.fetch_gemini_models)
        key_form.addRow(self.fetch_models_btn)

        save_keys_btn = QPushButton("Save Settings")
        save_keys_btn.clicked.connect(self.save_ai_settings)
        key_form.addRow(save_keys_btn)

        settings_layout.addWidget(key_group)
        settings_layout.addStretch()
        
        self.ai_tabs.addTab(settings_tab, "Settings")

        panel_layout.addWidget(self.ai_tabs)
        
        # Hidden by default, added to MainWindow's main layout later
        self.ai_panel.hide()
        return self.ai_panel

    def fetch_gemini_models(self):
        """Query the Gemini API to list available models for the given API key."""
        api_key = self.api_key_input.text().strip()
        if not api_key:
            QMessageBox.warning(self, "API Key Required", "Please enter an API Key first.")
            return

        if not GENAI_AVAILABLE:
            QMessageBox.critical(self, "Error", "New Google GenAI library not installed.")
            return

        try:
            from google import genai
            client = genai.Client(api_key=api_key)
            
            self.fetch_models_btn.setEnabled(False)
            self.fetch_models_btn.setText("Fetching...")
            
            # Use a QThread for this to avoid freezing UI
            class ModelFetcher(QThread):
                finished = pyqtSignal(list)
                error = pyqtSignal(str)
                def run(self):
                    try:
                        models = []
                        # New SDK method to list models
                        for m in client.models.list():
                            if m.supported_methods and 'generateContent' in m.supported_methods:
                                models.append(m.name)
                        self.finished.emit(models)
                    except Exception as e:
                        self.error.emit(str(e))

            self.model_fetcher = ModelFetcher()
            self.model_fetcher.finished.connect(self.update_model_list)
            self.model_fetcher.error.connect(lambda err: QMessageBox.warning(self, "Fetch Error", err))
            self.model_fetcher.finished.connect(lambda: self.fetch_models_btn.setEnabled(True))
            self.model_fetcher.finished.connect(lambda: self.fetch_models_btn.setText("Fetch Available Models"))
            self.model_fetcher.start()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to initialize fetch: {str(e)}")

    def update_model_list(self, models):
        if not models:
            return
        current = self.model_combo.currentText()
        self.model_combo.clear()
        self.model_combo.addItems(models)
        if current in models:
            self.model_combo.setCurrentText(current)
        QMessageBox.information(self, "Success", f"Retrieved {len(models)} models from API.")

    def save_ai_settings(self):
        self.settings.setValue("gemini_api_key", self.api_key_input.text())
        self.settings.setValue("gemini_model", self.model_combo.currentText())
        QMessageBox.information(self, "AI Settings", "Settings saved successfully.")

    def collect_system_state(self) -> Dict[str, Any]:
        """The 'State Bridge': Captures current UI values for the AI context."""
        state = {
            "main_system": {
                "mu": self.mu_box.value(),
                "landa": [b.value() for b in self.landa_boxes],
                "nu": [b.value() for b in self.nu_boxes],
                "f1": self.f_1_box.value(),
                "f2": self.f_2_box.value()
            },
            "dva_current": {
                "beta": [b.value() for b in self.beta_boxes],
                "lambda": [b.value() for b in self.lambda_boxes],
                "mu": [b.value() for b in self.mu_dva_boxes],
                "nu": [b.value() for b in self.nu_dva_boxes]
            }
        }
        return state

    def send_ai_query(self):
        query = self.chat_input.toPlainText().strip()
        if not query:
            return

        api_key = self.api_key_input.text()
        model_name = self.model_combo.currentText()
        
        # 1. Get documentation context (RAG)
        doc_context = RAGHelper.get_relevant_docs(query)
        
        # 2. Build System Prompt
        system_prompt = f"""You are the DeVana CoPilot, an expert in mechanical vibration optimization.
Your goal is to help the user design Dynamic Vibration Absorbers (DVAs).

RULES:
- Be technically precise.
- Use LaTeX for math.
- If suggesting parameter changes, reference the provided documentation.
- Do not hallucinate capabilities not present in the docs.

KNOWLEDGE CONTEXT:
{doc_context}
"""
        
        # 3. Get UI State
        context_data = self.collect_system_state()

        # 4. Start Worker
        self.chat_display.append(f"\n<b>You:</b> {query}")
        self.chat_input.clear()
        
        self.ai_worker = AIWorker(api_key, model_name, system_prompt, query, context_data)
        self.ai_worker.thinking_started.connect(lambda: self.thinking_progress.setVisible(True))
        self.ai_worker.response_received.connect(self.handle_ai_response)
        self.ai_worker.error_occurred.connect(lambda msg: QMessageBox.warning(self, "AI Error", msg))
        self.ai_worker.finished.connect(lambda: self.thinking_progress.setVisible(False))
        self.ai_worker.start()

    def handle_ai_response(self, text):
        self.chat_display.append(f"\n<b>CoPilot:</b>\n{text}")
        # Auto-scroll
        self.chat_display.verticalScrollBar().setValue(self.chat_display.verticalScrollBar().maximum())
