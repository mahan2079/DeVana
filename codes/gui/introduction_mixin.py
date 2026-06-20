from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

class IntroductionMixin:
    def create_introduction_page(self):
        """Create a vertically optimized, visually striking introduction page"""
        self.intro_page = QWidget()
        # Use a container widget to limit width
        container = QWidget()
        container.setFixedWidth(650) # Strict width limit to prevent horizontal scroll
        
        main_vbox = QVBoxLayout(self.intro_page)
        main_vbox.setAlignment(Qt.AlignHCenter)
        main_vbox.addWidget(container)
        
        self.intro_layout = QVBoxLayout(container)
        self.intro_layout.setContentsMargins(20, 20, 20, 20)
        self.intro_layout.setSpacing(15)

        # --- HEADER SECTION ---
        header_widget = QWidget()
        header_layout = QVBoxLayout(header_widget)
        header_layout.setContentsMargins(0, 0, 0, 5)
        
        title = QLabel("DeVana")
        title.setFont(QFont("Segoe UI", 28, QFont.Bold))
        title.setStyleSheet("color: #00BFA5; margin-bottom: -5px;") 
        title.setAlignment(Qt.AlignCenter)
        header_layout.addWidget(title)
        
        subtitle = QLabel("Algorithmic Dynamic Vibration Absorber Design")
        subtitle.setFont(QFont("Segoe UI", 12, QFont.Weight.Medium))
        subtitle.setStyleSheet("color: #1976D2;")
        subtitle.setAlignment(Qt.AlignCenter)
        header_layout.addWidget(subtitle)

        self.mission_label = QLabel(
            "Bridging the gap between engineering intuition and data-driven optimization."
        )
        self.mission_label.setFont(QFont("Segoe UI", 10, QFont.StyleItalic))
        self.mission_label.setAlignment(Qt.AlignCenter)
        self.mission_label.setWordWrap(True)
        header_layout.addWidget(self.mission_label)
        
        self.intro_layout.addWidget(header_widget)

        # --- WORKFLOW SECTION (Vertical) ---
        flow_group = QGroupBox("The Algorithmic Workflow")
        flow_group.setObjectName("workflow-group")
        flow_group.setFixedWidth(580)
        flow_inner = QVBoxLayout(flow_group)
        flow_inner.setContentsMargins(30, 25, 30, 15)
        flow_inner.setSpacing(5)

        def create_step_mini(text, color="#1976D2", is_future=False):
            frame = QFrame()
            frame.setFixedHeight(40)
            if is_future:
                frame.setStyleSheet("background-color: transparent; border: 1.5px dashed #757575; border-radius: 6px;")
            else:
                frame.setStyleSheet(f"background-color: {color}; border-radius: 6px; border: none;")
            
            l = QHBoxLayout(frame)
            l.setContentsMargins(10, 0, 10, 0)
            lbl = QLabel(text)
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setFont(QFont("Segoe UI", 9, QFont.Bold))
            if is_future:
                lbl.setStyleSheet("color: #757575; border: none;")
            else:
                lbl.setStyleSheet("color: white; border: none;")
            l.addWidget(lbl)
            return frame

        def arrow_mini():
            a = QLabel("↓")
            a.setAlignment(Qt.AlignCenter)
            a.setStyleSheet("color: #00BFA5; font-weight: bold; font-size: 14px; border: none;")
            return a

        flow_inner.addWidget(create_step_mini("Continuous System", "#455A64"))
        flow_inner.addWidget(arrow_mini())
        flow_inner.addWidget(create_step_mini("Discretization & Restrictions", "#0288D1"))
        flow_inner.addWidget(arrow_mini())
        flow_inner.addWidget(create_step_mini("Optimization Engine (DeVana Core)", "#388E3C"))
        flow_inner.addWidget(arrow_mini())
        
        out_hbox = QHBoxLayout()
        out_hbox.setSpacing(15)
        out_hbox.addWidget(create_step_mini("Topology (Q1)", "#F57C00"))
        out_hbox.addWidget(create_step_mini("Safe Ranges (Q2)", "#FBC02D"))
        flow_inner.addLayout(out_hbox)
        
        flow_inner.addWidget(arrow_mini())
        flow_inner.addWidget(create_step_mini("Future Continuous Validation", is_future=True))
        
        self.intro_layout.addWidget(flow_group, 0, Qt.AlignHCenter)

        # --- INFO CARDS (Vertical Stack) ---
        cards_vbox = QVBoxLayout()
        cards_vbox.setSpacing(12)

        self.q1_card = self._create_modern_card(
            "Q1: Optimal Configuration",
            "Identifies the best component combinations (mass, stiffness, damping, inerters) to minimize cost and complexity while maximizing vibration suppression.",
            "#F57C00"
        )
        self.q1_card.setFixedWidth(580)
        
        self.q2_card = self._create_modern_card(
            "Q2: Safe Parameter Ranges",
            "Extracts 'Safe Zones' ensuring any manufacturable value within the range meets engineering targets, accounting for real-world tolerances.",
            "#FBC02D"
        )
        self.q2_card.setFixedWidth(580)

        cards_vbox.addWidget(self.q1_card, 0, Qt.AlignHCenter)
        cards_vbox.addWidget(self.q2_card, 0, Qt.AlignHCenter)
        self.intro_layout.addLayout(cards_vbox)

        # --- FOOTER ---
        self.footer_text = QLabel(
            "DeVana transforms vibration control from a heuristic craft into an algorithmic science. "
            "By modeling continuous structures as discrete systems, we solve high-dimensional "
            "optimization problems to deliver reliable engineering results."
        )
        self.footer_text.setFont(QFont("Segoe UI", 9))
        self.footer_text.setWordWrap(True)
        self.footer_text.setAlignment(Qt.AlignCenter)
        self.footer_text.setStyleSheet("margin-top: 10px; margin-bottom: 10px;")
        self.intro_layout.addWidget(self.footer_text)

        # --- ACTION BUTTON ---
        self.start_btn = QPushButton("Go to Stochastic Design →")
        self.start_btn.setFixedWidth(280)
        self.start_btn.setObjectName("primary-button")
        self.start_btn.clicked.connect(lambda: self.change_page(1))
        self.intro_layout.addWidget(self.start_btn, 0, Qt.AlignHCenter)

        self.intro_layout.addStretch()

        self.update_introduction_theme()
        self.content_stack.addWidget(self.intro_page)

    def _create_modern_card(self, title, body, accent_color):
        card = QFrame()
        layout = QVBoxLayout(card)
        layout.setContentsMargins(15, 12, 15, 12)
        layout.setSpacing(5)

        t = QLabel(title)
        t.setFont(QFont("Segoe UI", 12, QFont.Bold))
        t.setStyleSheet(f"color: {accent_color}; border: none;")
        layout.addWidget(t)

        b = QLabel(body)
        b.setWordWrap(True)
        b.setFont(QFont("Segoe UI", 10))
        b.setStyleSheet("border: none;")
        layout.addWidget(b)

        card._title_label = t
        card._body_label = b
        return card

    def update_introduction_theme(self):
        """Sync visuals with Dark/Light themes"""
        theme = getattr(self, 'current_theme', 'Dark')
        is_dark = (theme == 'Dark')

        text_main = "#E0E0E0" if is_dark else "#333333"
        text_sec = "#9E9E9E" if is_dark else "#666666"
        card_bg = "#25252D" if is_dark else "#F9F9FB"
        border = "#3D3D4D" if is_dark else "#E0E0E5"

        self.mission_label.setStyleSheet(f"color: {text_sec};")
        self.footer_text.setStyleSheet(f"color: {text_sec};")
        
        card_style = f"""
            QFrame {{
                background-color: {card_bg};
                border: 1px solid {border};
                border-radius: 10px;
            }}
        """
        self.q1_card.setStyleSheet(card_style)
        self.q2_card.setStyleSheet(card_style)
        self.q1_card._body_label.setStyleSheet(f"color: {text_main}; border: none;")
        self.q2_card._body_label.setStyleSheet(f"color: {text_main}; border: none;")

        group_style = f"""
            QGroupBox#workflow-group {{
                font-weight: bold;
                border: 2px solid #00BFA5;
                border-radius: 12px;
                margin-top: 15px;
                background-color: {card_bg};
            }}
            QGroupBox#workflow-group::title {{
                subcontrol-origin: margin;
                subcontrol-position: top center;
                padding: 2px 20px;
                color: white;
                background-color: #00BFA5;
                border-radius: 6px;
            }}
        """
        self.intro_page.findChild(QGroupBox, "workflow-group").setStyleSheet(group_style)
