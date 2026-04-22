from PyQt6.QtWidgets import (
    QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QStackedWidget
)

from pages.business_model_page import BusinessModelPage
from pages.simple_page import SimplePage
from models import ProjectData


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.project_data = ProjectData()

        self.setWindowTitle("鲁棒优化管理系统")
        self.resize(1000, 650)

        self._build_ui()

    def _build_ui(self):
        self.btn_a = QPushButton("业务建模")
        self.btn_b = QPushButton("方案配置")
        self.btn_c = QPushButton("数据管理")
        self.btn_d = QPushButton("设计优化")
        self.btn_e = QPushButton("分析报告")

        top_layout = QHBoxLayout()
        top_layout.addWidget(self.btn_a)
        top_layout.addWidget(self.btn_b)
        top_layout.addWidget(self.btn_c)
        top_layout.addWidget(self.btn_d)
        top_layout.addWidget(self.btn_e)

        self.stacked_widget = QStackedWidget()

        self.page1 = BusinessModelPage(self.project_data)
        self.page2 = SimplePage("这是方案配置界面")
        self.page3 = SimplePage("这是数据管理界面")
        self.page4 = SimplePage("这是设计优化界面")
        self.page5 = SimplePage("这是分析报告界面")

        self.stacked_widget.addWidget(self.page1)
        self.stacked_widget.addWidget(self.page2)
        self.stacked_widget.addWidget(self.page3)
        self.stacked_widget.addWidget(self.page4)
        self.stacked_widget.addWidget(self.page5)

        self.btn_a.clicked.connect(lambda: self.switch_page(0))
        self.btn_b.clicked.connect(lambda: self.switch_page(1))
        self.btn_c.clicked.connect(lambda: self.switch_page(2))
        self.btn_d.clicked.connect(lambda: self.switch_page(3))
        self.btn_e.clicked.connect(lambda: self.switch_page(4))

        main_layout = QVBoxLayout()
        main_layout.addLayout(top_layout)
        main_layout.addWidget(self.stacked_widget)
        self.setLayout(main_layout)

    def switch_page(self, index: int):
        if index != 0:
            self.page1.sync_to_project_data()
        self.stacked_widget.setCurrentIndex(index)