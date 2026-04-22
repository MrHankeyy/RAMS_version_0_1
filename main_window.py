from PyQt6.QtWidgets import (
    QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QStackedWidget, QLabel
)
from PyQt6.QtCore import Qt

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

        '''
        包含五个组成部分：
        1. 菜单栏 Menu Bar
        2. 模块导航栏 Module Navigation Bar
        3. 主工作区 Main Workspace 
        4. 项目浏览器 Project Explorer
        5. 消息面板 Message Panel
        
        五个工作区：
        1. 业务建模 Model Setup, MOD
        2. 方案配置 Configuration, CFG
        3. 数据管理 Data Management, DM
        4. 设计优化 Optimization, OPT
        5. 分析报告 Report, REP
        '''

        main_layout = QVBoxLayout()
        mid_layout = QHBoxLayout()
        mid_right_layout = QVBoxLayout()
        
        menu_bar_panel = self._build_menu_bar()
        project_explorer_panel = self._build_project_explorer()
        module_nav_panel = self._build_module_navigation_bar()
        main_workspace_panel = self._build_main_workspace()
        message_panel_panel = self._build_message_panel()

        mid_layout.addWidget(project_explorer_panel, 1)
        mid_right_layout.addWidget(module_nav_panel, 1)
        mid_right_layout.addWidget(main_workspace_panel, 9)
        mid_layout.addLayout(mid_right_layout, 5)

        main_layout.addLayout(menu_bar_panel, 1)
        main_layout.addLayout(mid_layout, 10)
        main_layout.addWidget(message_panel_panel, 2)

        self.setLayout(main_layout)

    def on_test_click(self, name):
        print(f"已点击：{name}")

    def on_switch_module(self, module_name):
        print(f"已切换到模块：{module_name}")

    def _build_menu_bar(self):
        self.btn_file = QPushButton("文件")
        self.btn_help = QPushButton("帮助")
        self.btn_setting = QPushButton("设置")

        self.btn_file.clicked.connect(lambda: self.on_test_click("文件"))
        self.btn_help.clicked.connect(lambda: self.on_test_click("帮助"))
        self.btn_setting.clicked.connect(lambda: self.on_test_click("设置"))

        menu_bar_layout = QHBoxLayout()
        menu_bar_layout.addWidget(self.btn_file)
        menu_bar_layout.addWidget(self.btn_help)
        menu_bar_layout.addWidget(self.btn_setting) 

        return menu_bar_layout
    
    def _build_module_navigation_bar(self):
        widget = QWidget()        
        widget.setObjectName("module_navigation_bar")
        layout = QHBoxLayout(widget)

        self.btn_switch_MOD = QPushButton("业务建模")
        self.btn_switch_CFG = QPushButton("方案配置")
        self.btn_switch_DM = QPushButton("数据管理")
        self.btn_switch_OPT = QPushButton("设计优化")
        self.btn_switch_REP = QPushButton("分析报告")
        
        self.btn_switch_MOD.clicked.connect(
            lambda: self.on_switch_module("业务建模")
        )
        self.btn_switch_CFG.clicked.connect(
            lambda: self.on_switch_module("方案配置")
        )
        self.btn_switch_DM.clicked.connect(
            lambda: self.on_switch_module("数据管理")
        )
        self.btn_switch_OPT.clicked.connect(
            lambda: self.on_switch_module("设计优化")
        )
        self.btn_switch_REP.clicked.connect(
            lambda: self.on_switch_module("分析报告")
        )

        layout.addWidget(self.btn_switch_MOD)
        layout.addWidget(self.btn_switch_CFG)
        layout.addWidget(self.btn_switch_DM)
        layout.addWidget(self.btn_switch_OPT)
        layout.addWidget(self.btn_switch_REP)

        widget.setStyleSheet("""
        #module_navigation_bar {
            border: 2px solid black;
        }
        """)

        return widget
    
    def _build_main_workspace(self):
        widget = QWidget()
        widget.setObjectName("main_workspace")
        layout = QVBoxLayout(widget)

        temp_label = QLabel('这是主工作区') #临时标签，后续接入主工作区后去除
        temp_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(temp_label)

        widget.setStyleSheet("""
        #main_workspace {
            border: 2px solid black;
            border-radius: 12px;
            background-color: #ffffff;
        }
        """)

        return widget
    
    def _build_project_explorer(self):
        widget = QWidget()
        widget.setObjectName("project_explorer")
        layout = QVBoxLayout(widget)

        temp_label = QLabel('这是主项目浏览器') #临时标签，后续接入主项目浏览器后去除
        temp_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(temp_label)

        widget.setStyleSheet("""
        #project_explorer {
            border: 2px solid black;
        }
        """)

        return widget
    
    def _build_message_panel(self):
        widget = QWidget()
        widget.setObjectName("message_panel")
        layout = QVBoxLayout(widget)
        
        temp_label = QLabel('这是消息面板') #临时标签，后续接入消息面板后去除
        temp_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(temp_label)

        widget.setStyleSheet("""
        #message_panel {
            border: 2px solid black;
        }
        """)

        return widget
        
'''
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

'''