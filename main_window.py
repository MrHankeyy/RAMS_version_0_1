from PyQt6.QtWidgets import (
    QWidget, QPushButton, QVBoxLayout, QHBoxLayout, 
    QStackedWidget, QLabel, QMainWindow, QMenu, QSizePolicy, QApplication
)
from PyQt6.QtGui import QAction, QIcon
from PyQt6.QtCore import Qt

from pages.modeling_page import ModelingPage
from pages.configuration_page import ConfigurationPage
from pages.data_management_page import DataManagementPage
from pages.optimization_page import OptimizationPage
from pages.analysis_page import AnalysisPage
from models import ProjectData


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.project_data = ProjectData()

        self.setWindowTitle("鲁棒优化管理系统")
        self.resize(1360, 860)
        self.setWindowIcon(QIcon("icon.png"))
        font = self.font()
        font.setFamily("Microsoft YaHei")
        font.setPointSize(10)
        QApplication.instance().setFont(font)

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

        self._build_menu_bar()

        main_layout = QVBoxLayout()        
        mid_layout = QHBoxLayout()
        mid_right_layout = QVBoxLayout()
        
        project_explorer_panel = self._build_project_explorer()
        module_nav_panel = self._build_module_navigation_bar()
        main_workspace_panel = self._build_main_workspace()
        message_panel = self._build_message_panel()

        mid_layout.addWidget(project_explorer_panel, 1)
        mid_right_layout.addWidget(module_nav_panel)
        mid_right_layout.addWidget(main_workspace_panel, 9)
        mid_layout.addLayout(mid_right_layout, 5)

        main_layout.addLayout(mid_layout, 10)
        main_layout.addWidget(message_panel, 2)

        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)
        
        central_widget.setObjectName("central_widget")
        central_widget.setStyleSheet("""
        QWidget#central_widget { background: #f4f6f9; }
        QTabWidget::pane { border: 1px solid #cdd6e0; background: #ffffff;
                           border-radius: 4px; }
        QTabBar::tab { padding: 6px 12px; background: #e9eef4;
                       border: 1px solid #cdd6e0; border-bottom: none; }
        QTabBar::tab:selected { background: #ffffff; font-weight: bold; }
        QGroupBox { margin-top: 14px; font-weight: bold; color: #1f2937; }
        QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 6px; }
        QTableWidget { gridline-color: #dde4ec; alternate-background-color: #f8fafc; }
        QHeaderView::section { background: #eef2f7; padding: 5px; border: 1px solid #d6dee8; }
        QPushButton { border-radius: 4px; }
        """)
    
    def _build_menu_bar(self):
        menu_bar = self.menuBar()

        file_menu = menu_bar.addMenu("文件")

        action_new = QAction("新建项目", self)
        action_open = QAction("打开项目", self)
        action_save = QAction("保存", self)

        file_menu.addAction(action_new)
        file_menu.addAction(action_open)
        file_menu.addAction(action_save)

        setting_menu = menu_bar.addMenu("设置")
        action_setting = QAction("参数设置", self)
        setting_menu.addAction(action_setting)

        help_menu = menu_bar.addMenu("帮助")
        action_about = QAction("关于", self)
        help_menu.addAction(action_about)

        action_new.triggered.connect(lambda: print("新建项目"))
        action_open.triggered.connect(lambda: print("打开项目"))
        action_save.triggered.connect(lambda: print("保存"))
    
    def _build_module_navigation_bar(self):
        widget = QWidget()        
        widget.setObjectName("module_navigation_bar")

        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.btn_switch_MOD = QPushButton("业务建模")
        self.btn_switch_CFG = QPushButton("方案配置")
        self.btn_switch_DM = QPushButton("数据管理")
        self.btn_switch_OPT = QPushButton("设计优化")
        self.btn_switch_REP = QPushButton("分析报告")

        self.btns_switch = [
            self.btn_switch_MOD, 
            self.btn_switch_CFG, 
            self.btn_switch_DM, 
            self.btn_switch_OPT, 
            self.btn_switch_REP, 
        ]
        for btn in self.btns_switch:
            btn.setSizePolicy(
                QSizePolicy.Policy.Expanding, 
                QSizePolicy.Policy.Expanding
            )
            btn.setProperty("active", False)             #每个按钮默认为非激活状态，表示页面是否选中

        self.btn_switch_MOD.setProperty("active", True)                  #初始化默认激活业务建模页面
        
        self.btn_switch_MOD.clicked.connect(
            lambda: self.on_switch_module("业务建模", 0)
        )
        self.btn_switch_CFG.clicked.connect(
            lambda: self.on_switch_module("方案配置", 1)
        )
        self.btn_switch_DM.clicked.connect(
            lambda: self.on_switch_module("数据管理", 2)
        )
        self.btn_switch_OPT.clicked.connect(
            lambda: self.on_switch_module("设计优化", 3)
        )
        self.btn_switch_REP.clicked.connect(
            lambda: self.on_switch_module("分析报告", 4)
        )

        layout.addWidget(self.btn_switch_MOD)
        layout.addWidget(self.btn_switch_CFG)
        layout.addWidget(self.btn_switch_DM)
        layout.addWidget(self.btn_switch_OPT)
        layout.addWidget(self.btn_switch_REP)

        widget.setStyleSheet("""
        #module_navigation_bar {
            border: None;
        }

        /* 按钮默认 */
        QPushButton {
            border: 1px solid #bbb;
            background-color: #f0f0f0;
            font-weight: bold;
            padding: 7px;
            font-size: 15px;
        }

        /* 鼠标悬停 */
        QPushButton:hover {
            background-color: #e0e0e0;
        }

        /* 按下 */
        QPushButton:pressed {
            background-color: #d0d0d0;
        }

        /* 选中 */         
        QPushButton[active="true"] {
            background-color: #c8dfff;
            font-weight: bold;
        }
        """)
        widget.setFixedHeight(52)
        
        return widget
    
    def _build_main_workspace(self):
        widget = QWidget()
        widget.setObjectName("main_workspace")
        layout = QVBoxLayout(widget)

        self.stacked_widget = QStackedWidget()

        self.page_MOD = ModelingPage(self.project_data)
        self.page_CFG = ConfigurationPage(self.project_data)
        self.page_DM = DataManagementPage(self.project_data)
        self.page_OPT = OptimizationPage(self.project_data)
        self.page_REP = AnalysisPage(self.project_data)
        
        # 绑定优化计算完成信号：自动跳转到“分析报告”页
        self.page_OPT.optimization_finished.connect(lambda: self.on_switch_module("分析报告", 4))

        self.stacked_widget.addWidget(self.page_MOD)
        self.stacked_widget.addWidget(self.page_CFG)
        self.stacked_widget.addWidget(self.page_DM)
        self.stacked_widget.addWidget(self.page_OPT)
        self.stacked_widget.addWidget(self.page_REP)

        layout.addWidget(self.stacked_widget)

        widget.setStyleSheet("""
        #main_workspace {
            border: 1px solid #bbb;
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
        widget.setFixedWidth(230)

        widget.setStyleSheet("""
        #project_explorer {
            border: 1px solid #bbb;
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
        widget.setMinimumHeight(90)

        widget.setStyleSheet("""
        #message_panel {
            border: 1px solid #bbb;
        }
        """)

        return widget
        
    def on_test_click(self, name):
        print(f"已点击：{name}")

    def on_switch_module(self, module_name, index):
        self.switch_page(index)
        
        for i, btn in enumerate(self.btns_switch):
            btn.setProperty("active", i == index)
            btn.style().unpolish(btn)
            btn.style().polish(btn)
            btn.update()

    def switch_page(self, index: int):
        if index != 0:
            self.page_MOD.sync_to_project_data()
        self.stacked_widget.setCurrentIndex(index)
