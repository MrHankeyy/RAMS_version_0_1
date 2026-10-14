"""Shared desktop design system and analysis navigation."""
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (QApplication, QWidget, QHBoxLayout, QVBoxLayout,
                            QGroupBox, QScrollArea, QTableWidget, QAbstractItemView, QPushButton, QTabWidget)

STYLE = """QWidget { color:#26364a; font-size:13px; }
QMainWindow, QDialog { background:#eef1f5; }
QWidget#appChrome { background:#23364d; }
QLabel#brand { color:white; font-size:18px; font-weight:600; }
QPushButton[workflow="true"] { background:transparent; color:#c7d3e1; border:0; border-bottom:3px solid transparent; border-radius:0; padding:13px 18px; }
QPushButton[workflow="true"]:hover { background:#304962; color:white; }
QPushButton[workflow="true"][active="true"] { color:white; background:#304962; border-bottom:3px solid #6db9f2; font-weight:600; }
QWidget#pageHeading { background:#fff; border-bottom:1px solid #d7dee7; }
QLabel[role="heading"] { font-size:20px; font-weight:600; color:#20374f; }
QLabel[role="muted"] { color:#65768a; }
QLabel[role="badge"] { color:#365d80; background:#edf3f9; border:1px solid #d6e3ef; padding:5px 10px; border-radius:3px; }
QPushButton { background:#fff; border:1px solid #bbc9d7; border-radius:3px; padding:5px 12px; min-height:20px; }
QPushButton:hover { background:#edf5fc; border-color:#6d9bbd; }
QPushButton:pressed { background:#dcebf7; }
QPushButton:focus { border:1px solid #2477b5; }
QPushButton:disabled { color:#95a1ae; background:#f1f3f5; border-color:#dde3e9; }
QPushButton[role="primary"] { background:#246da4; border-color:#246da4; color:white; font-weight:600; }
QPushButton[role="primary"]:hover { background:#1d5d8e; }
QPushButton[role="primary"]:disabled { background:#a6bacb; border-color:#a6bacb; }
QGroupBox { background:#fff; border:1px solid #d5dee7; border-radius:4px; margin-top:12px; padding:12px; font-weight:600; }
QGroupBox::title { subcontrol-origin:margin; left:12px; padding:0 5px; }
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox { background:#fff; border:1px solid #bdcbd8; border-radius:3px; padding:4px 7px; min-height:20px; selection-background-color:#d7eafa; selection-color:#173951; }
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus { border-color:#2477b5; }
QLineEdit:disabled, QComboBox:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled { background:#f2f4f6; color:#96a2ae; border-color:#e0e5ea; }
QComboBox { padding-right:22px; }
QComboBox::drop-down { width:20px; border:0; border-left:1px solid #e2e7ed; }
QComboBox QAbstractItemView { background:white; selection-background-color:#dcecf8; selection-color:#20374f; }
QTextEdit, QTextBrowser, QTableWidget { background:#fff; border:1px solid #d5dee7; selection-background-color:#dcecf8; selection-color:#193d59; }
QTableWidget { gridline-color:#e5eaf0; alternate-background-color:#f7f9fb; }
QTableWidget QLineEdit, QTableWidget QComboBox { border:0; border-radius:0; padding:3px 5px; }
QHeaderView::section { background:#edf2f7; color:#42566b; padding:6px 8px; border:0; border-right:1px solid #dce3eb; border-bottom:1px solid #d5dee7; font-weight:600; }
QTableCornerButton::section { background:#edf2f7; border:1px solid #d5dee7; }
QTabWidget::pane { background:white; border:1px solid #d5dee7; }
QTabBar::tab { background:#edf1f5; color:#52667a; border:0; border-bottom:2px solid transparent; padding:8px 12px; }
QTabBar::tab:selected { background:white; color:#246da4; border-bottom:2px solid #246da4; font-weight:600; }
QTabBar::tab:hover { background:#e4edf5; }
QScrollArea#resultNavigation { border:0; border-right:1px solid #d5dee7; background:#f5f7fa; }
QWidget#resultNavigationContent { background:#f5f7fa; }
QGroupBox#resultNavigationGroup { background:#fff; border:1px solid #c5d2df; border-radius:4px; margin-top:12px; padding:4px; color:#36536b; font-weight:600; }
QGroupBox#resultNavigationGroup::title { subcontrol-origin:margin; left:10px; padding:0 4px; }
QPushButton#resultNavigationButton { text-align:left; background:transparent; border:0; border-left:3px solid transparent; border-radius:2px; padding:4px 7px; min-height:18px; font-weight:400; }
QPushButton#resultNavigationButton:hover { background:#edf5fc; }
QPushButton#resultNavigationButton:checked { background:#e1edf7; color:#215c89; border-left:3px solid #246da4; }
QSplitter::handle { background:#e1e7ee; }
QSplitter::handle:hover { background:#9fb9d0; }
QScrollArea { border:0; background:transparent; }
QScrollBar:vertical { background:#f0f3f6; width:11px; margin:0; }
QScrollBar::handle:vertical { background:#bbc8d4; border-radius:4px; min-height:28px; margin:2px; }
QScrollBar:horizontal { background:#f0f3f6; height:11px; margin:0; }
QScrollBar::handle:horizontal { background:#bbc8d4; border-radius:4px; min-width:28px; margin:2px; }
QScrollBar::add-line, QScrollBar::sub-line { width:0; height:0; }
QToolTip { background:#253a50; color:white; border:0; padding:6px; }
"""


def install_theme(app):
    from pathlib import Path
    app.setStyle('Fusion')
    app.setFont(QFont('Microsoft YaHei', 10))
    icons = (Path(__file__).parent / 'assets' / 'ui').as_posix()
    app.setStyleSheet(STYLE + f'''
    QComboBox::down-arrow {{ image:url("{icons}/chevron-down.svg"); width:10px; height:10px; }}
    QSpinBox::up-button, QDoubleSpinBox::up-button {{ subcontrol-origin:border; subcontrol-position:top right; width:18px; border-left:1px solid #e2e7ed; }}
    QSpinBox::down-button, QDoubleSpinBox::down-button {{ subcontrol-origin:border; subcontrol-position:bottom right; width:18px; border-left:1px solid #e2e7ed; }}
    QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{ image:url("{icons}/chevron-up.svg"); width:8px; height:8px; }}
    QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{ image:url("{icons}/chevron-down.svg"); width:8px; height:8px; }}
    ''')


def primary(button):
    button.setStyleSheet('')
    button.setProperty('role', 'primary')


def polish_tables(root):
    for table in root.findChildren(QTableWidget):
        table.setAlternatingRowColors(True)
        table.verticalHeader().setDefaultSectionSize(36)
        table.horizontalHeader().setMinimumSectionSize(70)
        table.setHorizontalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        table.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)


class ResultsWorkspace(QWidget):
    """Keep existing tab indices and dispatch, expose a grouped result navigator."""
    GROUPS = [('设计分析', [0, 1, 9, 10]), ('模型与数据', [2, 3, 4, 7, 11]), ('取值与改进', [5, 6, 8])]

    def __init__(self, tabs):
        super().__init__()
        self.tabs = tabs
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.navigation = QScrollArea()
        self.navigation.setObjectName('resultNavigation')
        self.navigation.setWidgetResizable(True)
        self.navigation.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.navigation.setFixedWidth(196)
        layout.addWidget(self.navigation)
        layout.addWidget(tabs, 1)
        tabs.tabBar().hide()
        tabs.currentChanged.connect(self._sync)
        self.refresh()

    def refresh(self):
        self.buttons = {}
        self.groups = {}
        content = QWidget()
        content.setObjectName('resultNavigationContent')
        column = QVBoxLayout(content)
        column.setContentsMargins(8, 6, 8, 10)
        column.setSpacing(8)
        for title, indices in self.GROUPS:
            visible = [i for i in indices if self.tabs.isTabVisible(i)]
            if not visible:
                continue
            group = QGroupBox(title)
            group.setObjectName('resultNavigationGroup')
            group_layout = QVBoxLayout(group)
            group_layout.setContentsMargins(5, 14, 5, 5)
            group_layout.setSpacing(2)
            self.groups[title] = group
            for index in visible:
                button = QPushButton(self.tabs.tabText(index))
                button.setObjectName('resultNavigationButton')
                button.setCheckable(True)
                button.setCursor(Qt.CursorShape.PointingHandCursor)
                button.clicked.connect(lambda checked=False, i=index: self._activate(i))
                group_layout.addWidget(button)
                self.buttons[index] = button
            column.addWidget(group)
        column.addStretch()
        old = self.navigation.takeWidget()
        if old is not None:
            old.deleteLater()
        self.navigation.setWidget(content)
        self._sync(self.tabs.currentIndex())

    def _activate(self, index):
        self.tabs.setCurrentIndex(index)
        self._sync(self.tabs.currentIndex())

    def _sync(self, index):
        for tab_index, button in self.buttons.items():
            button.setChecked(tab_index == index)


class ResultTabs(QTabWidget):
    """Hidden analysis pages must not impose their text height on the active page."""
    def minimumSizeHint(self):
        widget = self.currentWidget()
        size = widget.minimumSizeHint() if widget else QSize(640, 340)
        return QSize(max(640, size.width()+2), max(340, size.height()+2))

    def sizeHint(self):
        return QSize(960, 380)

    def hasHeightForWidth(self):
        return False

    def heightForWidth(self, width):
        return self.minimumSizeHint().height()
