import sys
from pathlib import Path

import fitz
from PyQt6.QtCore import QSize, Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpacerItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


THUMBNAIL_WIDTH = 220
VIEWER_WIDTH = 720


class PdfRenderer:
    @staticmethod
    def render_page(pdf_path: Path, page_index: int, target_width: int) -> QPixmap:
        """Render one PDF page to a Qt pixmap at the requested width."""
        with fitz.open(pdf_path) as document:
            page = document.load_page(page_index)
            zoom = target_width / page.rect.width
            pixmap = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)

        image = QImage(
            pixmap.samples,
            pixmap.width,
            pixmap.height,
            pixmap.stride,
            QImage.Format.Format_RGB888,
        )
        return QPixmap.fromImage(image.copy())

    @staticmethod
    def page_count(pdf_path: Path) -> int:
        """Return the number of pages in a PDF file."""
        with fitz.open(pdf_path) as document:
            return document.page_count


class PdfCard(QFrame):
    def __init__(self, pdf_path: Path, pixmap: QPixmap, on_selected):
        """Create a clickable preview card for a PDF."""
        super().__init__()
        self.pdf_path = pdf_path
        self.on_selected = on_selected
        self.setObjectName("pdfCard")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedWidth(260)

        preview = QLabel()
        preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        preview.setPixmap(pixmap)
        preview.setFixedSize(QSize(236, 310))
        preview.setScaledContents(False)

        filename = QLabel(pdf_path.name)
        filename.setAlignment(Qt.AlignmentFlag.AlignCenter)
        filename.setWordWrap(True)
        filename.setObjectName("fileName")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)
        layout.addWidget(preview)
        layout.addWidget(filename)

    def mousePressEvent(self, event):
        """Open the represented PDF when the card is clicked."""
        if event.button() == Qt.MouseButton.LeftButton:
            event.accept()
            self.on_selected(self.pdf_path)
            return
        super().mousePressEvent(event)


class PdfBrowser(QMainWindow):
    def __init__(self):
        """Build the main PDF browser window."""
        super().__init__()
        self.setWindowTitle("PDF Browser")
        self.resize(1180, 760)

        self.current_directory: Path | None = None
        self.current_pdf: Path | None = None
        self.current_page = 0
        self.current_page_count = 0

        self.search_bar = QLineEdit()
        self.search_bar.setPlaceholderText("Search")
        self.search_bar.setClearButtonEnabled(True)
        self.search_bar.setEnabled(False)

        self.directory_button = QPushButton("Select Directory")
        self.directory_button.setObjectName("selectDirectoryButton")
        self.directory_button.clicked.connect(self.select_directory)

        top_bar = QHBoxLayout()
        top_bar.setContentsMargins(16, 16, 16, 10)
        top_bar.setSpacing(12)
        top_bar.addWidget(self.search_bar, 1)
        top_bar.addWidget(self.directory_button)

        self.body = QScrollArea()
        self.body.setWidgetResizable(True)
        self.body.setObjectName("body")

        self.description = QTextEdit()
        self.description.setReadOnly(True)
        self.description.setObjectName("metaBox")

        self.tags = QTextEdit()
        self.tags.setReadOnly(True)
        self.tags.setFixedHeight(150)
        self.tags.setObjectName("metaBox")

        right_panel = QWidget()
        right_panel.setObjectName("rightPanel")
        description_label = QLabel("Description")
        description_label.setObjectName("sectionLabel")
        tags_label = QLabel("Tags")
        tags_label.setObjectName("sectionLabel")
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(16, 16, 16, 16)
        right_layout.setSpacing(10)
        right_layout.addWidget(description_label)
        right_layout.addWidget(self.description, 1)
        right_layout.addWidget(tags_label)
        right_layout.addWidget(self.tags)
        right_panel.setFixedWidth(320)

        content = QHBoxLayout()
        content.setContentsMargins(16, 0, 16, 16)
        content.setSpacing(16)
        content.addWidget(self.body, 1)
        content.addWidget(right_panel)

        root = QWidget()
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.addLayout(top_bar)
        root_layout.addLayout(content, 1)
        self.setCentralWidget(root)

        self.show_empty_body()
        self.apply_styles()

    def select_directory(self):
        """Prompt for a directory and load its PDF files."""
        directory = QFileDialog.getExistingDirectory(self, "Select PDF Directory")
        if directory:
            self.current_directory = Path(directory)
            self.current_pdf = None
            self.current_page = 0
            self.load_directory()

    def load_directory(self):
        """Show first-page previews for PDFs in the current directory."""
        if not self.current_directory:
            return

        pdfs = sorted(self.current_directory.glob("*.pdf"), key=lambda path: path.name.lower())
        if not pdfs:
            self.show_empty_body()
            return

        grid_host = QWidget()
        grid = QGridLayout(grid_host)
        grid.setContentsMargins(18, 18, 18, 18)
        grid.setHorizontalSpacing(18)
        grid.setVerticalSpacing(18)

        for index, pdf_path in enumerate(pdfs):
            try:
                pixmap = PdfRenderer.render_page(pdf_path, 0, THUMBNAIL_WIDTH)
            except Exception:
                pixmap = QPixmap(QSize(THUMBNAIL_WIDTH, 300))
                pixmap.fill(Qt.GlobalColor.lightGray)

            card = PdfCard(pdf_path, pixmap, self.open_pdf)
            row, column = divmod(index, 3)
            grid.addWidget(card, row, column)

        grid.addItem(QSpacerItem(0, 0, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding), grid.rowCount(), 0)
        self.body.setWidget(grid_host)
        self.description.clear()
        self.tags.clear()

    def open_pdf(self, pdf_path: Path):
        """Open a selected PDF in single-page viewing mode."""
        try:
            page_count = PdfRenderer.page_count(pdf_path)
        except Exception as exc:
            QMessageBox.warning(self, "Unable to Open PDF", str(exc))
            return

        self.current_pdf = pdf_path
        self.current_page = 0
        self.current_page_count = page_count
        self.show_current_pdf_page()
        self.description.clear()
        self.tags.clear()

    def show_current_pdf_page(self):
        """Render and display the currently selected PDF page."""
        if not self.current_pdf:
            return

        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(14)

        nav = QHBoxLayout()
        back = QPushButton("Back")
        previous = QPushButton("Previous")
        next_page = QPushButton("Next")
        counter = QLabel(f"{self.current_page + 1} / {self.current_page_count}")
        counter.setAlignment(Qt.AlignmentFlag.AlignCenter)

        back.clicked.connect(self.load_directory)
        previous.clicked.connect(self.previous_page)
        next_page.clicked.connect(self.next_page)
        previous.setEnabled(self.current_page > 0)
        next_page.setEnabled(self.current_page < self.current_page_count - 1)

        nav.addWidget(back)
        nav.addStretch(1)
        nav.addWidget(previous)
        nav.addWidget(counter)
        nav.addWidget(next_page)

        image = QLabel()
        image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        image.setPixmap(PdfRenderer.render_page(self.current_pdf, self.current_page, VIEWER_WIDTH))

        layout.addLayout(nav)
        layout.addWidget(image, 1, Qt.AlignmentFlag.AlignCenter)
        self.body.setWidget(page)

    def previous_page(self):
        """Move the viewer to the previous PDF page."""
        if self.current_pdf and self.current_page > 0:
            self.current_page -= 1
            self.show_current_pdf_page()

    def next_page(self):
        """Move the viewer to the next PDF page."""
        if self.current_pdf and self.current_page < self.current_page_count - 1:
            self.current_page += 1
            self.show_current_pdf_page()

    def show_empty_body(self):
        """Show an empty main content area."""
        empty = QLabel()
        empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.body.setWidget(empty)
        self.description.clear()
        self.tags.clear()

    def apply_styles(self):
        """Apply the app-wide Qt stylesheet."""
        self.setStyleSheet(
            """
            QWidget {
                font-size: 14px;
            }
            QLineEdit, QTextEdit {
                border: 1px solid #c9ced6;
                border-radius: 6px;
                padding: 8px;
                background: #ffffff;
            }
            QPushButton {
                color: #ffffff;
                border: 1px solid #238636;
                border-radius: 6px;
                padding: 8px 14px;
                background: #2ea043;
            }
            QPushButton:hover {
                background: #238636;
            }
            QPushButton:disabled {
                color: #ffffff;
                border-color: #75b883;
                background: #75b883;
            }
            QScrollArea#body {
                border: 1px solid #d8dde5;
                border-radius: 8px;
                background: #f6f7f9;
            }
            QFrame#pdfCard {
                border: 1px solid #d8dde5;
                border-radius: 8px;
                background: #ffffff;
            }
            QFrame#pdfCard:hover {
                border-color: #7b93ba;
            }
            QLabel#fileName {
                color: #2c3440;
            }
            QLabel#sectionLabel {
                color: #1f2933;
                font-weight: 600;
                padding: 0 0 2px 0;
            }
            QWidget#rightPanel {
                border: 1px solid #d8dde5;
                border-radius: 8px;
                background: #ffffff;
            }
            QTextEdit#metaBox {
                background: #fbfcfd;
            }
            """
        )


def main():
    """Start the PDF browser application."""
    app = QApplication(sys.argv)
    window = PdfBrowser()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
