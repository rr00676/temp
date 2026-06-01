import sys
from pathlib import Path

import fitz
from PyQt6.QtCore import QPoint, QRect, QSize, Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QLayout,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)


THUMBNAIL_WIDTH = 220


class FlowLayout(QLayout):
    """Layout widgets left-to-right and wrap them as space changes."""

    def __init__(self, parent=None, margin=0, spacing=18):
        super().__init__(parent)
        self.items = []
        self.setContentsMargins(margin, margin, margin, margin)
        self.setSpacing(spacing)

    def addItem(self, item):
        """Add a layout item to the flow."""
        self.items.append(item)

    def count(self):
        """Return the number of items in the layout."""
        return len(self.items)

    def itemAt(self, index):
        """Return the layout item at an index."""
        if 0 <= index < len(self.items):
            return self.items[index]
        return None

    def takeAt(self, index):
        """Remove and return the layout item at an index."""
        if 0 <= index < len(self.items):
            return self.items.pop(index)
        return None

    def expandingDirections(self):
        """Keep the flow from stretching child widgets apart."""
        return Qt.Orientations(Qt.Orientation(0))

    def hasHeightForWidth(self):
        """Tell Qt that height depends on the available width."""
        return True

    def heightForWidth(self, width):
        """Calculate the wrapped layout height for a width."""
        return self.do_layout(QRect(0, 0, width, 0), True)

    def setGeometry(self, rect):
        """Position child widgets inside the available rectangle."""
        super().setGeometry(rect)
        self.do_layout(rect, False)

    def sizeHint(self):
        """Return the preferred layout size."""
        return self.minimumSize()

    def minimumSize(self):
        """Return the minimum size needed for the layout."""
        size = QSize()
        for item in self.items:
            size = size.expandedTo(item.minimumSize())

        margins = self.contentsMargins()
        size += QSize(margins.left() + margins.right(), margins.top() + margins.bottom())
        return size

    def do_layout(self, rect, test_only):
        """Lay out items and return the resulting height."""
        margins = self.contentsMargins()
        effective_rect = rect.adjusted(
            margins.left(),
            margins.top(),
            -margins.right(),
            -margins.bottom(),
        )
        x = effective_rect.x()
        y = effective_rect.y()
        line_height = 0
        spacing = self.spacing()

        for item in self.items:
            item_size = item.sizeHint()
            next_x = x + item_size.width() + spacing

            if next_x - spacing > effective_rect.right() and line_height > 0:
                x = effective_rect.x()
                y += line_height + spacing
                next_x = x + item_size.width() + spacing
                line_height = 0

            if not test_only:
                item.setGeometry(QRect(QPoint(x, y), item_size))

            x = next_x
            line_height = max(line_height, item_size.height())

        return y + line_height + margins.bottom() - rect.y()


def render_first_page(pdf_path: Path) -> QPixmap:
    """Render the first page of a PDF as a thumbnail."""
    with fitz.open(pdf_path) as document:
        page = document.load_page(0)
        zoom = THUMBNAIL_WIDTH / page.rect.width
        pixmap = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)

    image = QImage(
        pixmap.samples,
        pixmap.width,
        pixmap.height,
        pixmap.stride,
        QImage.Format.Format_RGB888,
    )
    return QPixmap.fromImage(image.copy())


class PdfPreviewCard(QFrame):
    """Clickable preview card for one PDF."""

    def __init__(self, pdf_path: Path, pixmap: QPixmap):
        super().__init__()
        self.pdf_path = pdf_path
        self.setObjectName("pdfCard")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedWidth(260)

        preview = QLabel()
        preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        preview.setFixedSize(QSize(236, 310))
        preview.setPixmap(pixmap)

        filename = QLabel(pdf_path.name)
        filename.setObjectName("fileName")
        filename.setAlignment(Qt.AlignmentFlag.AlignCenter)
        filename.setWordWrap(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)
        layout.addWidget(preview)
        layout.addWidget(filename)

    def mousePressEvent(self, event):
        """Select the PDF card when clicked."""
        if event.button() == Qt.MouseButton.LeftButton:
            event.accept()
            QMessageBox.information(self, "PDF Selected", self.pdf_path.name)
            return
        super().mousePressEvent(event)


class PdfPreviewBrowser(QMainWindow):
    """Main window for browsing PDF first-page previews."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("PDF Preview Browser")
        self.resize(1050, 760)

        self.search_bar = QLineEdit()
        self.search_bar.setPlaceholderText("Search")
        self.search_bar.setClearButtonEnabled(True)

        self.search_button = QPushButton("Search")
        self.search_button.setEnabled(False)

        self.select_directory_button = QPushButton("Select Directory")
        self.select_directory_button.clicked.connect(self.select_directory)

        top_bar = QHBoxLayout()
        top_bar.setContentsMargins(16, 16, 16, 10)
        top_bar.setSpacing(10)
        top_bar.addWidget(self.search_bar, 1)
        top_bar.addWidget(self.search_button)
        top_bar.addWidget(self.select_directory_button)

        self.scroll_area = QScrollArea()
        self.scroll_area.setObjectName("body")
        self.scroll_area.setWidgetResizable(True)

        root = QWidget()
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.addLayout(top_bar)
        root_layout.addWidget(self.scroll_area, 1)

        self.setCentralWidget(root)
        self.show_empty_state()
        self.apply_styles()

    def select_directory(self):
        """Ask the user for a directory and display its PDFs."""
        directory = QFileDialog.getExistingDirectory(self, "Select PDF Directory")
        if directory:
            self.load_pdf_directory(Path(directory))

    def load_pdf_directory(self, directory: Path):
        """Load PDF thumbnails from a directory into the scroll area."""
        pdfs = sorted(directory.glob("*.pdf"), key=lambda path: path.name.lower())
        if not pdfs:
            self.show_empty_state("No PDFs found in the selected directory.")
            return

        grid_host = QWidget()
        grid = FlowLayout(grid_host, margin=18, spacing=18)

        for pdf_path in pdfs:
            card = self.create_pdf_card(pdf_path)
            grid.addWidget(card)

        self.scroll_area.setWidget(grid_host)

    def create_pdf_card(self, pdf_path: Path) -> PdfPreviewCard:
        """Create a preview card, using a blank thumbnail if rendering fails."""
        try:
            pixmap = render_first_page(pdf_path)
        except Exception:
            pixmap = QPixmap(QSize(THUMBNAIL_WIDTH, 300))
            pixmap.fill(Qt.GlobalColor.lightGray)
        return PdfPreviewCard(pdf_path, pixmap)

    def show_empty_state(self, message: str = "Select a directory to view PDF previews."):
        """Show an empty-state message in the scroll area."""
        label = QLabel(message)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setObjectName("emptyState")
        self.scroll_area.setWidget(label)

    def apply_styles(self):
        """Apply the app stylesheet."""
        self.setStyleSheet(
            """
            QWidget {
                font-size: 14px;
            }
            QLineEdit {
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
                border-color: #75b883;
                background: #75b883;
            }
            QScrollArea#body {
                border: 1px solid #d8dde5;
                border-radius: 8px;
                background: #f6f7f9;
                margin: 0 16px 16px 16px;
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
            QLabel#emptyState {
                color: #667085;
            }
            """
        )


def main():
    """Start the PDF preview browser."""
    app = QApplication(sys.argv)
    window = PdfPreviewBrowser()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
