from pathlib import Path

from PIL import Image


PAGE_SIZE = (2550, 3300)
MARGIN = 260


def make_page(image_path: str) -> Image.Image:
    page = Image.new("RGB", PAGE_SIZE, "white")
    image = Image.open(image_path).convert("RGB")
    max_w = PAGE_SIZE[0] - (MARGIN * 2)
    max_h = PAGE_SIZE[1] - (MARGIN * 2)
    image.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)
    x = (PAGE_SIZE[0] - image.width) // 2
    y = (PAGE_SIZE[1] - image.height) // 2
    page.paste(image, (x, y))
    return page


if __name__ == "__main__":
    pages = [
        make_page("dog_page_image.png"),
        make_page("cat_page_image.png"),
    ]
    pages[0].save(
        "dog_cat_pages.pdf",
        "PDF",
        resolution=300.0,
        save_all=True,
        append_images=pages[1:],
    )
