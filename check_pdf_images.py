import fitz
import os

PDF_PATH = "test_image_embed.pdf"

if not os.path.exists(PDF_PATH):
    raise FileNotFoundError(f"PDF not found: {PDF_PATH}")

doc = fitz.open(PDF_PATH)

total = 0
for i, page in enumerate(doc):
    imgs = page.get_images(full=True)
    print(f"page {i+1}: images={len(imgs)}")
    total += len(imgs)

print("TOTAL images:", total)