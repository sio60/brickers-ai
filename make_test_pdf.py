from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader

PDF_PATH = "test_image_embed.pdf"
IMG_PATH = "pyramid_step_04.png"  # 여기에 진짜 png/jpg 경로 넣기

c = canvas.Canvas(PDF_PATH, pagesize=A4)
w, h = A4

c.setFont("Helvetica", 14)
c.drawString(50, h - 50, "PDF image embed test")

img = ImageReader(IMG_PATH)
c.drawImage(img, 50, h - 350, width=300, height=300, preserveAspectRatio=True, mask='auto')

c.showPage()
c.save()

print("saved:", PDF_PATH)
