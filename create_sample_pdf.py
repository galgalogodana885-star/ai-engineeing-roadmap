from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

c = canvas.Canvas("addis_tech_handbook.pdf", pagesize=letter)

# Page 1
c.drawString(72, 740, "ADDIS TECH ENGINEERING HANDBOOK - PAGE 1")
c.drawString(72, 700, "1. Remote & Hybrid Work Policy")
c.drawString(72, 680, "Engineers based in Addis Ababa follow a hybrid schedule: Tuesday to Thursday in the Bole office,")
c.drawString(72, 665, "and Monday/Friday remote. Core collaboration hours are 10:00 AM to 3:00 PM EAT.")
c.drawString(72, 635, "2. Annual Leave & Public Holidays")
c.drawString(72, 615, "Full-time staff receive 20 paid vacation days per year plus all Ethiopian public holidays.")
c.drawString(72, 600, "Leave requests longer than 3 days must be submitted at least 2 weeks in advance.")
c.showPage()

# Page 2
c.drawString(72, 740, "ADDIS TECH ENGINEERING HANDBOOK - PAGE 2")
c.drawString(72, 700, "3. Backend Architecture & Deployment Standards")
c.drawString(72, 680, "All production APIs must be built with Python 3.11+ and FastAPI, validated with Pydantic.")
c.drawString(72, 665, "Services are containerized with Docker and deployed to Kubernetes with automated health checks.")
c.drawString(72, 635, "4. Travel & Meal Stipends")
c.drawString(72, 615, "Approved business travel includes a meal allowance of 2,500 ETB per day for domestic trips")
c.drawString(72, 600, "and $60 USD per day for international conferences. Receipts must be uploaded within 7 days.")
c.save()

print("Created addis_tech_handbook.pdf successfully!")