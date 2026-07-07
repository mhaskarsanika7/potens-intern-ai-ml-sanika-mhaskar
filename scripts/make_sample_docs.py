"""
make_sample_docs.py
--------------------
Generates 5 small sample PDFs into docs/ so the pipeline can be run
end-to-end without needing real source documents. Two of the five
(refund_policy_v1 / refund_policy_v2) deliberately contradict each
other on the refund window, to give the /contradict endpoint something
real to detect. Not part of the graded deliverable - just test fixtures.

Run: python scripts/make_sample_docs.py
"""

import os
from reportlab.lib.pagesizes import LETTER
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "docs")
os.makedirs(OUT_DIR, exist_ok=True)

styles = getSampleStyleSheet()


def make_pdf(filename, title, paragraphs):
    path = os.path.join(OUT_DIR, filename)
    doc = SimpleDocTemplate(path, pagesize=LETTER)
    story = [Paragraph(title, styles["Title"]), Spacer(1, 12)]
    for p in paragraphs:
        story.append(Paragraph(p, styles["BodyText"]))
        story.append(Spacer(1, 10))
    doc.build(story)
    print(f"wrote {path}")


make_pdf(
    "employee_handbook.pdf",
    "Acme Corp Employee Handbook",
    [
        "All full-time employees are entitled to 20 days of paid vacation per calendar year, "
        "accrued monthly at a rate of 1.67 days per month.",
        "Employees must submit vacation requests at least two weeks in advance through the HR "
        "portal. Requests submitted with less notice may be denied at manager discretion.",
        "Remote work is permitted up to 3 days per week for employees who have completed their "
        "90-day probationary period. Fully remote arrangements require VP approval.",
        "The standard workweek is 40 hours, Monday through Friday. Overtime for non-exempt "
        "employees is paid at 1.5x the base hourly rate for hours worked beyond 40 in a week.",
        "New hires receive a laptop, a company phone stipend of $50 per month, and access to the "
        "group health insurance plan starting on their first day of employment.",
    ],
)

make_pdf(
    "refund_policy_v1.pdf",
    "Acme Corp Refund Policy (Consumer Products) - Effective Jan 2025",
    [
        "Customers may request a full refund within 30 days of purchase, provided the product is "
        "unused and returned in its original packaging.",
        "Refunds are processed to the original payment method within 5-7 business days of the "
        "returned item being received at our warehouse.",
        "Digital products, including software licenses and downloadable content, are non-refundable "
        "once the license key has been activated.",
        "Shipping costs for the original order are non-refundable, except in cases where the "
        "product arrived damaged or defective.",
    ],
)

make_pdf(
    "refund_policy_v2.pdf",
    "Acme Corp Refund Policy (Consumer Products) - Effective Jan 2026",
    [
        "Effective January 2026, customers may request a full refund within 14 days of purchase. "
        "Items must be unused and in original packaging.",
        "Refunds are processed to the original payment method within 3-5 business days of the "
        "returned item being received at our warehouse.",
        "Digital products remain non-refundable once activated, consistent with prior policy.",
        "A restocking fee of 10 percent will now be applied to returns of large appliances over $500.",
    ],
)

make_pdf(
    "security_guide.pdf",
    "Acme Corp Information Security Guide",
    [
        "All employees must enable multi-factor authentication on their corporate email and VPN "
        "accounts within their first week of employment.",
        "Passwords must be at least 12 characters and rotated every 90 days. Password reuse across "
        "the last 5 passwords is prohibited by the identity system.",
        "Company laptops must have full-disk encryption enabled and must not be left unattended in "
        "public spaces. Loss or theft must be reported to IT Security within 24 hours.",
        "Sensitive customer data may only be stored in approved systems (the primary database and "
        "the designated data warehouse), never in local spreadsheets or personal cloud storage.",
    ],
)

make_pdf(
    "onboarding_guide.pdf",
    "Acme Corp New Hire Onboarding Guide",
    [
        "On day one, new hires complete IT setup, including laptop provisioning, email account "
        "creation, and enrollment in multi-factor authentication for VPN and email access.",
        "New hires are assigned an onboarding buddy for their first 30 days to help navigate team "
        "processes and answer day-to-day questions.",
        "Benefits enrollment, including health insurance selection, must be completed within the "
        "first 30 days of employment through the HR portal.",
        "Managers should schedule a 30-60-90 day check-in cadence to review goals and provide "
        "feedback during the new hire's ramp-up period.",
    ],
)

print("Done.")
