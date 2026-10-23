"""
Script to generate a professional, formatted 'hr_manual.pdf' with realistic
HR policies for testing and running the HR Policy Assistant RAG application.
"""

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

def create_hr_manual(filename="hr_manual.pdf"):
    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        rightMargin=54,
        leftMargin=54,
        topMargin=54,
        bottomMargin=54
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=24,
        leading=28,
        textColor=colors.HexColor("#1A365D"),
        spaceAfter=6
    )

    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#4A5568"),
        spaceAfter=15
    )

    h1_style = ParagraphStyle(
        'SectionH1',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=16,
        leading=20,
        textColor=colors.HexColor("#2B6CB0"),
        spaceBefore=14,
        spaceAfter=8
    )

    h2_style = ParagraphStyle(
        'SectionH2',
        parent=styles['Heading3'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#2D3748"),
        spaceBefore=10,
        spaceAfter=4
    )

    body_style = ParagraphStyle(
        'BodyDark',
        parent=styles['BodyText'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#2D3748"),
        spaceAfter=8
    )

    bullet_style = ParagraphStyle(
        'BulletText',
        parent=styles['BodyText'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        leftIndent=15,
        textColor=colors.HexColor("#2D3748"),
        spaceAfter=4
    )

    story = []

    # Title & Header
    story.append(Paragraph("Acme Corp — Employee HR Policy Manual", title_style))
    story.append(Paragraph("Version 3.2 | Operational Guidelines and Employee Regulations", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#2B6CB0"), spaceAfter=14))

    # Section 1
    story.append(Paragraph("1. Employment Classification and Scope", h1_style))
    story.append(Paragraph(
        "This manual outlines the operational human resource policies governing permanent, probationary, "
        "and contract employees at Acme Corp. All employees are expected to familiarize themselves with these guidelines.",
        body_style
    ))

    # Section 2
    story.append(Paragraph("2. Probation and Confirmation Policy", h1_style))
    story.append(Paragraph("Section 2.1: Objective", h2_style))
    story.append(Paragraph(
        "The probationary period allows both the organization and the employee to assess role compatibility, performance standards, "
        "and cultural alignment.",
        body_style
    ))
    story.append(Paragraph("Section 2.2: Duration of Probation", h2_style))
    story.append(Paragraph(
        "The probation period for all new employees up to the grade of Senior Manager is six (6) months from the official date of joining. "
        "For executive roles at the level of Director and above, the probation period shall be twelve (12) months. "
        "Under exceptional circumstances where performance evaluation is inconclusive, the management reserves the right to extend the probation "
        "period by an additional three (3) months with formal written notification.",
        body_style
    ))
    story.append(Paragraph("Section 2.3: Notice Period During Probation", h2_style))
    story.append(Paragraph(
        "During the probationary period, either party may terminate the employment contract by providing fifteen (15) calendar days of written notice "
        "or salary in lieu thereof.",
        body_style
    ))

    # Section 3
    story.append(Paragraph("3. Leave and Absence Administration", h1_style))
    story.append(Paragraph("Section 3.1: Privilege Leave (PL)", h2_style))
    story.append(Paragraph(
        "Permanent employees are entitled to eighteen (18) days of Privilege Leave (PL) per calendar year. "
        "Privilege Leave is credited at the rate of 1.5 days per completed month of active service. "
        "Employees serving their probation period may accrue Privilege Leave; however, utilization of PL is permitted only upon successful confirmation. "
        "A maximum of forty-five (45) unused PL days can be carried forward into the subsequent calendar year. Any excess beyond 45 days shall lapse.",
        body_style
    ))
    story.append(Paragraph("Section 3.2: Casual Leave (CL) and Sick Leave (SL)", h2_style))
    story.append(Paragraph(
        "In addition to Privilege Leave, permanent and probationary employees receive seven (7) days of Casual Leave and seven (7) days of Sick Leave per annum. "
        "Casual Leave cannot be combined with Privilege Leave or extended across more than three consecutive days without prior management approval.",
        body_style
    ))
    story.append(Paragraph("Section 3.3: Maternity and Parental Leave", h2_style))
    story.append(Paragraph(
        "In compliance with statutory requirements, eligible female employees are entitled to twenty-six (26) weeks, equivalent to one hundred eighty-two (182) days, "
        "of fully paid maternity leave for up to the first two surviving children. To qualify, the employee must have completed a minimum of eighty (80) days of "
        "active service with the company in the twelve (12) months preceding the expected delivery date. "
        "For employees with two or more surviving children, maternity leave entitlement is twelve (12) weeks (84 days). "
        "Male employees are entitled to five (5) working days of paid Paternity Leave to be utilized within four months of child birth.",
        body_style
    ))

    # Section 4
    story.append(Paragraph("4. Employee Benefits and Insurance Coverage", h1_style))
    story.append(Paragraph("Section 4.1: Group Medical Health Insurance", h2_style))
    story.append(Paragraph(
        "All confirmed and probationary permanent employees are enrolled into the company Group Medical Cover (GMC) effective from their first working day. "
        "The policy offers hospitalization coverage up to INR 5,00,000 per family unit (covering the employee, spouse, and up to two dependent children). "
        "The company pays an annual health-insurance premium amount of INR 15,000 per employee family unit directly to the insurer. "
        "Employees are not required to make any monthly or annual co-contribution towards this standard base premium amount.",
        body_style
    ))
    story.append(Paragraph("Section 4.2: Group Term Life Insurance", h2_style))
    story.append(Paragraph(
        "A Term Life Insurance benefit equal to three times (3x) the employee's Annual Fixed CTC is provided to all active permanent personnel at no cost to the employee.",
        body_style
    ))

    # Section 5
    story.append(Paragraph("5. General Code of Conduct and Disciplinary Norms", h1_style))
    story.append(Paragraph("Section 5.1: Working Hours & Remote Work", h2_style))
    story.append(Paragraph(
        "Standard core operational hours are from 9:30 AM to 6:00 PM IST, Monday through Friday. "
        "Hybrid work arrangements permit eligible staff to work from home up to two (2) days per week with supervisory approval.",
        body_style
    ))
    story.append(Paragraph("Section 5.2: Separation and Notice Period Post Confirmation", h2_style))
    story.append(Paragraph(
        "Upon confirmation, permanent employees desiring to resign must provide sixty (60) calendar days of formal notice or equivalent gross salary payout.",
        body_style
    ))

    doc.build(story)
    print(f"Successfully generated {filename} with complete HR policies and sections!")

if __name__ == "__main__":
    create_hr_manual("hr_manual.pdf")
