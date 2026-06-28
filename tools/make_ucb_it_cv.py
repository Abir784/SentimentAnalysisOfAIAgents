from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


OUTPUT = "outputs/Md_Abir_Hossain_UCB_IT_Intern_CV.pdf"


def p(text, style):
    return Paragraph(text, style)


def section(title, styles):
    return [
        Spacer(1, 8),
        Paragraph(title, styles["Section"]),
        Spacer(1, 4),
    ]


def bullets(items, styles):
    rows = []
    for item in items:
        rows.append([
            Paragraph("•", styles["BulletMark"]),
            Paragraph(item, styles["Body"]),
        ])
    table = Table(rows, colWidths=[0.16 * inch, 6.05 * inch], hAlign="LEFT")
    table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 1),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 1),
    ]))
    return table


def build():
    doc = SimpleDocTemplate(
        OUTPUT,
        pagesize=A4,
        rightMargin=0.65 * inch,
        leftMargin=0.65 * inch,
        topMargin=0.55 * inch,
        bottomMargin=0.55 * inch,
        title="Md. Abir Hossain - UCB IT Intern CV",
        author="Md. Abir Hossain",
    )

    base = getSampleStyleSheet()
    styles = {
        "Name": ParagraphStyle(
            "Name",
            parent=base["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=17,
            leading=20,
            alignment=TA_CENTER,
            spaceAfter=2,
        ),
        "Contact": ParagraphStyle(
            "Contact",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=8.8,
            leading=11,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#222222"),
        ),
        "Section": ParagraphStyle(
            "Section",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=10.5,
            leading=12.5,
            textColor=colors.HexColor("#111111"),
            borderWidth=0,
            borderPadding=0,
            spaceAfter=2,
        ),
        "Body": ParagraphStyle(
            "Body",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=8.7,
            leading=11.1,
            textColor=colors.HexColor("#222222"),
        ),
        "BodySmall": ParagraphStyle(
            "BodySmall",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=8.4,
            leading=10.6,
            textColor=colors.HexColor("#222222"),
        ),
        "Role": ParagraphStyle(
            "Role",
            parent=base["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=9.0,
            leading=11.3,
            textColor=colors.HexColor("#111111"),
        ),
        "Meta": ParagraphStyle(
            "Meta",
            parent=base["BodyText"],
            fontName="Helvetica-Oblique",
            fontSize=8.4,
            leading=10.5,
            textColor=colors.HexColor("#444444"),
        ),
        "BulletMark": ParagraphStyle(
            "BulletMark",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=8.7,
            leading=11.1,
        ),
    }

    story = []
    story.append(p("Md. Abir Hossain", styles["Name"]))
    story.append(p(
        "Dhaka, Bangladesh | 01400554400 | "
        '<a href="mailto:abirhossainofficial784@gmail.com">abirhossainofficial784@gmail.com</a> | '
        '<a href="https://github.com/Abir784">github.com/Abir784</a> | '
        '<a href="https://linkedin.com/in/mdabirhossainabir">linkedin.com/in/mdabirhossainabir</a>',
        styles["Contact"],
    ))

    story += section("PROFILE", styles)
    story.append(p(
        "Computer Science and Engineering graduate from BRAC University (CGPA 3.87/4.00, Highest Distinction) with hands-on experience in "
        "software troubleshooting, database-backed applications, Linux/Git workflows, secure authentication, and technical "
        "user support. Strong foundation in operating systems, networking concepts, MySQL/SQL, Python, PHP/Laravel, and "
        "Microsoft Office. Interested in IT operations, system support, infrastructure maintenance, and secure technology "
        "support within a professional financial services environment.",
        styles["Body"],
    ))

    story += section("TECHNICAL SKILLS", styles)
    skill_rows = [
        ["IT Support", "Hardware/software troubleshooting, desktop and application issue diagnosis, user support, documentation"],
        ["Operating Systems", "Windows, Linux, command line basics, application installation/configuration, system maintenance awareness"],
        ["Networking", "TCP/IP, DNS, DHCP, LAN/Wi-Fi concepts, basic network troubleshooting and connectivity checks"],
        ["Databases", "MySQL, SQL joins, indexing basics, schema design, query optimization, database monitoring concepts"],
        ["Programming", "Python, PHP, JavaScript, C/C++, SQL"],
        ["Web & Tools", "Laravel, RESTful APIs, Git, VS Code, Jupyter Notebook, Bootstrap, Tailwind CSS"],
        ["Office & Reporting", "Microsoft Office, Excel, PowerPoint, technical documentation, data reporting"],
        ["Security Awareness", "Secure authentication, access control, responsible handling of confidential information"],
    ]
    table = Table(
        [[Paragraph(f"<b>{k}</b>", styles["BodySmall"]), Paragraph(v, styles["BodySmall"])] for k, v in skill_rows],
        colWidths=[1.35 * inch, 4.85 * inch],
        hAlign="LEFT",
    )
    table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 1.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5),
    ]))
    story.append(table)

    story += section("EDUCATION", styles)
    story.append(p("<b>Bachelor of Science in Computer Science and Engineering</b> | BRAC University, Dhaka | 2022 - 2026", styles["Body"]))
    story.append(p("CGPA: 3.87 / 4.00, Highest Distinction", styles["BodySmall"]))
    story.append(Spacer(1, 2))
    story.append(p("<b>Higher Secondary Certificate (Science)</b> | St. Gregory's High School and College, Dhaka | 2018 - 2020", styles["Body"]))
    story.append(p("GPA: 4.83", styles["BodySmall"]))

    story += section("WORK EXPERIENCE", styles)
    story.append(p("Student Tutor | BRAC University, Dhaka | June 2025 - January 2026", styles["Role"]))
    story.append(bullets([
        "Mentored 20+ undergraduate students in Data Structures and Algorithms through structured troubleshooting and problem-solving sessions.",
        "Supported students in debugging logic errors, understanding technical concepts, and building stronger analytical habits.",
        "Prepared targeted practice materials and communicated recurring issues to faculty, strengthening documentation and user-support skills.",
    ], styles))

    story += section("SELECTED IT / SOFTWARE PROJECTS", styles)
    story.append(p("laravel-query-watchdog - Database Monitoring Package | GitHub: github.com/Abir784/laravel-query-watchdog", styles["Role"]))
    story.append(bullets([
        "Built a Laravel package for monitoring slow database queries, SQL fingerprinting, N+1 burst detection, percentile aggregation, and version-based regression detection.",
        "Implemented a secured live dashboard, configurable retention/pruning, multi-channel alerts, and PHPUnit test coverage.",
        "Relevant to IT operations: database monitoring, issue detection, logging, documentation, and operational reliability.",
    ], styles))

    story.append(Spacer(1, 3))
    story.append(p("ClydePixel - Secure Client Platform | order.clydepixel.com", styles["Role"]))
    story.append(bullets([
        "Contributed Laravel/PHP backend logic and encryption mechanisms for secure client-side login authentication.",
        "Performed UI testing and debugging to improve stability and reduce user-facing issues across the platform.",
    ], styles))

    story.append(Spacer(1, 3))
    story.append(p("FosterPet - Full-Stack Web Platform | GitHub: github.com/Abir784/FosterPet", styles["Role"]))
    story.append(bullets([
        "Developed multi-role registration, authentication, data management, and request workflows using Laravel, PHP, MySQL, Tailwind CSS, Blade, Vite, and JavaScript.",
        "Sustained iterative development across 137 commits using Git, showing version-controlled delivery and maintainable project practice.",
    ], styles))

    story.append(Spacer(1, 3))
    story.append(p("Sentiment Analysis of AI Agents - Python Data Pipeline | GitHub: github.com/Abir784/SentimentAnalysisOfAIAgents", styles["Role"]))
    story.append(bullets([
        "Built a reproducible Python pipeline for data cleaning, model evaluation, reporting, and a Streamlit dashboard.",
        "Used Git, Python, Scikit-learn, Pandas, Matplotlib, Seaborn, and structured documentation for reproducible technical work.",
    ], styles))

    story += section("TRAINING", styles)
    story.append(p("<b>Web Development in Laravel</b> | Creative IT Institute, Dhaka | 2021 (6-Month Program)", styles["Body"]))
    story.append(bullets([
        "Built dynamic web applications using Laravel, PHP, MVC architecture, RESTful APIs, MySQL, authentication, and form validation.",
    ], styles))

    story += section("KEY STRENGTHS", styles)
    story.append(bullets([
        "Strong analytical and problem-solving capability with practical experience debugging applications and database-backed systems.",
        "Comfortable supporting users, documenting technical issues, and communicating clearly with technical and non-technical stakeholders.",
        "Fast learner with high interest in IT operations, system support, infrastructure management, and secure financial technology environments.",
        "Responsible with confidential information and careful about access control, authentication, and data handling.",
    ], styles))

    story += section("LANGUAGES", styles)
    story.append(p("Bangla (Native) | English (Intermediate)", styles["Body"]))

    doc.build(story)


if __name__ == "__main__":
    build()
