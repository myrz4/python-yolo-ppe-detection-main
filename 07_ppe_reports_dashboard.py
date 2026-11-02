"""
PPE Detection Reports Dashboard
Standalone reporting and analytics dashboard for PPE detection data
"""

import sys
import sqlite3
import os
from datetime import datetime, date, timedelta
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                               QHBoxLayout, QPushButton, QLabel, QComboBox, QTextEdit,
                               QGroupBox, QTableWidget, QTableWidgetItem, QDateEdit,
                               QTabWidget, QScrollArea, QMessageBox, QFileDialog)
from PySide6.QtCore import Qt, QDate
from PySide6.QtGui import QPixmap, QImage
import pytz
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (SimpleDocTemplate, Table, TableStyle, Paragraph,
                                Spacer, Image, PageBreak, KeepTogether)
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import cv2

# ==================== CONFIGURATION ====================
# Timezone Configuration
TIMEZONE = pytz.timezone('Asia/Kuala_Lumpur')

DATABASE_PATH = 'data/ppe_detection.db'
IMAGES_BASE_PATH = 'data/violations'
# =======================================================


class MplCanvas(FigureCanvas):
    """Matplotlib canvas for embedding charts in Qt"""

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)


class ReportsDatabase:
    """Handle database queries for reporting"""

    def __init__(self, db_path):
        self.db_path = db_path

    def get_violations_by_date_range(self, start_date, end_date, zone_name=None):
        """Get all violations within date range"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        if zone_name:
            cursor.execute('''
                SELECT id, timestamp, zone_name, violation_type, person_id, confidence, image_path
                FROM violations
                WHERE DATE(timestamp) BETWEEN ? AND ?
                AND zone_name = ?
                ORDER BY timestamp DESC
            ''', (start_date, end_date, zone_name))
        else:
            cursor.execute('''
                SELECT id, timestamp, zone_name, violation_type, person_id, confidence, image_path
                FROM violations
                WHERE DATE(timestamp) BETWEEN ? AND ?
                ORDER BY timestamp DESC
            ''', (start_date, end_date))

        results = cursor.fetchall()
        conn.close()
        return results

    def get_detections_by_date_range(self, start_date, end_date, zone_name=None):
        """Get all detections within date range"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        if zone_name:
            cursor.execute('''
                SELECT COUNT(*) as total,
                       SUM(CASE WHEN is_compliant = 1 THEN 1 ELSE 0 END) as compliant,
                       SUM(CASE WHEN is_compliant = 0 THEN 1 ELSE 0 END) as violations
                FROM detections
                WHERE DATE(timestamp) BETWEEN ? AND ?
                AND zone_name = ?
            ''', (start_date, end_date, zone_name))
        else:
            cursor.execute('''
                SELECT COUNT(*) as total,
                       SUM(CASE WHEN is_compliant = 1 THEN 1 ELSE 0 END) as compliant,
                       SUM(CASE WHEN is_compliant = 0 THEN 1 ELSE 0 END) as violations
                FROM detections
                WHERE DATE(timestamp) BETWEEN ? AND ?
            ''', (start_date, end_date))

        result = cursor.fetchone()
        conn.close()

        return {
            'total': result[0] or 0,
            'compliant': result[1] or 0,
            'violations': result[2] or 0
        }

    def get_daily_trends(self, start_date, end_date, zone_name=None):
        """Get daily violation trends"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        if zone_name:
            cursor.execute('''
                SELECT DATE(timestamp) as date, COUNT(*) as count
                FROM violations
                WHERE DATE(timestamp) BETWEEN ? AND ?
                AND zone_name = ?
                GROUP BY DATE(timestamp)
                ORDER BY date
            ''', (start_date, end_date, zone_name))
        else:
            cursor.execute('''
                SELECT DATE(timestamp) as date, COUNT(*) as count
                FROM violations
                WHERE DATE(timestamp) BETWEEN ? AND ?
                GROUP BY DATE(timestamp)
                ORDER BY date
            ''', (start_date, end_date))

        results = cursor.fetchall()
        conn.close()
        return results

    def get_violation_types_breakdown(self, start_date, end_date, zone_name=None):
        """Get breakdown of violation types"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        if zone_name:
            cursor.execute('''
                SELECT violation_type, COUNT(*) as count
                FROM violations
                WHERE DATE(timestamp) BETWEEN ? AND ?
                AND zone_name = ?
                GROUP BY violation_type
                ORDER BY count DESC
            ''', (start_date, end_date, zone_name))
        else:
            cursor.execute('''
                SELECT violation_type, COUNT(*) as count
                FROM violations
                WHERE DATE(timestamp) BETWEEN ? AND ?
                GROUP BY violation_type
                ORDER BY count DESC
            ''', (start_date, end_date))

        results = cursor.fetchall()
        conn.close()
        return results

    def get_hourly_distribution(self, start_date, end_date, zone_name=None):
        """Get hourly distribution of violations"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        if zone_name:
            cursor.execute('''
                SELECT strftime('%H', timestamp) as hour, COUNT(*) as count
                FROM violations
                WHERE DATE(timestamp) BETWEEN ? AND ?
                AND zone_name = ?
                GROUP BY hour
                ORDER BY hour
            ''', (start_date, end_date, zone_name))
        else:
            cursor.execute('''
                SELECT strftime('%H', timestamp) as hour, COUNT(*) as count
                FROM violations
                WHERE DATE(timestamp) BETWEEN ? AND ?
                GROUP BY hour
                ORDER BY hour
            ''', (start_date, end_date))

        results = cursor.fetchall()
        conn.close()
        return results

    def get_all_zones(self):
        """Get list of all monitoring zones"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT DISTINCT zone_name
            FROM violations
            ORDER BY zone_name
        ''')

        results = [row[0] for row in cursor.fetchall()]
        conn.close()
        return results

    def get_daily_stats_summary(self, start_date, end_date, zone_name=None):
        """Get summary from daily_stats table"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        if zone_name:
            cursor.execute('''
                SELECT SUM(total_people), SUM(compliant_people), SUM(violation_people),
                       AVG(compliance_rate)
                FROM daily_stats
                WHERE date BETWEEN ? AND ?
                AND zone_name = ?
            ''', (start_date, end_date, zone_name))
        else:
            cursor.execute('''
                SELECT SUM(total_people), SUM(compliant_people), SUM(violation_people),
                       AVG(compliance_rate)
                FROM daily_stats
                WHERE date BETWEEN ? AND ?
            ''', (start_date, end_date))

        result = cursor.fetchone()
        conn.close()

        if result and result[0]:
            return {
                'total_people': result[0] or 0,
                'compliant': result[1] or 0,
                'violations': result[2] or 0,
                'avg_compliance_rate': result[3] or 0
            }
        return None


class PDFReportGenerator:
    """Generate PDF reports"""

    def __init__(self):
        self.styles = getSampleStyleSheet()
        self.title_style = ParagraphStyle(
            'CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1a1a1a'),
            spaceAfter=30,
            alignment=TA_CENTER
        )
        self.heading_style = ParagraphStyle(
            'CustomHeading',
            parent=self.styles['Heading2'],
            fontSize=14,
            textColor=colors.HexColor('#2c3e50'),
            spaceAfter=12
        )

    def generate_report(self, output_path, data, start_date, end_date, zone_name):
        """Generate comprehensive PDF report"""
        doc = SimpleDocTemplate(output_path, pagesize=A4,
                               topMargin=0.5*inch, bottomMargin=0.5*inch)

        story = []

        # Title
        title = Paragraph("PPE SAFETY COMPLIANCE REPORT", self.title_style)
        story.append(title)
        story.append(Spacer(1, 0.3*inch))

        # Report metadata with local timezone
        local_now = datetime.now(TIMEZONE)
        metadata = [
            ["Report Period:", f"{start_date} to {end_date}"],
            ["Monitoring Zone:", zone_name or "All Zones"],
            ["Generated:", f"{local_now.strftime('%Y-%m-%d %H:%M:%S')} (Asia/Kuala_Lumpur)"]
        ]
        metadata_table = Table(metadata, colWidths=[2*inch, 4*inch])
        metadata_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.grey),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ]))
        story.append(metadata_table)
        story.append(Spacer(1, 0.3*inch))

        # Executive Summary
        story.append(Paragraph("EXECUTIVE SUMMARY", self.heading_style))

        stats = data['statistics']
        compliance_rate = (stats['compliant'] / stats['total'] * 100) if stats['total'] > 0 else 0

        summary_data = [
            ["Metric", "Value"],
            ["Total People Detected", str(stats['total'])],
            ["Compliant", f"{stats['compliant']} ({compliance_rate:.1f}%)"],
            ["Violations", f"{stats['violations']} ({100-compliance_rate:.1f}%)"],
            ["Compliance Rate", f"{compliance_rate:.2f}%"]
        ]

        summary_table = Table(summary_data, colWidths=[3*inch, 2*inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#34495e')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
        ]))
        story.append(summary_table)
        story.append(Spacer(1, 0.3*inch))

        # Violation Types Breakdown
        if data['violation_types']:
            story.append(Paragraph("TOP VIOLATION TYPES", self.heading_style))

            viol_data = [["Violation Type", "Count", "Percentage"]]
            total_violations = sum([count for _, count in data['violation_types']])

            for viol_type, count in data['violation_types'][:5]:  # Top 5
                percentage = (count / total_violations * 100) if total_violations > 0 else 0
                viol_data.append([viol_type, str(count), f"{percentage:.1f}%"])

            viol_table = Table(viol_data, colWidths=[3*inch, 1*inch, 1.5*inch])
            viol_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e74c3c')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 11),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 1), (-1, -1), 10),
            ]))
            story.append(viol_table)
            story.append(Spacer(1, 0.3*inch))

        # Charts
        if data['charts']:
            story.append(PageBreak())
            story.append(Paragraph("ANALYTICS & TRENDS", self.heading_style))

            for chart_path in data['charts']:
                if os.path.exists(chart_path):
                    img = Image(chart_path, width=6*inch, height=4*inch)
                    story.append(img)
                    story.append(Spacer(1, 0.2*inch))

        # Violation Log
        if data['violations']:
            story.append(PageBreak())
            story.append(Paragraph("DETAILED VIOLATION LOG", self.heading_style))

            log_data = [["Timestamp", "Zone", "Violation Type", "Person ID"]]

            for violation in data['violations'][:20]:  # Limit to 20 most recent
                timestamp = violation[1]
                zone = violation[2]
                viol_type = violation[3]
                person_id = violation[4]

                log_data.append([
                    timestamp.split('.')[0],  # Remove microseconds
                    zone,
                    viol_type[:30],  # Truncate long messages
                    str(person_id)
                ])

            log_table = Table(log_data, colWidths=[2*inch, 1.5*inch, 2.5*inch, 0.8*inch])
            log_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498db')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 1), (-1, -1), 8),
            ]))
            story.append(log_table)

            if len(data['violations']) > 20:
                story.append(Spacer(1, 0.1*inch))
                story.append(Paragraph(f"<i>Showing 20 of {len(data['violations'])} violations. Full data available in database.</i>",
                                      self.styles['Normal']))

        # Sample violation images
        if data['sample_images']:
            story.append(PageBreak())
            story.append(Paragraph("SAMPLE VIOLATION IMAGES", self.heading_style))

            for i, img_path in enumerate(data['sample_images'][:3]):  # Max 3 images
                if os.path.exists(img_path):
                    try:
                        img = Image(img_path, width=5*inch, height=3.5*inch)
                        story.append(img)
                        story.append(Spacer(1, 0.1*inch))
                        story.append(Paragraph(f"<i>Violation Image {i+1}</i>", self.styles['Normal']))
                        story.append(Spacer(1, 0.2*inch))
                    except:
                        pass

        # Footer
        story.append(Spacer(1, 0.3*inch))
        story.append(Paragraph("<i>Generated by PPE Detection System</i>",
                              ParagraphStyle('Footer', parent=self.styles['Normal'],
                                           fontSize=8, textColor=colors.grey,
                                           alignment=TA_CENTER)))

        # Build PDF
        doc.build(story)


class ReportsDashboard(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PPE Detection Reports Dashboard")
        self.setGeometry(100, 100, 1400, 900)

        # Initialize database
        if not os.path.exists(DATABASE_PATH):
            QMessageBox.warning(self, "Database Not Found",
                               f"Database not found at {DATABASE_PATH}\n"
                               "Please run detection system first to generate data.")

        self.db = ReportsDatabase(DATABASE_PATH)

        # Setup UI
        self.setup_ui()

        # Load initial data
        self.load_data()

    def setup_ui(self):
        """Setup the user interface"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)

        # Header
        header_label = QLabel("<h1>PPE Detection Reports & Analytics</h1>")
        header_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(header_label)

        # Filters
        filter_group = QGroupBox("Report Filters")
        filter_layout = QHBoxLayout()

        # Date range preset
        filter_layout.addWidget(QLabel("Quick Select:"))
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(["Today", "Yesterday", "Last 7 Days", "Last 30 Days",
                                    "This Month", "Last Month", "This Year", "Custom"])
        self.preset_combo.currentTextChanged.connect(self.on_preset_changed)
        filter_layout.addWidget(self.preset_combo)

        # Date range
        filter_layout.addWidget(QLabel("From:"))
        self.start_date_edit = QDateEdit()
        self.start_date_edit.setCalendarPopup(True)
        self.start_date_edit.setDate(QDate.currentDate())
        filter_layout.addWidget(self.start_date_edit)

        filter_layout.addWidget(QLabel("To:"))
        self.end_date_edit = QDateEdit()
        self.end_date_edit.setCalendarPopup(True)
        self.end_date_edit.setDate(QDate.currentDate())
        filter_layout.addWidget(self.end_date_edit)

        # Zone filter
        filter_layout.addWidget(QLabel("Zone:"))
        self.zone_combo = QComboBox()
        self.zone_combo.addItem("All Zones", None)
        filter_layout.addWidget(self.zone_combo)

        # Apply button
        self.apply_btn = QPushButton("Apply Filters")
        self.apply_btn.clicked.connect(self.load_data)
        filter_layout.addWidget(self.apply_btn)

        # Export PDF button
        self.export_pdf_btn = QPushButton("Export PDF Report")
        self.export_pdf_btn.clicked.connect(self.export_pdf)
        filter_layout.addWidget(self.export_pdf_btn)

        filter_layout.addStretch()
        filter_group.setLayout(filter_layout)
        main_layout.addWidget(filter_group)

        # Tabs
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        # Tab 1: Statistics Overview
        self.stats_tab = QWidget()
        self.setup_stats_tab()
        self.tabs.addTab(self.stats_tab, "Statistics Overview")

        # Tab 2: Analytics & Charts
        self.analytics_tab = QWidget()
        self.setup_analytics_tab()
        self.tabs.addTab(self.analytics_tab, "Analytics & Charts")

        # Tab 3: Violation Log
        self.violations_tab = QWidget()
        self.setup_violations_tab()
        self.tabs.addTab(self.violations_tab, "Violation Log")

        # Tab 4: Image Gallery
        self.gallery_tab = QWidget()
        self.setup_gallery_tab()
        self.tabs.addTab(self.gallery_tab, "Image Gallery")

        # Load zones
        self.load_zones()

    def setup_stats_tab(self):
        """Setup statistics overview tab"""
        layout = QVBoxLayout()

        # Summary cards
        cards_layout = QHBoxLayout()

        # Total people card
        self.total_people_label = QLabel()
        self.total_people_label.setStyleSheet("""
            QLabel {
                background-color: #3498db;
                color: white;
                padding: 20px;
                border-radius: 10px;
                font-size: 16px;
                font-weight: bold;
            }
        """)
        self.total_people_label.setAlignment(Qt.AlignCenter)
        cards_layout.addWidget(self.total_people_label)

        # Compliant card
        self.compliant_label = QLabel()
        self.compliant_label.setStyleSheet("""
            QLabel {
                background-color: #27ae60;
                color: white;
                padding: 20px;
                border-radius: 10px;
                font-size: 16px;
                font-weight: bold;
            }
        """)
        self.compliant_label.setAlignment(Qt.AlignCenter)
        cards_layout.addWidget(self.compliant_label)

        # Violations card
        self.violations_label = QLabel()
        self.violations_label.setStyleSheet("""
            QLabel {
                background-color: #e74c3c;
                color: white;
                padding: 20px;
                border-radius: 10px;
                font-size: 16px;
                font-weight: bold;
            }
        """)
        self.violations_label.setAlignment(Qt.AlignCenter)
        cards_layout.addWidget(self.violations_label)

        # Compliance rate card
        self.compliance_rate_label = QLabel()
        self.compliance_rate_label.setStyleSheet("""
            QLabel {
                background-color: #9b59b6;
                color: white;
                padding: 20px;
                border-radius: 10px;
                font-size: 16px;
                font-weight: bold;
            }
        """)
        self.compliance_rate_label.setAlignment(Qt.AlignCenter)
        cards_layout.addWidget(self.compliance_rate_label)

        layout.addLayout(cards_layout)

        # Detailed statistics table
        details_group = QGroupBox("Detailed Breakdown")
        details_layout = QVBoxLayout()

        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        details_layout.addWidget(self.stats_text)

        details_group.setLayout(details_layout)
        layout.addWidget(details_group)

        self.stats_tab.setLayout(layout)

    def setup_analytics_tab(self):
        """Setup analytics and charts tab"""
        layout = QVBoxLayout()

        # Chart canvases
        self.trend_canvas = MplCanvas(self, width=10, height=4, dpi=100)
        layout.addWidget(self.trend_canvas)

        charts_layout = QHBoxLayout()
        self.violation_types_canvas = MplCanvas(self, width=5, height=4, dpi=100)
        charts_layout.addWidget(self.violation_types_canvas)

        self.hourly_canvas = MplCanvas(self, width=5, height=4, dpi=100)
        charts_layout.addWidget(self.hourly_canvas)

        layout.addLayout(charts_layout)

        self.analytics_tab.setLayout(layout)

    def setup_violations_tab(self):
        """Setup violations log tab"""
        layout = QVBoxLayout()

        self.violations_table = QTableWidget()
        self.violations_table.setColumnCount(7)
        self.violations_table.setHorizontalHeaderLabels([
            "ID", "Timestamp", "Zone", "Violation Type", "Person ID", "Confidence", "Image"
        ])
        self.violations_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.violations_table.setSelectionBehavior(QTableWidget.SelectRows)

        layout.addWidget(self.violations_table)

        self.violations_tab.setLayout(layout)

    def setup_gallery_tab(self):
        """Setup image gallery tab"""
        layout = QVBoxLayout()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)

        self.gallery_widget = QWidget()
        self.gallery_layout = QVBoxLayout()
        self.gallery_widget.setLayout(self.gallery_layout)

        scroll.setWidget(self.gallery_widget)
        layout.addWidget(scroll)

        self.gallery_tab.setLayout(layout)

    def load_zones(self):
        """Load monitoring zones"""
        try:
            zones = self.db.get_all_zones()
            for zone in zones:
                self.zone_combo.addItem(zone, zone)
        except Exception as e:
            print(f"Error loading zones: {str(e)}")

    def on_preset_changed(self, preset):
        """Handle preset date range selection"""
        today = QDate.currentDate()

        if preset == "Today":
            self.start_date_edit.setDate(today)
            self.end_date_edit.setDate(today)
        elif preset == "Yesterday":
            yesterday = today.addDays(-1)
            self.start_date_edit.setDate(yesterday)
            self.end_date_edit.setDate(yesterday)
        elif preset == "Last 7 Days":
            self.start_date_edit.setDate(today.addDays(-7))
            self.end_date_edit.setDate(today)
        elif preset == "Last 30 Days":
            self.start_date_edit.setDate(today.addDays(-30))
            self.end_date_edit.setDate(today)
        elif preset == "This Month":
            self.start_date_edit.setDate(QDate(today.year(), today.month(), 1))
            self.end_date_edit.setDate(today)
        elif preset == "Last Month":
            last_month = today.addMonths(-1)
            self.start_date_edit.setDate(QDate(last_month.year(), last_month.month(), 1))
            self.end_date_edit.setDate(QDate(today.year(), today.month(), 1).addDays(-1))
        elif preset == "This Year":
            self.start_date_edit.setDate(QDate(today.year(), 1, 1))
            self.end_date_edit.setDate(today)

    def load_data(self):
        """Load data from database and update UI"""
        start_date = self.start_date_edit.date().toString("yyyy-MM-dd")
        end_date = self.end_date_edit.date().toString("yyyy-MM-dd")
        zone_name = self.zone_combo.currentData()

        try:
            # Get statistics
            stats = self.db.get_detections_by_date_range(start_date, end_date, zone_name)

            # Update summary cards
            self.update_statistics(stats)

            # Get violations
            violations = self.db.get_violations_by_date_range(start_date, end_date, zone_name)
            self.update_violations_table(violations)

            # Get analytics data
            trends = self.db.get_daily_trends(start_date, end_date, zone_name)
            violation_types = self.db.get_violation_types_breakdown(start_date, end_date, zone_name)
            hourly_dist = self.db.get_hourly_distribution(start_date, end_date, zone_name)

            # Update charts
            self.update_charts(trends, violation_types, hourly_dist)

            # Update image gallery
            self.update_gallery(violations)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error loading data: {str(e)}")

    def update_statistics(self, stats):
        """Update statistics display"""
        total = stats['total']
        compliant = stats['compliant']
        violations = stats['violations']
        compliance_rate = (compliant / total * 100) if total > 0 else 0

        self.total_people_label.setText(f"Total People\n{total}")
        self.compliant_label.setText(f"Compliant\n{compliant}")
        self.violations_label.setText(f"Violations\n{violations}")
        self.compliance_rate_label.setText(f"Compliance Rate\n{compliance_rate:.2f}%")

        # Detailed text
        details = f"""
<h3>Summary Statistics</h3>
<table width='100%'>
<tr><td><b>Total People Detected:</b></td><td>{total}</td></tr>
<tr><td><b>Compliant:</b></td><td>{compliant} ({compliance_rate:.2f}%)</td></tr>
<tr><td><b>Violations:</b></td><td>{violations} ({100-compliance_rate:.2f}%)</td></tr>
<tr><td><b>Compliance Rate:</b></td><td>{compliance_rate:.2f}%</td></tr>
</table>
        """
        self.stats_text.setHtml(details)

    def update_violations_table(self, violations):
        """Update violations table"""
        self.violations_table.setRowCount(len(violations))

        for row, violation in enumerate(violations):
            self.violations_table.setItem(row, 0, QTableWidgetItem(str(violation[0])))
            self.violations_table.setItem(row, 1, QTableWidgetItem(violation[1]))
            self.violations_table.setItem(row, 2, QTableWidgetItem(violation[2]))
            self.violations_table.setItem(row, 3, QTableWidgetItem(violation[3]))
            self.violations_table.setItem(row, 4, QTableWidgetItem(str(violation[4])))
            self.violations_table.setItem(row, 5, QTableWidgetItem(f"{violation[5]:.2f}"))
            self.violations_table.setItem(row, 6, QTableWidgetItem(violation[6] if violation[6] else "N/A"))

        self.violations_table.resizeColumnsToContents()

    def update_charts(self, trends, violation_types, hourly_dist):
        """Update all charts"""
        # Trend chart
        self.trend_canvas.axes.clear()
        if trends:
            dates = [t[0] for t in trends]
            counts = [t[1] for t in trends]
            self.trend_canvas.axes.plot(dates, counts, marker='o', linewidth=2, markersize=6)
            self.trend_canvas.axes.set_xlabel('Date')
            self.trend_canvas.axes.set_ylabel('Violations')
            self.trend_canvas.axes.set_title('Daily Violation Trends')
            self.trend_canvas.axes.grid(True, alpha=0.3)
            # Rotate x-axis labels for better readability
            for label in self.trend_canvas.axes.get_xticklabels():
                label.set_rotation(45)
                label.set_ha('right')
        else:
            self.trend_canvas.axes.text(0.5, 0.5, 'No data available',
                                       ha='center', va='center', fontsize=12)
        self.trend_canvas.draw()

        # Violation types pie chart
        self.violation_types_canvas.axes.clear()
        if violation_types:
            labels = [v[0] for v in violation_types[:5]]  # Top 5
            sizes = [v[1] for v in violation_types[:5]]
            colors_pie = ['#e74c3c', '#f39c12', '#f1c40f', '#16a085', '#3498db']
            self.violation_types_canvas.axes.pie(sizes, labels=labels, autopct='%1.1f%%',
                                                colors=colors_pie, startangle=90)
            self.violation_types_canvas.axes.set_title('Top Violation Types')
        else:
            self.violation_types_canvas.axes.text(0.5, 0.5, 'No data available',
                                                 ha='center', va='center', fontsize=12)
        self.violation_types_canvas.draw()

        # Hourly distribution bar chart
        self.hourly_canvas.axes.clear()
        if hourly_dist:
            hours = [int(h[0]) for h in hourly_dist]
            counts = [h[1] for h in hourly_dist]
            self.hourly_canvas.axes.bar(hours, counts, color='#3498db', alpha=0.7)
            self.hourly_canvas.axes.set_xlabel('Hour of Day')
            self.hourly_canvas.axes.set_ylabel('Violations')
            self.hourly_canvas.axes.set_title('Hourly Distribution')
            self.hourly_canvas.axes.set_xticks(range(24))
            self.hourly_canvas.axes.grid(True, alpha=0.3, axis='y')
        else:
            self.hourly_canvas.axes.text(0.5, 0.5, 'No data available',
                                        ha='center', va='center', fontsize=12)
        self.hourly_canvas.draw()

    def update_gallery(self, violations):
        """Update image gallery"""
        # Clear existing gallery
        while self.gallery_layout.count():
            child = self.gallery_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        # Add images
        images_added = 0
        for violation in violations[:10]:  # Show max 10 images
            image_path = violation[6]
            if image_path and os.path.exists(image_path):
                try:
                    # Create image widget
                    img_label = QLabel()
                    pixmap = QPixmap(image_path)
                    scaled_pixmap = pixmap.scaled(600, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    img_label.setPixmap(scaled_pixmap)
                    img_label.setAlignment(Qt.AlignCenter)

                    # Create info label
                    info_text = f"<b>Timestamp:</b> {violation[1]}<br>"
                    info_text += f"<b>Zone:</b> {violation[2]}<br>"
                    info_text += f"<b>Violation:</b> {violation[3]}"
                    info_label = QLabel(info_text)

                    # Add to gallery
                    self.gallery_layout.addWidget(img_label)
                    self.gallery_layout.addWidget(info_label)
                    self.gallery_layout.addWidget(QLabel("<hr>"))

                    images_added += 1
                except Exception as e:
                    print(f"Error loading image {image_path}: {str(e)}")

        if images_added == 0:
            no_images_label = QLabel("No violation images available")
            no_images_label.setAlignment(Qt.AlignCenter)
            self.gallery_layout.addWidget(no_images_label)

        self.gallery_layout.addStretch()

    def export_pdf(self):
        """Export report to PDF"""
        # Get file path from user
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save PDF Report", "", "PDF Files (*.pdf)"
        )

        if not file_path:
            return

        try:
            start_date = self.start_date_edit.date().toString("yyyy-MM-dd")
            end_date = self.end_date_edit.date().toString("yyyy-MM-dd")
            zone_name = self.zone_combo.currentData()

            # Gather data for PDF
            stats = self.db.get_detections_by_date_range(start_date, end_date, zone_name)
            violations = self.db.get_violations_by_date_range(start_date, end_date, zone_name)
            violation_types = self.db.get_violation_types_breakdown(start_date, end_date, zone_name)

            # Save chart images temporarily
            chart_paths = []
            temp_chart1 = 'temp_trend_chart.png'
            temp_chart2 = 'temp_viol_types_chart.png'
            temp_chart3 = 'temp_hourly_chart.png'

            self.trend_canvas.figure.savefig(temp_chart1, bbox_inches='tight', dpi=150)
            chart_paths.append(temp_chart1)

            self.violation_types_canvas.figure.savefig(temp_chart2, bbox_inches='tight', dpi=150)
            chart_paths.append(temp_chart2)

            self.hourly_canvas.figure.savefig(temp_chart3, bbox_inches='tight', dpi=150)
            chart_paths.append(temp_chart3)

            # Get sample violation images
            sample_images = [v[6] for v in violations[:3] if v[6] and os.path.exists(v[6])]

            # Prepare data for PDF generator
            pdf_data = {
                'statistics': stats,
                'violations': violations,
                'violation_types': violation_types,
                'charts': chart_paths,
                'sample_images': sample_images
            }

            # Generate PDF
            pdf_gen = PDFReportGenerator()
            pdf_gen.generate_report(
                file_path,
                pdf_data,
                start_date,
                end_date,
                zone_name or "All Zones"
            )

            # Clean up temporary chart files
            for chart_path in chart_paths:
                if os.path.exists(chart_path):
                    os.remove(chart_path)

            QMessageBox.information(self, "Success",
                                   f"PDF report generated successfully!\n{file_path}")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error generating PDF: {str(e)}")


def main():
    app = QApplication(sys.argv)
    window = ReportsDashboard()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
