import os
import sys
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
)
from reportlab.pdfgen import canvas

class NumberedCanvas(canvas.Canvas):
    """
    Two-pass canvas for exact page numbering ('Page X of 4') and clean headers/footers.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_decorations(self, num_pages):
        p = self._pageNumber
        w, h = letter

        # Running Header on Pages 2, 3, 4
        if p > 1:
            self.saveState()
            self.setFont("Helvetica-Bold", 8)
            self.setFillColor(colors.HexColor("#1A365D"))
            self.drawString(36, h - 24, "GUARD-TRAJECTORY: FACULTY PROJECT PROGRESS REPORT")
            self.setFont("Helvetica", 8)
            self.setFillColor(colors.HexColor("#64748B"))
            self.drawRightString(w - 36, h - 24, "Department of Computer Science | October 2026")
            self.setStrokeColor(colors.HexColor("#CBD5E1"))
            self.setLineWidth(0.6)
            self.line(36, h - 28, w - 36, h - 28)
            self.restoreState()

        # Running Footer on All Pages
        self.saveState()
        self.setStrokeColor(colors.HexColor("#CBD5E1"))
        self.setLineWidth(0.6)
        self.line(36, 28, w - 36, 28)

        self.setFont("Helvetica-Bold", 7.8)
        self.setFillColor(colors.HexColor("#334155"))
        self.drawString(36, 17, "Students: Palli Pavan Kumar, Rithvik")

        self.setFont("Helvetica", 7.8)
        self.setFillColor(colors.HexColor("#64748B"))
        self.drawCentredString(w / 2.0, 17, "Sequence-Wise Academic Progress & Empirical Verification Dossier")

        self.setFont("Helvetica-Bold", 7.8)
        self.setFillColor(colors.HexColor("#1A365D"))
        self.drawRightString(w - 36, 17, f"Page {p} of {num_pages}")
        self.restoreState()


def build_sequence_faculty_report(output_pdf="Guard_Trajectory_Faculty_Progress_Report.pdf"):
    doc = SimpleDocTemplate(
        output_pdf,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=32,
        bottomMargin=32
    )

    printable_w = letter[0] - 72  # 540 pt
    story = []

    # Palette
    c_navy = colors.HexColor("#1A365D")
    c_blue = colors.HexColor("#1D4ED8")
    c_slate = colors.HexColor("#1E293B")
    c_muted = colors.HexColor("#64748B")
    c_card_bg = colors.HexColor("#F8FAFC")
    c_alt_row = colors.HexColor("#F1F5F9")
    c_border = colors.HexColor("#CBD5E1")
    c_green = colors.HexColor("#16A34A")

    styles = getSampleStyleSheet()

    t_title = ParagraphStyle(
        'DocTitle', parent=styles['Normal'],
        fontName='Helvetica-Bold', fontSize=17, leading=20, textColor=c_navy, spaceAfter=2
    )
    t_sub = ParagraphStyle(
        'DocSub', parent=styles['Normal'],
        fontName='Helvetica-Bold', fontSize=10, leading=13, textColor=c_blue, spaceAfter=4
    )
    t_meta = ParagraphStyle(
        'DocMeta', parent=styles['Normal'],
        fontName='Helvetica', fontSize=8.2, leading=11.5, textColor=c_muted
    )
    t_h1 = ParagraphStyle(
        'SecH1', parent=styles['Normal'],
        fontName='Helvetica-Bold', fontSize=11, leading=14, textColor=c_navy,
        spaceBefore=8, spaceAfter=3.5, keepWithNext=True
    )
    t_h2 = ParagraphStyle(
        'SecH2', parent=styles['Normal'],
        fontName='Helvetica-Bold', fontSize=9.2, leading=12, textColor=c_blue,
        spaceBefore=5, spaceAfter=2.5, keepWithNext=True
    )
    t_body = ParagraphStyle(
        'BodyTxt', parent=styles['Normal'],
        fontName='Helvetica', fontSize=8.6, leading=11.8, textColor=c_slate, spaceAfter=3.5
    )
    t_bullet = ParagraphStyle(
        'BulletTxt', parent=t_body, leftIndent=11, firstLineIndent=-7, spaceAfter=2.5
    )
    t_callout = ParagraphStyle(
        'CalloutTxt', parent=styles['Normal'],
        fontName='Helvetica', fontSize=8.4, leading=11.4, textColor=c_slate
    )
    t_caption = ParagraphStyle(
        'CaptTxt', parent=styles['Normal'],
        fontName='Helvetica-Oblique', fontSize=7.4, leading=9.5, textColor=c_muted, alignment=1, spaceAfter=3
    )
    th_style = ParagraphStyle(
        'TH', parent=styles['Normal'],
        fontName='Helvetica-Bold', fontSize=7.6, leading=9.5, textColor=colors.white, alignment=1
    )
    tc_style = ParagraphStyle(
        'TC', parent=styles['Normal'],
        fontName='Helvetica', fontSize=7.5, leading=9.5, textColor=c_slate
    )
    tc_bold = ParagraphStyle('TCB', parent=tc_style, fontName='Helvetica-Bold')
    tc_center = ParagraphStyle('TCC', parent=tc_style, alignment=1)

    # =========================================================================
    # PAGE 1: SEQUENCE 1, 2, 3 — INTRODUCTION, OBJECTIVES & METHODOLOGY
    # =========================================================================
    story.append(Paragraph("GUARD-TRAJECTORY: Project Progress & Reach Report", t_title))
    story.append(Paragraph("Training-Free Real-Time Safety Monitoring and Interception in Discrete Diffusion Language Models", t_sub))
    story.append(Paragraph(
        "<b>Student Researchers:</b> Palli Pavan Kumar, Rithvik &nbsp;|&nbsp; "
        "<b>Department:</b> Computer Science &nbsp;|&nbsp; "
        "<b>Date:</b> October 2026<br/>"
        "<b>Academic Review Stage:</b> Empirical Milestone Completion & Faculty Progress Evaluation",
        t_meta
    ))
    story.append(Spacer(1, 5))

    # Executive Overview Card
    exec_summary = (
        "<b>Executive Summary:</b> This report presents the sequence-wise progress, scientific reach, and empirical outputs "
        "of our research project: <b>Guard-Trajectory</b>. While conventional Autoregressive models (GPT/LLaMA) generate tokens "
        "left-to-right with no ability to reverse unsafe tokens without throwing away the whole sequence, <b>Discrete Masked Diffusion "
        "Language Models (MDLM)</b> unmask text iteratively over reverse timesteps. We introduce a training-free monitoring "
        "hook that inspects candidate tokens <i>during generation</i> and steers them into safe basins. We have successfully implemented "
        "the full pipeline, benchmarked 7 policies across 2,485 trajectories on consumer GPU hardware, and achieved early detection "
        "with 0.932 AUROC and 13.4 steps lead time."
    )
    tbl_exec = Table([[Paragraph(exec_summary, t_callout)]], colWidths=[printable_w])
    tbl_exec.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#F0FDF4")),
        ('BOX', (0,0), (-1,-1), 0.8, colors.HexColor("#86EFAC")),
        ('LINEBEFORE', (0,0), (0,-1), 3.5, colors.HexColor("#16A34A")),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('LEFTPADDING', (0,0), (-1,-1), 7),
        ('RIGHTPADDING', (0,0), (-1,-1), 7),
    ]))
    story.append(tbl_exec)
    story.append(Spacer(1, 5))

    # Sequence 1: Problem Statement
    story.append(Paragraph("1. Problem Statement & Research Motivation", t_h1))
    story.append(Paragraph(
        "<b>The Irreversible Generation Problem in Autoregressive Models:</b> Modern safety guardrails for Autoregressive (AR) LLMs "
        "(such as GPT and LLaMA) suffer from a fundamental architectural limitation: tokens are generated causal-unidirectionally. "
        "Once an unsafe token is committed to the KV-cache, the model cannot steer backward without discarding the entire sequence. "
        "Consequently, post-hoc reject filters discard 100% of the decoding compute.<br/>"
        "<b>The Opportunity in Discrete Diffusion Models:</b> Masked Discrete Diffusion Language Models (MDLM) generate text through iterative "
        "bidirectional de-masking over reverse steps (t = 1.0 &rarr; 0.0). At every step, the predicted clean candidate sequence "
        "x&#770;<sub>0</sub><sup>(t)</sup> is exposed before tokens finalize. This creates a powerful opportunity to inspect and steer "
        "diffusion trajectories in real time <b>without retraining model weights</b>.",
        t_body
    ))
    story.append(Spacer(1, 3))

    # Sequence 2: Objectives & Expected Reach
    story.append(Paragraph("2. Project Objectives & The Target 'Reach'", t_h1))
    story.append(Paragraph("<b>&bull; Objective 1 (Early Warning Reach):</b> Can intermediate diffusion states forecast final safety violations early, providing an actionable runway before tokens crystallize?", t_bullet))
    story.append(Paragraph("<b>&bull; Objective 2 (Active Trajectory Steering):</b> Can dynamic re-masking and Directional Taboo Steering prevent toxic outputs without fine-tuning model parameters?", t_bullet))
    story.append(Paragraph("<b>&bull; Objective 3 (Compute Preservation):</b> Can the system deliver 100% of user completions safely, eliminating the severe coverage drop of rejection filters?", t_bullet))
    story.append(Paragraph("<b>&bull; Objective 4 (Lightweight Deployment):</b> Can the framework run locally on student consumer GPUs (&lt;4 GB VRAM) with minimal latency overhead (&lt;20%)?", t_bullet))
    story.append(Spacer(1, 3))

    # Sequence 3: Step-by-Step System Methodology Pipeline
    story.append(Paragraph("3. Step-by-Step System Methodology & Workflow Pipeline", t_h1))
    story.append(Paragraph(
        "The Guard-Trajectory system operates through a sequential 5-stage inference pipeline without altering backbone weights:",
        t_body
    ))

    pipe_data = [
        [
            Paragraph("<b>Stage</b>", th_style),
            Paragraph("<b>Pipeline Step Name</b>", th_style),
            Paragraph("<b>Process Description & Internal Operations</b>", th_style),
            Paragraph("<b>Output / Result</b>", th_style),
        ],
        [
            Paragraph("<b>Step 1</b>", tc_bold),
            Paragraph("<b>Input & Mask Init</b>", tc_bold),
            Paragraph("Prompt prefix loaded; continuation positions initialized with [MASK] tokens", tc_style),
            Paragraph("Sequence x<sub>t=1.0</sub> ready", tc_style),
        ],
        [
            Paragraph("<b>Step 2</b>", tc_bold),
            Paragraph("<b>Reverse Diffusion</b>", tc_bold),
            Paragraph("MDLM backbone (169.6M params) denoises sequence across 16 reverse steps", tc_style),
            Paragraph("Candidate clean text x&#770;<sub>0</sub><sup>(t)</sup>", tc_style),
        ],
        [
            Paragraph("<b>Step 3</b>", tc_bold),
            Paragraph("<b>Trajectory Hook</b>", tc_bold),
            Paragraph("Intermediate hook extracts candidate tokens and evaluates risk score r(t)", tc_style),
            Paragraph("Risk score [0.0 to 1.0]", tc_style),
        ],
        [
            Paragraph("<b>Step 4</b>", tc_bold),
            Paragraph("<b>Adaptive Steering</b>", tc_bold),
            Paragraph("If risk &ge; &tau;(t) for 2 checks, re-masks tokens and adds candidate words to Taboo set", tc_style),
            Paragraph("Steered trajectory path", tc_style),
        ],
        [
            Paragraph("<b>Step 5</b>", tc_bold),
            Paragraph("<b>Safe Completion</b>", tc_bold),
            Paragraph("Tokens finalize deterministically in safe semantic basin with 100% coverage", tc_style),
            Paragraph("Safe, fluent final text", tc_style),
        ],
    ]
    tbl_pipe = Table(pipe_data, colWidths=[45, 115, 260, 120])
    tbl_pipe.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_navy),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('GRID', (0,0), (-1,-1), 0.5, c_border),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, c_alt_row]),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('LEFTPADDING', (0,0), (-1,-1), 4),
        ('RIGHTPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(tbl_pipe)

    # =========================================================================
    # PAGE 2: SEQUENCE 4 & 5 — WORK DONE & EARLY WARNING "REACH" (RQ1)
    # =========================================================================
    story.append(PageBreak())
    story.append(Paragraph("4. Sequence of Work Completed Till Date (Milestone Checklist)", t_h1))
    story.append(Paragraph(
        "The project has successfully progressed through all foundational, algorithmic, and evaluation milestones:",
        t_body
    ))

    mile_data = [
        [
            Paragraph("<b>#</b>", th_style),
            Paragraph("<b>Milestone Name</b>", th_style),
            Paragraph("<b>Scope & Implementation Details</b>", th_style),
            Paragraph("<b>Status</b>", th_style),
        ],
        [
            Paragraph("1", tc_center),
            Paragraph("<b>Backbone Pipeline</b>", tc_bold),
            Paragraph("Pretrained MDLM backbone (<code>kuleshov-group/mdlm-owt</code>, 169.6M params, 16 reverse steps)", tc_style),
            Paragraph("<font color='#16A34A'><b>COMPLETED</b></font>", tc_center),
        ],
        [
            Paragraph("2", tc_center),
            Paragraph("<b>Trajectory Step-Hook</b>", tc_bold),
            Paragraph("Interference hook wrapper exposing candidate intermediate states x&#770;<sub>0</sub><sup>(t)</sup> in real time", tc_style),
            Paragraph("<font color='#16A34A'><b>COMPLETED</b></font>", tc_center),
        ],
        [
            Paragraph("3", tc_center),
            Paragraph("<b>7 Policy Implementations</b>", tc_bold),
            Paragraph("Engineered P0 (Baseline), P1 (Post-hoc), P2-P4 (Rollback variants), P5 (Adaptive), P6 (Ensemble)", tc_style),
            Paragraph("<font color='#16A34A'><b>COMPLETED</b></font>", tc_center),
        ],
        [
            Paragraph("4", tc_center),
            Paragraph("<b>Multi-Split Dataset</b>", tc_bold),
            Paragraph("Stratified RealToxicityPrompts: 5 debug, 50 dev, 200 test, 100 robustness (2,485 runs logged)", tc_style),
            Paragraph("<font color='#16A34A'><b>COMPLETED</b></font>", tc_center),
        ],
        [
            Paragraph("5", tc_center),
            Paragraph("<b>Early Detection Study</b>", tc_bold),
            Paragraph("Analyzed intermediate predictability across reverse timesteps (AUROC, AUPRC, Lead time)", tc_style),
            Paragraph("<font color='#16A34A'><b>COMPLETED</b></font>", tc_center),
        ],
        [
            Paragraph("6", tc_center),
            Paragraph("<b>Independent Validation</b>", tc_bold),
            Paragraph("Cross-evaluated outputs using an external RoBERTa classifier not used during generation", tc_style),
            Paragraph("<font color='#16A34A'><b>COMPLETED</b></font>", tc_center),
        ],
        [
            Paragraph("7", tc_center),
            Paragraph("<b>Conference Paper Draft</b>", tc_bold),
            Paragraph("Full IEEE conference manuscript formatted with LaTeX, figures, and Overleaf submission package", tc_style),
            Paragraph("<font color='#16A34A'><b>COMPLETED</b></font>", tc_center),
        ],
    ]
    tbl_mile = Table(mile_data, colWidths=[25, 120, 315, 80])
    tbl_mile.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_navy),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('GRID', (0,0), (-1,-1), 0.5, c_border),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, c_alt_row]),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('LEFTPADDING', (0,0), (-1,-1), 4),
        ('RIGHTPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(tbl_mile)
    story.append(Spacer(1, 4))

    # Sequence 5: Key Breakthrough — The Early Warning "Reach" (RQ1)
    story.append(Paragraph("5. The Key Breakthrough: Early Risk Detection 'Reach' (RQ1)", t_h1))
    story.append(Paragraph(
        "<b>The Core Discovery:</b> Does the model reveal its safety trajectory early? We evaluated intermediate candidate "
        "predictions across 50 development trajectories. The results demonstrate remarkable early discrimination:",
        t_body
    ))

    rq1_data = [
        [
            Paragraph("<b>Diffusion Stage (t)</b>", th_style),
            Paragraph("<b>Reverse Progress</b>", th_style),
            Paragraph("<b>AUROC (Accuracy)</b>", th_style),
            Paragraph("<b>AUPRC</b>", th_style),
            Paragraph("<b>Detection Recall</b>", th_style),
            Paragraph("<b>Early Warning Lead Runway</b>", th_style),
        ],
        [
            Paragraph("<b>t = 0.813 (Step 3)</b>", tc_bold),
            Paragraph("18% into generation", tc_center),
            Paragraph("<b>0.9324 (93.2%)</b>", tc_bold),
            Paragraph("<b>0.8932</b>", tc_bold),
            Paragraph("<b>90.9%</b>", tc_bold),
            Paragraph("<b>13.36 / 16 steps early!</b>", tc_bold),
        ],
        [
            Paragraph("<b>t = 0.625 (Step 6)</b>", tc_bold),
            Paragraph("37% into generation", tc_center),
            Paragraph("<b>0.9347 (93.5%)</b>", tc_bold),
            Paragraph("<b>0.8861</b>", tc_bold),
            Paragraph("90.9%", tc_center),
            Paragraph("10.00 / 16 steps early", tc_center),
        ],
        [
            Paragraph("<b>t = 0.376 (Step 10)</b>", tc_bold),
            Paragraph("62% into generation", tc_center),
            Paragraph("0.9231 (92.3%)", tc_center),
            Paragraph("0.8608", tc_center),
            Paragraph("90.9%", tc_center),
            Paragraph("6.00 / 16 steps early", tc_center),
        ],
    ]
    tbl_rq1 = Table(rq1_data, colWidths=[95, 95, 85, 75, 80, 110])
    tbl_rq1.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_navy),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('GRID', (0,0), (-1,-1), 0.5, c_border),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, c_alt_row]),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(tbl_rq1)
    story.append(Spacer(1, 4))

    # Figure 1: Early Risk Trajectories
    fig1_p = os.path.join("plots", "fig1_risk_trajectories.png")
    if os.path.exists(fig1_p):
        story.append(Image(fig1_p, width=490, height=190))
        story.append(Paragraph("<b>Figure 1:</b> Intermediate risk trajectories r(x&#770;<sub>0</sub><sup>(t)</sup>) over reverse steps (t=1.0 &rarr; 0.0). Unsafe trajectories (red) diverge sharply from safe trajectories (blue) by step 3 (t &approx; 0.81), providing an average 13.4-step lead window for intervention.", t_caption))

    # =========================================================================
    # PAGE 3: SEQUENCE 6 & 7 — OUTPUTS & EXPERIMENTAL RESULTS OBTAINED
    # =========================================================================
    story.append(PageBreak())
    story.append(Paragraph("6. Outputs We Are Getting: Policy Performance Benchmark", t_h1))
    story.append(Paragraph(
        "We evaluated all policies on 200 stratified test prompts from RealToxicityPrompts under matched seeds. "
        "The table below summarizes the quantitative outputs obtained across safety, compute efficiency, and diversity:",
        t_body
    ))

    # Benchmark Results Table
    bench_data = [
        [
            Paragraph("<b>Policy</b>", th_style),
            Paragraph("<b>Approach & Mechanism</b>", th_style),
            Paragraph("<b>Unsafe Output %</b>", th_style),
            Paragraph("<b>Interventions</b>", th_style),
            Paragraph("<b>Diversity (Dist-1)</b>", th_style),
            Paragraph("<b>Latency (s)</b>", th_style),
            Paragraph("<b>Overhead</b>", th_style),
            Paragraph("<b>Delivery Success</b>", th_style),
        ],
        [
            Paragraph("<b>P0 Baseline</b>", tc_bold),
            Paragraph("Unguarded vanilla MDLM reverse sampling", tc_style),
            Paragraph("6.5%", tc_center),
            Paragraph("0.00", tc_center),
            Paragraph("0.4532", tc_center),
            Paragraph("0.514s", tc_center),
            Paragraph("0.0% (Ref)", tc_center),
            Paragraph("100% (Has toxic)", tc_center),
        ],
        [
            Paragraph("<b>P1 Post-Hoc</b>", tc_bold),
            Paragraph("Rejects output if final risk &ge; 0.5 (discards text)", tc_style),
            Paragraph("6.5%*", tc_center),
            Paragraph("0.00", tc_center),
            Paragraph("0.4532", tc_center),
            Paragraph("0.517s", tc_center),
            Paragraph("+0.6%", tc_center),
            Paragraph("<b>93.5% (Drops 13!)</b>", tc_bold),
        ],
        [
            Paragraph("<b>P2 Dense</b>", tc_bold),
            Paragraph("Dense rollback at every single reverse step (naive)", tc_style),
            Paragraph("10.5%", tc_center),
            Paragraph("5.36", tc_center),
            Paragraph("0.4418", tc_center),
            Paragraph("0.736s", tc_center),
            Paragraph("+43.3%", tc_center),
            Paragraph("100% (Drift failure)", tc_center),
        ],
        [
            Paragraph("<b>P5 Adaptive</b>", tc_bold),
            Paragraph("Dynamic threshold + Directional Taboo Steering", tc_style),
            Paragraph("<b>6.0%</b>", tc_bold),
            Paragraph("1.24", tc_center),
            Paragraph("0.4699", tc_center),
            Paragraph("<b>0.600s</b>", tc_bold),
            Paragraph("<b>+16.7%</b>", tc_bold),
            Paragraph("<b>100% Delivered</b>", tc_bold),
        ],
        [
            Paragraph("<b>P6 Ensemble</b>", tc_bold),
            Paragraph("Dual-scorer consensus (toxic-bert + offensive-roberta)", tc_style),
            Paragraph("<b>5.0%</b>", tc_bold),
            Paragraph("1.40", tc_center),
            Paragraph("<b>0.4745</b>", tc_bold),
            Paragraph("0.723s", tc_center),
            Paragraph("+40.7%", tc_center),
            Paragraph("<b>100% Delivered</b>", tc_bold),
        ],
    ]
    tbl_bench = Table(bench_data, colWidths=[65, 145, 52, 45, 65, 48, 48, 72])
    tbl_bench.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_navy),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('GRID', (0,0), (-1,-1), 0.5, c_border),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, c_alt_row]),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('LEFTPADDING', (0,0), (-1,-1), 3),
        ('RIGHTPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(tbl_bench)
    story.append(Spacer(1, 4))

    # Core Output Takeaways
    out_takeaways = (
        "<b>Core Output Takeaways for Faculty:</b><br/>"
        "&bull; <b>P5 is Pareto-Optimal:</b> Lowers violation rate to 6.0% with only 1.24 interventions per continuation, incurring only "
        "<b>16.7% latency overhead</b> (0.60s vs 0.51s baseline) while improving unigram diversity (Dist-1: 0.470 vs 0.453).<br/>"
        "&bull; <b>P6 Ensemble Achieves Lowest Risk:</b> Reduces violations down to <b>5.0%</b> (1.5 percentage points absolute drop) "
        "and produces the highest linguistic richness (Entropy: 9.664 vs 9.510 baseline).<br/>"
        "&bull; <b>Post-Hoc Baseline Penalty (P1):</b> P1 rejects 13 outputs, delivering only <b>93.5% coverage</b>. Guard-Trajectory delivers "
        "<b>100% full coverage</b> by steering generations in-flight rather than discarding user queries.<br/>"
        "&bull; <b>Failure of Naive Interventions (P2):</b> Intervening at every step induces exploratory drift, <i>worsening</i> violations to 10.5% (+4.0 pp worse)."
    )
    story.append(Paragraph(out_takeaways, t_body))
    story.append(Spacer(1, 3))

    # Figure 2 & 3: Violations and Pareto Tradeoff
    fig2_p = os.path.join("plots", "fig2_policy_violations_test.png")
    fig3_p = os.path.join("plots", "fig3_pareto_tradeoff_test.png")
    if os.path.exists(fig2_p) and os.path.exists(fig3_p):
        fig_tbl = Table(
            [[Image(fig2_p, width=265, height=135), Image(fig3_p, width=265, height=135)]],
            colWidths=[270, 270]
        )
        fig_tbl.setStyle(TableStyle([
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('LEFTPADDING', (0,0), (-1,-1), 0),
            ('RIGHTPADDING', (0,0), (-1,-1), 0),
            ('TOPPADDING', (0,0), (-1,-1), 0),
            ('BOTTOMPADDING', (0,0), (-1,-1), 0),
        ]))
        story.append(fig_tbl)
        story.append(Paragraph("<b>Figure 2:</b> Left: Policy violation percentages on Test prompts (N=200). Right: Multi-objective Pareto frontier showing P5 dominating P2, P3, and P4 in the Safety vs Latency trade-off.", t_caption))
    story.append(Spacer(1, 3))

    # Sequence 7: Independent Validation
    story.append(Paragraph("7. Independent Cross-Validation (External Model Verification)", t_h1))
    valid_text = (
        "To ensure results are not an artifact of scorer bias, all 1,400 test outputs were re-evaluated with an independent model: "
        "<code>s-nlp/roberta_toxicity_classifier</code> (not used during generation).<br/>"
        "&bull; <b>Test Set (N=200):</b> Baseline P0 violations drop from <b>6.5% down to 4.0%</b> under both P5 and P6 (+2.5 pp reduction).<br/>"
        "&bull; <b>Robustness Set (N=100):</b> Baseline violations drop from <b>9.0% down to 6.0% (P5)</b> and <b>5.0% (P6)</b> (+4.0 pp reduction)."
    )
    story.append(Paragraph(valid_text, t_body))

    # =========================================================================
    # PAGE 4: SEQUENCE 8, 9, 10, 11 — HARDWARE, NEXT STEPS & FACULTY SIGN-OFF
    # =========================================================================
    story.append(PageBreak())
    story.append(Paragraph("8. Hardware Analysis & Computational Feasibility", t_h1))
    story.append(Paragraph(
        "A critical asset of this work is that it was engineered and empirically verified on consumer hardware. "
        "Full profiling was performed on a student laptop GPU (NVIDIA GeForce RTX 3050 Laptop GPU, 4 GB VRAM):",
        t_body
    ))

    hw_data = [
        [
            Paragraph("<b>Hardware Component / Dimension</b>", th_style),
            Paragraph("<b>Memory Footprint</b>", th_style),
            Paragraph("<b>Execution Target</b>", th_style),
            Paragraph("<b>Feasibility & Headroom Assessment</b>", th_style),
        ],
        [
            Paragraph("<b>MDLM Backbone (169.6M params)</b>", tc_bold),
            Paragraph("678.4 MB (Weights)", tc_center),
            Paragraph("Local RTX 3050 (4 GB)", tc_center),
            Paragraph("Loaded locally in PyTorch; fast inference", tc_style),
        ],
        [
            Paragraph("<b>Primary Scorer (toxic-bert)</b>", tc_bold),
            Paragraph("426.9 MB (Weights)", tc_center),
            Paragraph("Co-located on GPU", tc_center),
            Paragraph("Sub-second forward pass (16.4 ms per check)", tc_style),
        ],
        [
            Paragraph("<b>Intermediate State Buffers</b>", tc_bold),
            Paragraph("735.6 MB (Peak)", tc_center),
            Paragraph("PyTorch Tensor Buffer", tc_center),
            Paragraph("Zero memory leak across 2,485 runs", tc_style),
        ],
        [
            Paragraph("<b>Total Peak Allocated VRAM</b>", tc_bold),
            Paragraph("<b>1,840.9 MB</b>", tc_bold),
            Paragraph("<b>4,096 MB VRAM Cap</b>", tc_bold),
            Paragraph("<b>55.1% Safety Headroom Available!</b>", tc_bold),
        ],
    ]
    tbl_hw = Table(hw_data, colWidths=[140, 100, 110, 190])
    tbl_hw.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_navy),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('GRID', (0,0), (-1,-1), 0.5, c_border),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, c_alt_row]),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('LEFTPADDING', (0,0), (-1,-1), 4),
        ('RIGHTPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(tbl_hw)
    story.append(Spacer(1, 5))

    # Sequence 9: Tangible Project Deliverables Built
    story.append(Paragraph("9. Tangible Deliverables & Code Assets Built", t_h1))
    story.append(Paragraph(
        "All code, experiment runners, logs, and manuscripts are stored and reproducible in the project workspace:<br/>"
        "&bull; <b>Diffusion Hook Engine:</b> <code>src/samplers/diffusion_step_hook.py</code><br/>"
        "&bull; <b>Policy Implementations (P0&ndash;P6):</b> <code>src/policies/guard_policies.py</code><br/>"
        "&bull; <b>Experiment Logs & Datasets:</b> <code>results/experiment_runs_test.jsonl</code> (2,485 generation trajectories)<br/>"
        "&bull; <b>Independent Validation Records:</b> <code>results/mdlm_independent_test_scores.csv</code><br/>"
        "&bull; <b>Colab Reproducibility Pipeline:</b> <code>colab/guard_trajectory_colab.ipynb</code><br/>"
        "&bull; <b>Conference Paper Manuscript:</b> <code>paper/ieee_conference_paper.tex</code> & Overleaf zip package",
        t_body
    ))
    story.append(Spacer(1, 4))

    # Sequence 10: Next Steps Roadmap
    story.append(Paragraph("10. Immediate Next Steps & Publication Roadmap", t_h1))
    road_data = [
        [
            Paragraph("<b>Phase</b>", th_style),
            Paragraph("<b>Target Timeline</b>", th_style),
            Paragraph("<b>Core Deliverables & Action Items</b>", th_style),
        ],
        [
            Paragraph("<b>Phase 1</b>", tc_bold),
            Paragraph("October 2026 (Done)", tc_center),
            Paragraph("7-policy benchmark complete, P1 audit verified, progress report built", tc_style),
        ],
        [
            Paragraph("<b>Phase 2</b>", tc_bold),
            Paragraph("November 2026", tc_center),
            Paragraph("Conduct 3-annotator blinded human evaluation for fluency and safety preservation", tc_style),
        ],
        [
            Paragraph("<b>Phase 3</b>", tc_bold),
            Paragraph("December 2026", tc_center),
            Paragraph("Scale prompt evaluation to 1,000 prompts across 5 seeds on lab GPU cluster", tc_style),
        ],
        [
            Paragraph("<b>Phase 4</b>", tc_bold),
            Paragraph("January 2027", tc_center),
            Paragraph("Finalize camera-ready IEEE / ACL Findings conference submission", tc_style),
        ],
    ]
    tbl_road = Table(road_data, colWidths=[65, 105, 370])
    tbl_road.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_navy),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('GRID', (0,0), (-1,-1), 0.5, c_border),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, c_alt_row]),
        ('TOPPADDING', (0,0), (-1,-1), 2.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
        ('LEFTPADDING', (0,0), (-1,-1), 4),
        ('RIGHTPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(tbl_road)
    story.append(Spacer(1, 5))

    # Sequence 11: Faculty Review & Sign-Off Block
    story.append(Paragraph("11. Faculty Guide Evaluation & Sign-Off Block", t_h1))
    signoff_p = Paragraph(
        "<b>Faculty Guide Comments & Observations:</b><br/>"
        "____________________________________________________________________________________________________________<br/>"
        "____________________________________________________________________________________________________________<br/>"
        "<b>Action Decision:</b> &nbsp;&nbsp;[&nbsp;&nbsp;]&nbsp;Approved to proceed to scaling &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; "
        "[&nbsp;&nbsp;]&nbsp;Proceed with Human Evaluation &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; "
        "[&nbsp;&nbsp;]&nbsp;Ready for Overleaf Submission<br/><br/>"
        "<b>Faculty Guide Signature:</b> ____________________________________ &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; "
        "<b>Date:</b> ________________________",
        t_callout
    )
    tbl_signoff = Table([[signoff_p]], colWidths=[printable_w])
    tbl_signoff.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#F8FAFC")),
        ('BOX', (0,0), (-1,-1), 0.8, colors.HexColor("#94A3B8")),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('LEFTPADDING', (0,0), (-1,-1), 7),
        ('RIGHTPADDING', (0,0), (-1,-1), 7),
    ]))
    story.append(tbl_signoff)

    print(f"Building sequence-wise report {output_pdf} with NumberedCanvas...")
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Successfully generated: {output_pdf}")

if __name__ == "__main__":
    out_pdf = sys.argv[1] if len(sys.argv) > 1 else "Guard_Trajectory_Faculty_Progress_Report.pdf"
    build_sequence_faculty_report(out_pdf)
