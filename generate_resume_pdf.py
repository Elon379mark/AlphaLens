import os
import sys
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas

class NumberedCanvas(canvas.Canvas):
    """
    Two-pass canvas to dynamically compute and render total page count and header/footer.
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
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica-Bold", 8)
        self.setFillColor(colors.HexColor("#475569"))
        
        # Running Header (pages 2+)
        if self._pageNumber > 1:
            self.drawString(54, 750, "ALPHALENS: QUANTITATIVE RESEARCH & CAUSAL ML PLATFORM")
            self.drawRightString(612 - 54, 750, "RESUME & TECHNICAL CONCEPTS GUIDE")
            self.setStrokeColor(colors.HexColor("#CBD5E1"))
            self.setLineWidth(0.75)
            self.line(54, 742, 612 - 54, 742)

        # Running Footer (all pages)
        self.setStrokeColor(colors.HexColor("#CBD5E1"))
        self.setLineWidth(0.75)
        self.line(54, 45, 612 - 54, 45)
        
        self.setFont("Helvetica", 8)
        self.drawString(54, 32, "Confidential • Quantitative Research & Technical Interview Portfolio")
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(612 - 54, 32, page_str)
        self.restoreState()


def build_pdf(filename="AlphaLens_Resume_Technical_Guide.pdf"):
    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )

    styles = getSampleStyleSheet()
    
    # Palette
    c_primary = colors.HexColor("#0F172A")     # Dark Slate / Navy
    c_accent = colors.HexColor("#2563EB")      # Royal Blue
    c_sub = colors.HexColor("#475569")         # Muted Slate
    c_bg_light = colors.HexColor("#F8FAFC")    # Background Tint
    c_border = colors.HexColor("#E2E8F0")      # Border Line

    # Typography Styles
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=22,
        leading=26,
        textColor=c_primary,
        spaceAfter=4
    )
    
    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=15,
        textColor=c_accent,
        spaceAfter=12
    )

    h1_style = ParagraphStyle(
        "SectionH1",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=17,
        textColor=c_primary,
        spaceBefore=14,
        spaceAfter=6,
        keepWithNext=True
    )

    h2_style = ParagraphStyle(
        "SectionH2",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=10,
        leading=14,
        textColor=c_accent,
        spaceBefore=8,
        spaceAfter=4,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        "BodyCustom",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=c_primary,
        spaceAfter=5
    )

    bullet_style = ParagraphStyle(
        "BulletCustom",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.8,
        leading=12.5,
        textColor=c_primary,
        leftIndent=12,
        firstLineIndent=-8,
        spaceAfter=4
    )

    callout_style = ParagraphStyle(
        "CalloutText",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.8,
        leading=12.5,
        textColor=colors.HexColor("#1E293B")
    )

    story = []

    # ==================== HEADER SECTION ====================
    story.append(Paragraph("AlphaLens: Autonomous Quantitative Platform", title_style))
    story.append(Paragraph("Resume Master Guide • Causal Inference, Multi-Agent Engineering & Portfolio Math", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=c_accent, spaceAfter=10))

    # Executive Summary Box
    summary_text = (
        "<b>Executive Overview:</b> AlphaLens is an autonomous multi-agent quantitative alpha research platform "
        "built on LangGraph, PyTorch, and Groq (LLaMA 3.3 70B). It pioneers the integration of Judea Pearl's <i>do-calculus</i> "
        "causal discovery with state-of-the-art time-series deep learning (TFT, N-BEATS, PatchTST, GAT) and convex Mean-CVaR portfolio optimization. "
        "By grounding signals in true causal mechanisms rather than spurious co-movements, AlphaLens delivers a <b>71% live signal survival rate</b>, "
        "a <b>2.14 annualized Sharpe ratio</b>, and a <b>23.1% net annualized return</b> across 3,500 US equities."
    )
    summary_table = Table([[Paragraph(summary_text, callout_style)]], colWidths=[504])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), c_bg_light),
        ('BOX', (0,0), (-1,-1), 1, c_accent),
        ('PADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 10))

    # ==================== SECTION 1: RESUME BULLETS ====================
    story.append(Paragraph("1. Resume Ready Bullet Points (Copy & Paste for Resume)", h1_style))
    story.append(HRFlowable(width="100%", thickness=0.5, color=c_border, spaceAfter=6))

    resume_bullets = [
        "<b>Quantitative Multi-Agent Architecture:</b> Engineered an autonomous 5-agent quantitative alpha discovery pipeline using LangGraph, Python, and Groq (LLaMA 3.3 70B), automating literature RAG, predictive screening, causal gating, backtesting, and portfolio optimization.",
        "<b>Causal Machine Learning & Pearl's Do-Calculus:</b> Operationalized Judea Pearl's do-calculus via PC-Algorithm DAG discovery and Double/Debiased Machine Learning (DML) with Random Forest nuisance models to estimate Average Treatment Effects (ATE), boosting live signal survival from 42% to 71%.",
        "<b>Time-Series Deep Learning & GNNs:</b> Integrated multi-modal forecasting models including Temporal Fusion Transformers (TFT), N-BEATS, PatchTST, and Graph Attention Networks (GAT) to capture non-linear asset cross-correlations and market regime shifts.",
        "<b>Purged Walk-Forward Backtesting:</b> Built a vectorized out-of-sample simulation engine featuring an anchored walk-forward scheme (5-yr train, 1-yr validation, 6-mo test, 2h embargo) and Kyle's lambda market impact model (TC = σ√(q/V_ADV) · P), yielding a 2.14 Sharpe ratio and 23.1% net return.",
        "<b>Convex Risk Minimization & Black-Litterman:</b> Implemented a Mean-CVaR Linear Program solver (CVXPY / CLARABEL) enforcing daily CVaR_0.99 ≤ 2.5%, 20% max turnover, and 0.90–1.10 net exposure, blended with Black-Litterman active views and Brinson-Hood-Beebower (BHB) performance risk attribution.",
        "<b>Production Engineering & Auditability:</b> Deployed TimescaleDB, PostgreSQL, Redis, Kafka, and MLflow with state checkpointing (PostgresCheckpointSaver) for deterministic decision-provenance auditing and real-time Streamlit dashboard visualization."
    ]

    for b in resume_bullets:
        story.append(Paragraph(f"• {b}", bullet_style))

    story.append(Spacer(1, 10))

    # ==================== SECTION 2: ARCHITECTURE TABLE ====================
    story.append(Paragraph("2. Multi-Agent System Architecture & 12-Step Workflow", h1_style))
    story.append(HRFlowable(width="100%", thickness=0.5, color=c_border, spaceAfter=6))

    arch_rows = [
        [Paragraph("<b>Step / Node</b>", h2_style), Paragraph("<b>Agent Role</b>", h2_style), Paragraph("<b>Core Concepts & Tech Stack</b>", h2_style)],
        [
            Paragraph("<b>Step 1-4: Literature & Hypothesis</b>", body_style),
            Paragraph("Literature RAG Agent", body_style),
            Paragraph("Academic paper paragraph chunking, ChromaDB vector RAG, Cosine Similarity deduplication (θ > 0.85), Evaluator-Optimizer prompt chaining.", body_style)
        ],
        [
            Paragraph("<b>Step 5-6: Feature & Screening</b>", body_style),
            Paragraph("Signal Gen Agent", body_style),
            Paragraph("312 base features, 1%/99% Winsorization, Cross-Sectional Z-scores, LASSO CV (λ ∈ [10⁻⁴, 10⁰]), Random Forest MDA, rolling IC (|IC| ≥ 0.03) & ICIR (|ICIR| ≥ 0.5) gating.", body_style)
        ],
        [
            Paragraph("<b>Step 7-8: Causal Validation</b>", body_style),
            Paragraph("Causal Validation Agent", body_style),
            Paragraph("PC-Algorithm DAG discovery, EconML/DoWhy DML ATE estimation, Rosenbaum Sensitivity Bounds (Γ ≤ 2.0), Partial R² omitted variable bias, HC3 SEs.", body_style)
        ],
        [
            Paragraph("<b>Step 9: Vectorized Backtest</b>", body_style),
            Paragraph("Backtest Agent", body_style),
            Paragraph("Anchored Walk-Forward CV (5y W_train, 1y W_val, 6m W_test, Δ=2h embargo), Kyle's Lambda market impact, Sharpe (excess r_f), Information Ratio (r_b), Calmar, MaxDD.", body_style)
        ],
        [
            Paragraph("<b>Step 10-12: Portfolio & Risk</b>", body_style),
            Paragraph("Portfolio Agent & UI", body_style),
            Paragraph("CVXPY Rockafellar-Uryasev Mean-CVaR LP (CVaR_0.99 ≤ 2.5%, turnover ≤ 20%), Black-Litterman view blending, Brinson-Hood-Beebower (BHB) attribution, Streamlit UI.", body_style)
        ]
    ]

    t_arch = Table(arch_rows, colWidths=[120, 114, 270])
    t_arch.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_bg_light),
        ('GRID', (0,0), (-1,-1), 0.5, c_border),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('PADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(t_arch)

    story.append(Spacer(1, 10))

    # ==================== SECTION 3: MATH FORMULAS & CONCEPTS ====================
    story.append(Paragraph("3. Deep-Dive Quant & Causal Formulas", h1_style))
    story.append(HRFlowable(width="100%", thickness=0.5, color=c_border, spaceAfter=6))

    formulas = [
        ("Pearl's Do-Calculus & Double Machine Learning (DML)",
         "<b>DML Average Treatment Effect (ATE):</b><br/>"
         "<i>ATE_DML = (1/n) Σ [ (T_i - m̂(X_i))(Y_i - l̂(X_i)) / (m̂(X_i)(1 - m̂(X_i))) ]</i><br/>"
         "Conditions DAG discovery on macro confounders (VIX, credit spread, rate levels). Evaluates Rosenbaum sensitivity bounds up to Γ ≤ 2.0 with leverage-adjusted HC3 standard errors."),

        ("Kyle's Lambda Market Impact Cost Model",
         "<b>Market Impact & Friction:</b><br/>"
         "<i>TC(q) = |q| · [ (Spread/2 + Commission) · P + σ · √( |q| / V_ADV ) · P ]</i><br/>"
         "Models non-linear price impact to prevent high-turnover strategies from passing gating on zero-friction assumptions."),

        ("Rockafellar-Uryasev Mean-CVaR Optimization",
         "<b>Convex Risk Minimization LP:</b><br/>"
         "<i>min_{w, γ, z} γ + [ 1 / ((1 - α)S) ] Σ z_s   s.t.   z_s ≥ -r_{p,s} - γ,   z_s ≥ 0</i><br/>"
         "<b>Portfolio Constraints:</b> Daily CVaR_0.99 ≤ 2.5%, max weight w_i ≤ 35%, turnover ≤ 20% per rebalance, net exposure ∈ [0.90, 1.10]."),

        ("Brinson-Hood-Beebower (BHB) Performance Risk Attribution",
         "<b>Performance Decomposition relative to Benchmark R_b:</b><br/>"
         "• <i>Allocation Effect: A_i = (w_{p,i} - w_{b,i}) · (R_{b,i} - R_b)</i><br/>"
         "• <i>Selection Effect: S_i = w_{b,i} · (R_{p,i} - R_{b,i})</i><br/>"
         "• <i>Interaction Effect: I_i = (w_{p,i} - w_{b,i}) · (R_{p,i} - R_{b,i})</i>")
    ]

    for title, desc in formulas:
        story.append(Paragraph(title, h2_style))
        story.append(Paragraph(desc, body_style))
        story.append(Spacer(1, 3))

    story.append(Spacer(1, 10))

    # ==================== SECTION 4: EMPIRICAL RESULTS ====================
    story.append(Paragraph("4. Empirical Results & Performance Outcomes", h1_style))
    story.append(HRFlowable(width="100%", thickness=0.5, color=c_border, spaceAfter=6))

    perf_rows = [
        [Paragraph("<b>Metric</b>", h2_style), Paragraph("<b>Without Causal Validation</b>", h2_style), Paragraph("<b>With Causal Validation (AlphaLens)</b>", h2_style)],
        [Paragraph("Signal Survival Rate (Live)", body_style), Paragraph("42%", body_style), Paragraph("<b>71%</b> (+29% improvement)", body_style)],
        [Paragraph("Median Out-of-Sample IC", body_style), Paragraph("0.031", body_style), Paragraph("<b>0.052</b> (+67.7% IC gain)", body_style)],
        [Paragraph("Median Signal Half-Life", body_style), Paragraph("18 days", body_style), Paragraph("<b>47 days</b> (2.6x longer persistence)", body_style)],
        [Paragraph("Average Sharpe Contribution", body_style), Paragraph("0.12", body_style), Paragraph("<b>0.31</b> (+158% Sharpe boost)", body_style)],
        [Paragraph("Market Crowding Overlap", body_style), Paragraph("68%", body_style), Paragraph("<b>34%</b> (50% reduction in overlap)", body_style)],
        [Paragraph("Annualized Return (2015-2024)", body_style), Paragraph("11.8%", body_style), Paragraph("<b>23.1%</b> (net of transaction costs)", body_style)],
        [Paragraph("Annualized Sharpe Ratio", body_style), Paragraph("1.04", body_style), Paragraph("<b>2.14</b>", body_style)],
        [Paragraph("Maximum Drawdown", body_style), Paragraph("-24.6%", body_style), Paragraph("<b>-11.4%</b> (-14.2% COVID 2020 stress)", body_style)]
    ]

    t_perf = Table(perf_rows, colWidths=[180, 160, 164])
    t_perf.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_bg_light),
        ('GRID', (0,0), (-1,-1), 0.5, c_border),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('PADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t_perf)

    story.append(Spacer(1, 10))

    # ==================== SECTION 5: INTERVIEW Q&A ====================
    story.append(Paragraph("5. Technical Interview System Design Q&A", h1_style))
    story.append(HRFlowable(width="100%", thickness=0.5, color=c_border, spaceAfter=6))

    qa = [
        ("Q: Why use Causal Inference instead of standard ML correlation in quantitative finance?",
         "A: Financial markets are non-stationary and prone to spurious correlations that fail during regime shifts. Deep learning alone predicts co-occurrences without mechanism. By conditioning DAG discovery on economic confounders and estimating ATE via DML, AlphaLens ensures a signal causally drives return variations."),

        ("Q: How does AlphaLens prevent look-ahead bias and data leakage?",
         "A: Feature transformations are strictly point-in-time cross-sectional. Backtesting uses Anchored Walk-Forward Cross-Validation with an explicit embargo period Δ = 2h (twice prediction horizon) between train/val/test splits, purging overlapping labels."),

        ("Q: How does the agent rejection-refinement feedback loop operate?",
         "A: Nodes in the LangGraph state machine evaluate statistical gating thresholds (|IC| ≥ 0.03, |ICIR| ≥ 0.5, p < 0.05, Sharpe ≥ 1.0). If a threshold fails, failure reason codes persist in semantic memory, and conditional routers loop control back to the Literature Agent to formulate an alternative hypothesis (max 3 iterations).")
    ]

    for q_text, a_text in qa:
        story.append(Paragraph(f"<b>{q_text}</b>", h2_style))
        story.append(Paragraph(a_text, body_style))
        story.append(Spacer(1, 3))

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Successfully generated PDF at: {filename}")

if __name__ == "__main__":
    build_pdf()
