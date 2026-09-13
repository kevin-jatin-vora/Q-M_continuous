from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.platypus import (
    BaseDocTemplate,
    Flowable,
    Frame,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output" / "pdf" / "smooth_noise_setup_guide.pdf"

NAVY = colors.HexColor("#17324D")
BLUE = colors.HexColor("#2878B5")
TEAL = colors.HexColor("#2A9D8F")
PALE_BLUE = colors.HexColor("#EAF3F8")
PALE_TEAL = colors.HexColor("#E8F5F2")
PALE_GOLD = colors.HexColor("#FFF4D6")
GOLD = colors.HexColor("#E9A23B")
INK = colors.HexColor("#24313D")
MUTED = colors.HexColor("#5F6F7C")
LINE = colors.HexColor("#CAD7E0")
WHITE = colors.white


class NumberedCanvasMixin:
    pass


class HeaderFooterDoc(BaseDocTemplate):
    def __init__(self, filename, **kwargs):
        super().__init__(filename, **kwargs)
        frame = Frame(
            self.leftMargin,
            self.bottomMargin,
            self.width,
            self.height,
            id="normal",
            leftPadding=0,
            rightPadding=0,
            topPadding=0,
            bottomPadding=0,
        )
        self.addPageTemplates(PageTemplate(id="all", frames=[frame], onPage=self._decorate))

    def _decorate(self, canvas, doc):
        canvas.saveState()
        if doc.page > 1:
            canvas.setStrokeColor(LINE)
            canvas.setLineWidth(0.6)
            canvas.line(doc.leftMargin, 0.54 * inch, letter[0] - doc.rightMargin, 0.54 * inch)
            canvas.setFont("Helvetica", 8)
            canvas.setFillColor(MUTED)
            canvas.drawString(doc.leftMargin, 0.34 * inch, "Smooth-noise 16-tile setup")
            canvas.drawRightString(letter[0] - doc.rightMargin, 0.34 * inch, f"{doc.page}")
        canvas.restoreState()


class Pipeline(Flowable):
    def __init__(self, labels, width=470, height=72):
        super().__init__()
        self.labels = labels
        self.width = width
        self.height = height

    def draw(self):
        c = self.canv
        n = len(self.labels)
        gap = 15
        box_w = (self.width - gap * (n - 1)) / n
        y = 18
        for i, label in enumerate(self.labels):
            x = i * (box_w + gap)
            c.setFillColor(PALE_BLUE if i % 2 == 0 else PALE_TEAL)
            c.setStrokeColor(BLUE if i % 2 == 0 else TEAL)
            c.roundRect(x, y, box_w, 42, 7, fill=1, stroke=1)
            c.setFillColor(NAVY)
            c.setFont("Helvetica-Bold", 8.4)
            words = label.split(" ")
            lines, current = [], ""
            for word in words:
                trial = (current + " " + word).strip()
                if stringWidth(trial, "Helvetica-Bold", 8.4) <= box_w - 10:
                    current = trial
                else:
                    lines.append(current)
                    current = word
            if current:
                lines.append(current)
            base = y + 25 + 5 * (len(lines) - 1)
            for j, text in enumerate(lines[:3]):
                c.drawCentredString(x + box_w / 2, base - 10 * j, text)
            if i < n - 1:
                ax = x + box_w
                ay = y + 21
                c.setStrokeColor(MUTED)
                c.line(ax + 2, ay, ax + gap - 3, ay)
                c.line(ax + gap - 7, ay + 3, ax + gap - 3, ay)
                c.line(ax + gap - 7, ay - 3, ax + gap - 3, ay)


class EnvironmentMap(Flowable):
    def __init__(self, width=470, height=280):
        super().__init__()
        self.width = width
        self.height = height

    def draw(self):
        c = self.canv
        ox, oy, cell = 24, 28, 50
        cat_colors = {
            1: colors.HexColor("#7FC8A9"), 2: colors.HexColor("#FFD166"),
            3: colors.HexColor("#90CAF9"), 4: colors.HexColor("#F4978E"),
        }
        grid = [[3, 3, 4, 4], [3, 3, 4, 4], [2, 2, 1, 1], [2, 2, 1, 1]]
        for row in range(4):
            for col in range(4):
                cat = grid[row][col]
                x, y = ox + col * cell, oy + row * cell
                c.setFillColor(cat_colors[cat])
                c.setStrokeColor(WHITE)
                c.rect(x, y, cell, cell, fill=1, stroke=1)
                c.setFillColor(NAVY)
                c.setFont("Helvetica-Bold", 10)
                c.drawCentredString(x + cell / 2, y + cell / 2 - 3, f"C{cat}")
                c.setFillColor(INK)
                c.circle(x + cell / 2, y + cell / 2, 1.8, fill=1, stroke=0)
        c.setFillColor(NAVY)
        c.setFont("Helvetica-Bold", 10)
        c.drawString(24, 244, "Example at determinism = 0")
        c.setFont("Helvetica", 8.5)
        c.setFillColor(MUTED)
        c.drawString(24, 12, "Dots are tile centers; their sigma targets are interpolation anchors.")

        x0 = 270
        c.setFillColor(NAVY)
        c.setFont("Helvetica-Bold", 10)
        c.drawString(x0, 244, "One interpolation cell")
        qx, qy, q = x0 + 18, 100, 120
        c.setFillColor(PALE_BLUE)
        c.setStrokeColor(BLUE)
        c.rect(qx, qy, q, q, fill=1, stroke=1)
        corners = [(0, 0, "m00"), (q, 0, "m10"), (0, q, "m01"), (q, q, "m11")]
        for dx, dy, label in corners:
            c.setFillColor(GOLD)
            c.circle(qx + dx, qy + dy, 4, fill=1, stroke=0)
            c.setFillColor(INK)
            c.setFont("Helvetica", 8)
            c.drawString(qx + dx + (5 if dx == 0 else -23), qy + dy + 7, label)
        px, py = qx + 0.35 * q, qy + 0.65 * q
        c.setFillColor(TEAL)
        c.circle(px, py, 4.5, fill=1, stroke=0)
        c.setDash(2, 2)
        c.setStrokeColor(TEAL)
        for dx, dy, _ in corners:
            c.line(px, py, qx + dx, qy + dy)
        c.setDash()
        c.setFillColor(INK)
        c.setFont("Helvetica", 8.5)
        c.drawString(x0, 73, "At (tx, ty), the four weights are")
        c.drawString(x0, 60, "(1-tx)(1-ty), tx(1-ty),")
        c.drawString(x0, 47, "(1-tx)ty, and tx ty.")
        c.setFillColor(MUTED)
        c.drawString(x0, 24, "At the square center: 25% from each anchor.")


class BoundSketch(Flowable):
    def __init__(self, width=470, height=140):
        super().__init__()
        self.width = width
        self.height = height

    def draw(self):
        c = self.canv
        c.setStrokeColor(LINE)
        c.setLineWidth(1)
        for i in range(5):
            c.line(20 + i * 80, 15, 20 + i * 80, 125)
        for j in range(3):
            c.line(20, 15 + j * 55, 340, 15 + j * 55)
        c.setFillColor(PALE_GOLD)
        c.setStrokeColor(GOLD)
        c.circle(178, 69, 48, fill=1, stroke=1)
        c.setFillColor(NAVY)
        c.circle(178, 69, 4, fill=1, stroke=0)
        c.setFont("Helvetica-Bold", 9)
        c.drawString(185, 74, "nominal next state")
        c.setFont("Helvetica", 8.5)
        c.setFillColor(INK)
        c.drawString(185, 60, "ball radius r")
        c.setFillColor(MUTED)
        c.drawString(358, 100, "Use every tile")
        c.drawString(358, 87, "touched by the ball.")
        c.setFillColor(TEAL)
        c.setFont("Helvetica-Bold", 8.5)
        c.drawString(358, 57, "Choose worst relevant")
        c.drawString(358, 44, "ordinary/cross LC.")
        c.setStrokeColor(TEAL)
        c.line(348, 74, 330, 74)
        c.line(330, 74, 335, 78)
        c.line(330, 74, 335, 70)


styles = getSampleStyleSheet()
styles.add(ParagraphStyle(
    name="TitleBig", parent=styles["Title"], fontName="Helvetica-Bold",
    fontSize=29, leading=33, textColor=NAVY, alignment=TA_LEFT, spaceAfter=16,
))
styles.add(ParagraphStyle(
    name="Subtitle", parent=styles["Normal"], fontSize=13.5, leading=19,
    textColor=MUTED, spaceAfter=16,
))
styles.add(ParagraphStyle(
    name="H1x", parent=styles["Heading1"], fontName="Helvetica-Bold",
    fontSize=20, leading=24, textColor=NAVY, spaceAfter=12,
))
styles.add(ParagraphStyle(
    name="H2x", parent=styles["Heading2"], fontName="Helvetica-Bold",
    fontSize=12.5, leading=15, textColor=BLUE, spaceBefore=8, spaceAfter=5,
))
styles.add(ParagraphStyle(
    name="Bodyx", parent=styles["BodyText"], fontName="Helvetica",
    fontSize=9.3, leading=13.2, textColor=INK, spaceAfter=7,
))
styles.add(ParagraphStyle(
    name="Smallx", parent=styles["BodyText"], fontName="Helvetica",
    fontSize=8, leading=10.5, textColor=MUTED, spaceAfter=4,
))
styles.add(ParagraphStyle(
    name="Equation", parent=styles["BodyText"], fontName="Helvetica-Bold",
    fontSize=10, leading=14, textColor=NAVY, backColor=PALE_BLUE,
    borderPadding=8, spaceBefore=5, spaceAfter=8,
))
styles.add(ParagraphStyle(
    name="Callout", parent=styles["BodyText"], fontName="Helvetica",
    fontSize=9, leading=13, textColor=INK, backColor=PALE_TEAL,
    borderColor=TEAL, borderWidth=0.7, borderPadding=8, spaceBefore=5, spaceAfter=8,
))
styles.add(ParagraphStyle(
    name="Step", parent=styles["BodyText"], fontName="Helvetica",
    fontSize=9.1, leading=12.5, textColor=INK, leftIndent=17, firstLineIndent=-17,
    spaceAfter=6,
))
styles.add(ParagraphStyle(
    name="Cell", parent=styles["BodyText"], fontName="Helvetica",
    fontSize=8.2, leading=10.7, textColor=INK,
))
styles.add(ParagraphStyle(
    name="HeaderCell", parent=styles["BodyText"], fontName="Helvetica-Bold",
    fontSize=8.2, leading=10.7, textColor=WHITE,
))


def P(text, style="Bodyx"):
    return Paragraph(text, styles[style])


def step(n, title, body):
    return P(f"<b>{n}. {title}</b> - {body}", "Step")


def table(data, widths, header=True, font=8.2):
    wrapped = []
    for row_index, row in enumerate(data):
        style = styles["HeaderCell"] if header and row_index == 0 else styles["Cell"]
        wrapped.append([
            value if isinstance(value, Flowable) else Paragraph(str(value), style)
            for value in row
        ])
    t = Table(wrapped, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
    commands = [
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), font),
        ("LEADING", (0, 0), (-1, -1), font + 2.5),
        ("GRID", (0, 0), (-1, -1), 0.45, LINE),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]
    if header:
        commands += [
            ("BACKGROUND", (0, 0), (-1, 0), NAVY),
            ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ]
    for row in range(1 if header else 0, len(data)):
        if row % 2 == 0:
            commands.append(("BACKGROUND", (0, row), (-1, row), colors.HexColor("#F6F9FB")))
    t.setStyle(TableStyle(commands))
    return t


def build_story():
    story = []
    story += [
        Spacer(1, 0.55 * inch),
        P("The Smooth-Noise<br/>16-Tile Experiment", "TitleBig"),
        P("A simple guide to the environment, data collection, Wasserstein Lipschitz constants, Q bounds, and action pruning", "Subtitle"),
        Spacer(1, 0.15 * inch),
        Pipeline(["Environment", "Collect R1 and R2", "Fit transition model", "Compute LCs", "Train Q bounds", "Prune actions"]),
        Spacer(1, 0.25 * inch),
        P("The one-sentence idea", "H2x"),
        P("Every terrain has the same intended movement but a different, smoothly varying noise level. Samples are used to fit that stochastic transition model. Wasserstein distance then compares transition distributions, not unrelated noise draws. Those dynamics constants and reward constants produce learned upper and lower Q bounds; an action is removed only when its best possible value is below another action's guaranteed value.", "Callout"),
        Spacer(1, 0.15 * inch),
        P("Scope", "H2x"),
        P("This guide describes the code in <b>dollar_euro_16tile - smooth noise same mean</b>. It distinguishes the four diagnostic variants from the normal experiment where their estimators differ.", "Bodyx"),
        Spacer(1, 1.0 * inch),
        P("Prepared from the implemented repository setup", "Smallx"),
        P("September 2026", "Smallx"),
        PageBreak(),
    ]

    story += [
        P("1. Environment", "H1x"),
        P("The state is a point s = (x,y) in the unit square [0,1] x [0,1]. The square is divided into 16 equal tiles. Each tile keeps a discrete terrain category for layout, grouping, and plots.", "Bodyx"),
        EnvironmentMap(),
        P("Actions and rewards", "H2x"),
        table([
            ["Item", "Meaning"],
            ["Actions", "0 Down, 1 Up, 2 Left, 3 Right; intended step length is 0.04."],
            ["Start", "(0.5, 0.35). Episodes end at Dollar, Euro, Both, or the horizon."],
            ["R1", "Prefers the Dollar target; also values Both."],
            ["R2", "Prefers the Euro target; also values Both."],
            ["Combined task", "Uses R1 + R2, so Both is the joint objective."],
        ], [1.05 * inch, 5.35 * inch]),
        P("At determinism = 0, categories 1-4 occupy the four 2 x 2 quadrants. At determinism > 0, the requested fraction of tiles becomes category 5 and the remaining categories are reproducibly shuffled.", "Smallx"),
        PageBreak(),
    ]

    story += [
        P("2. Smooth stochastic dynamics", "H1x"),
        P("All categories now have the same mean movement. Terrain changes only the standard deviation of isotropic Gaussian noise.", "Bodyx"),
        P("s' = clip( s + 0.04 v[a] + sigma(s) Z, 0, 1 ), &nbsp;&nbsp; Z ~ N(0, I2)", "Equation"),
        table([
            ["Category center", "Noise multiplier", "Local target standard deviation"],
            ["1", "1.0", "1.0 x base sigma"],
            ["2", "1.1", "1.1 x base sigma"],
            ["3", "1.2", "1.2 x base sigma"],
            ["4", "1.3", "1.3 x base sigma"],
            ["5", "deterministic_sigma_scale", "scale x base sigma"],
        ], [1.55 * inch, 1.65 * inch, 3.2 * inch]),
        P("How interpolation works", "H2x"),
        P("Each tile center is an anchor. Between four neighboring centers, sigma(s) is their bilinear weighted average. Exactly at a tile center, its assigned value is recovered. Halfway between four centers, each contributes 25%. Outside the outermost centers, the nearest value is extended to the boundary.", "Bodyx"),
        P("Why this change matters", "H2x"),
        P("The mean map is a translation, so its Lipschitz constant is 1 away from clipping. The noise scale changes continuously across terrain boundaries, so neighboring transition distributions do not jump abruptly. Direct clipping is non-expansive: it cannot enlarge Euclidean or Wasserstein separation.", "Callout"),
        P("Before data collection, the known field gives K_sigma = sup ||grad sigma(s)||2 and the theoretical global reference K_P = sqrt(1 + 2 K_sigma^2). The run stops early if gamma K_P >= 1.", "Bodyx"),
        PageBreak(),
    ]

    story += [
        P("3. What data is collected", "H1x"),
        Pipeline(["Train R1 behavior", "Record transitions", "Train R2 behavior", "Record transitions", "Pool and index"]),
        step(1, "Train behavior R1", "An epsilon-greedy DQN learns using only reward component R1. This makes it visit states relevant to the Dollar preference."),
        step(2, "Record each step", "For every interaction, save source state s, action a, next state s', reward, source category, source tile, next tile, and whether clipping affected the boundary."),
        step(3, "Train behavior R2", "A second epsilon-greedy DQN uses reward component R2. By default it uses seed + 1,000,000, providing a second behavior distribution."),
        step(4, "Keep free transitions", "Boundary-clipped records are excluded from LC and transition statistics by default. This prevents the wall from being mistaken for unconstrained terrain dynamics."),
        step(5, "Build reusable groups", "The same physical transitions are indexed by source category/action for dynamics, source tile/action for empirical Q checks, and next-state tile for reward slopes. R1 and R2 are stored separately and then combined where required."),
        step(6, "Show coverage", "Four source-state count heatmaps, one per action, combine R1 and R2. Dark cells mean more samples; pale cells reveal sparse coverage."),
        P("What --reuse-data means", "H2x"),
        P("Reuse occurs only when provenance matches: sigma, gamma, determinism, deterministic scale, steps, seeds, boundary policy, noise-model version, interpolation, multipliers, K_sigma, and theoretical K_P. Old hard-boundary transitions are therefore rejected automatically.", "Callout"),
        PageBreak(),
    ]

    story += [
        P("4. Fit a stochastic transition model", "H1x"),
        P("Raw pairs of sampled successors contain different Gaussian draws. Comparing those draws directly measures sample luck. Instead, the code first estimates the conditional distributions that generated the data.", "Bodyx"),
        step(1, "Estimate common movement", "For each action a, pool R1 and R2 across categories and compute mu_hat[a] = mean(s' - s). The fitted mean is f_hat(s,a) = s + mu_hat[a]."),
        step(2, "Compute residuals", "For each record, e = (s' - s) - mu_hat[a]. Residuals contain the stochastic part after removing intended movement."),
        step(3, "Fit five center sigmas", "The local sigma is a bilinear mixture of five nonnegative category-center parameters. Maximum likelihood fits these parameters under a 2D isotropic Gaussian model."),
        P("For a record i, the fitted local scale sigma_i minimizes the Gaussian negative log likelihood: 2 log(sigma_i) + ||e_i||2^2 / (2 sigma_i^2).", "Equation"),
        step(4, "Check the fit", "The diagnostic writes fitted action means, fitted center sigmas, normalized residual mean/covariance, a fitted global K_P, and the known theoretical K_P. The theoretical value is a verification reference, not a constant copied into every local row."),
        P("Fitted model", "H2x"),
        P("P_hat(. | s,a) = N( s + mu_hat[a], sigma_hat(s)^2 I2 ). The mean displacement is shared; local terrain variation appears through sigma_hat(s).", "Callout"),
        P("If base sigma is zero, the code takes a numerical-floor branch and fits all center sigmas as exactly zero.", "Smallx"),
        PageBreak(),
    ]

    story += [
        P("5. How the Lipschitz constants are computed", "H1x"),
        P("Dynamics LC (Lf or K_P)", "H2x"),
        P("For two source states s and t under the same action, compare their fitted Gaussian transition laws with 2-Wasserstein distance:", "Bodyx"),
        P("W2^2 = ||s - t||2^2 + 2 [sigma_hat(s) - sigma_hat(t)]^2", "Equation"),
        P("pair dynamics ratio = W2 / ||s - t||2", "Equation"),
        P("The common mean action displacement cancels. The factor 2 appears because the noise is isotropic in two state dimensions. Clipping is non-expansive, so this pre-clipping comparison is conservative for the clipped kernel.", "Bodyx"),
        P("Reward LC (Lr)", "H2x"),
        P("Reward is a function of the next state. For eligible next-state pairs, compute |R(s1') - R(s2')| / ||s1' - s2'||. Compute R1 and R2 separately, then add their reward constants for the combined task.", "Bodyx"),
        P("Four diagnostic variants", "H2x"),
        table([
            ["Variant", "Which states are paired", "Aggregation"],
            ["local_mean", "Within each category/action; plus neighboring category pairs", "Arithmetic mean"],
            ["local_max", "Same local and neighboring groups", "Maximum"],
            ["global_mean", "All source states pooled per action", "Arithmetic mean"],
            ["global_max", "All source states pooled per action", "Maximum"],
        ], [1.15 * inch, 3.55 * inch, 1.7 * inch]),
        P("In diagnostics, a pair is eligible only when its denominator distance is at least --min-pair-distance (default 0.04). The gap avoids tiny-denominator amplification; it does not change the fitted stochastic model. In the normal experiment, the corresponding data-fitted ratios use a 10% trimmed mean.", "Callout"),
        PageBreak(),
    ]

    story += [
        P("6. Local, cross, and global meaning", "H1x"),
        table([
            ["Quantity", "Grouping", "Why it exists"],
            ["Ordinary Lf", "Source category x action", "Captures variation among states assigned to one terrain category."],
            ["Cross Lf", "Neighboring source-category pair x action", "Measures smooth transition-distribution change across terrain boundaries."],
            ["Global Lf", "All source states x action", "Tests whether pooling hides or amplifies local differences."],
            ["Ordinary Lr", "Next-state tile", "Reward geometry depends on destination location."],
            ["Cross Lr", "Physically neighboring next-state tiles", "Covers an uncertainty ball that crosses a tile edge."],
        ], [1.25 * inch, 2.05 * inch, 3.1 * inch]),
        P("Combining R1 and R2", "H2x"),
        P("For dynamics, the operational combined value is max(Lf_R1, Lf_R2): the bound must cover either behavior's estimate. For reward, Lr_total = Lr_R1 + Lr_R2 because the task reward is their sum.", "Bodyx"),
        P("Bellman Q Lipschitz bound", "H2x"),
        P("Lq = Lr Lf / (1 - gamma Lf), valid only when gamma Lf < 1", "Equation"),
        P("This follows the discounted Bellman Lipschitz argument: reward sensitivity contributes Lr, transition sensitivity propagates future value by Lf, and repeated discounted propagation forms a geometric series. If gamma Lf >= 1, the denominator is nonpositive; the code marks that variant invalid and does not train bounds from it.", "Bodyx"),
        P("The empirical network slope |Q(s,a)-Q(t,a)|/||s-t|| is saved as a diagnostic. It does not replace the theoretical Bellman Lq used for safe margins.", "Callout"),
        PageBreak(),
    ]

    story += [
        P("7. Build a next-state uncertainty ball", "H1x"),
        P("For every source category c and current action a, pool free R1 and R2 displacements delta = s' - s.", "Bodyx"),
        table([
            ["Stored value", "Computation", "Purpose"],
            ["delta_mean", "pooled mean(delta)", "Nominal next-state movement"],
            ["delta_std", "pooled sample standard deviation by axis", "Observed spread"],
            ["half-width hw", "Student-t multiplier x delta_std x sqrt(1 + 1/n)", "95% prediction interval for a new displacement"],
            ["radius r", "||hw||2", "Circular uncertainty radius used by Q bounds"],
        ], [1.25 * inch, 2.75 * inch, 2.4 * inch]),
        P("The nominal state is s_bar' = clip(s + delta_mean[c,a], 0,1). The ball B(s_bar', r) describes plausible next states for that current state and action.", "Equation"),
        BoundSketch(),
        P("Overlap-aware constants", "H2x"),
        P("The ball may touch several tiles and categories. The code chooses the largest relevant ordinary or cross-tile Lr. For each possible future action b, it also chooses the largest relevant ordinary or cross-category Lf, then computes Lq_eff(b). Finally it takes max over all four future actions because the Bellman target contains max_b Q(s',b).", "Bodyx"),
        P("delta_r = Lr_eff r &nbsp;&nbsp;&nbsp; and &nbsp;&nbsp;&nbsp; delta_q = max_b Lq_eff(b) r", "Equation"),
        PageBreak(),
    ]

    story += [
        P("8. Train upper and lower Q bounds", "H1x"),
        P("Two Q networks start from exactly the same weights. One learns an optimistic Bellman target and the other a pessimistic target. Their target networks are updated with Polyak averaging.", "Bodyx"),
        P("y_UB = R + delta_r + gamma (1-done) [ max_b Q_UB_target(s_bar',b) + delta_q ]", "Equation"),
        P("y_LB = R - delta_r + gamma (1-done) [ max_b Q_LB_target(s_bar',b) - delta_q ]", "Equation"),
        P("The reward margin covers moving anywhere inside the next-state ball. The future-value margin covers how much the optimal continuation value can change inside that ball. Larger radius or larger LCs widen the interval; wider intervals lead to less pruning.", "Callout"),
        P("Action pruning rule", "H2x"),
        P("At state s, first compute threshold = max_b Q_LB(s,b). Keep action a when:", "Bodyx"),
        P("Q_UB(s,a) >= threshold - tolerance", "Equation"),
        P("If an action's optimistic value is already below another action's pessimistic value, it cannot be optimal under the learned bounds and is pruned. RA-DQN explores and selects greedily only among surviving actions. During learning, its next-action maximization is also restricted to survivors. A small ranking loss discourages pruned actions from overtaking the best allowed action.", "Bodyx"),
        P("Safety behavior", "H2x"),
        P("The LC validity condition is enforced rather than clipped or silently replaced. The action-mask helper also keeps all actions if numerical inconsistency ever produces an empty mask, avoiding an undefined policy decision.", "Smallx"),
        PageBreak(),
    ]

    story += [
        P("9. End-to-end checklist and outputs", "H1x"),
        table([
            ["Stage", "Input", "Main output"],
            ["1. Environment check", "sigma field, gamma", "theoretical K_sigma, K_P, gamma K_P"],
            ["2. R1/R2 collection", "two epsilon-greedy behaviors", "raw_transitions.npz, coverage heatmaps"],
            ["3. Transition statistics", "free displacement samples", "transition_bounds.json"],
            ["4. Stochastic fit", "states and residuals", "fitted means/sigmas and verification JSON"],
            ["5. LC variants", "eligible state pairs", "ordinary/cross reward and Wasserstein constants"],
            ["6. Validity", "gamma and operational Lf", "continue only if gamma Lf < 1"],
            ["7. Q bounds", "LCs and Student-t radii", "q_ub_theoretical.pth, q_lb_theoretical.pth"],
            ["8. Analysis/pruning", "frozen Q bounds", "value, width, violation, and pruning heatmaps"],
        ], [1.35 * inch, 2.05 * inch, 3.0 * inch], font=7.8),
        P("Diagnostic command", "H2x"),
        P("run_diagnostics.cmd configs\\radial_match.json --sigma 0.0005 --gamma 0.96 --determinism 0.0 --deterministic-sigma-scale 0.001 --steps 150000 --seed 0 --reuse-data --min-pair-distance 0.04", "Equation"),
        P("How to read the result", "H2x"),
        step(1, "Coverage first", "Use the shared action heatmaps to identify sparsely sampled areas."),
        step(2, "Fit second", "Check fitted sigmas, normalized residuals, and fitted-vs-theoretical global K_P."),
        step(3, "Compare variants", "Use wasserstein_variant_comparison.json to see whether local/global and mean/max differ materially."),
        step(4, "Check validity", "If gamma Lf >= 1, that variant is skipped. Otherwise inspect bound width and pruning heatmaps."),
        P("The key interpretation", "H2x"),
        P("The setup separates three uncertainties: stochastic terrain is represented by a fitted transition distribution; finite data creates a Student-t next-state radius; and Lipschitz constants translate that radius into reward and future-value margins. Pruning happens only after all three pieces are combined.", "Callout"),
    ]
    return story


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc = HeaderFooterDoc(
        str(OUT), pagesize=letter,
        leftMargin=0.72 * inch, rightMargin=0.72 * inch,
        topMargin=0.65 * inch, bottomMargin=0.72 * inch,
        title="The Smooth-Noise 16-Tile Experiment",
        author="Codex",
        subject="Environment, data collection, Wasserstein Lipschitz constants, Q bounds, and pruning",
    )
    doc.build(build_story())
    print(OUT)


if __name__ == "__main__":
    main()
