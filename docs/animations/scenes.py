"""ksearch teaching scenes (Manim Community).

    manim -pql docs/animations/scenes.py TransformerBlock
"""

from __future__ import annotations

from manim import *

config.background_color = "#0d1117"
config.pixel_height = 1080
config.pixel_width = 1920

INK = "#e6edf3"
MUTED = "#8b949e"
ACCENT = "#58a6ff"
GREEN = "#3fb950"
ORANGE = "#d29922"
PURPLE = "#bc8cff"
RED = "#f85149"
BOX_FILL = "#161b22"
BOX_STROKE = "#30363d"


def box(text: str, color: str = ACCENT, width: float = 2.6, height: float = 0.9) -> VGroup:
    r = RoundedRectangle(
        corner_radius=0.12,
        width=width,
        height=height,
        fill_color=BOX_FILL,
        fill_opacity=1,
        stroke_color=color,
        stroke_width=2,
    )
    t = Text(text, font_size=22, color=INK)
    t.move_to(r.get_center())
    if t.width > width - 0.24:
        t.scale_to_fit_width(width - 0.24)
    return VGroup(r, t)


def caption(text: str) -> Text:
    return Text(text, font_size=28, color=INK)


def note(text: str) -> Text:
    return Text(text, font_size=20, color=MUTED)


def arrow(a: Mobject, b: Mobject, color: str = MUTED) -> Arrow:
    return Arrow(
        a.get_right(),
        b.get_left(),
        buff=0.12,
        stroke_width=3,
        color=color,
        max_tip_length_to_length_ratio=0.12,
    )


def down_arrow(a: Mobject, b: Mobject, color: str = MUTED) -> Arrow:
    return Arrow(
        a.get_bottom(),
        b.get_top(),
        buff=0.12,
        stroke_width=3,
        color=color,
        max_tip_length_to_length_ratio=0.12,
    )


class TransformerBlock(Scene):
    """Residual stream: x + Attn(RMS(x)), then x + MLP(RMS(x))."""

    def construct(self):
        title = caption("Decoder block (residual stream)")
        title.to_edge(UP, buff=0.4)
        self.play(FadeIn(title))

        x = box("x  hidden", GREEN, 2.2)
        x.shift(LEFT * 5)
        n1 = box("RMSNorm", ACCENT)
        attn = box("Attention", ORANGE, 2.8)
        add1 = box("+", GREEN, 1.2, 0.9)
        n2 = box("RMSNorm", ACCENT)
        mlp = box("MLP", PURPLE)
        add2 = box("+", GREEN, 1.2, 0.9)
        out = box("x'", GREEN, 2.2)
        out.shift(RIGHT * 5)

        row1 = VGroup(n1, attn, add1).arrange(RIGHT, buff=0.55)
        row2 = VGroup(n2, mlp, add2).arrange(RIGHT, buff=0.55)
        stack = VGroup(row1, row2).arrange(DOWN, buff=1.4)
        stack.move_to(ORIGIN + RIGHT * 0.3)

        self.play(FadeIn(x))
        self.play(FadeIn(row1))
        a1 = arrow(x, n1, GREEN)
        skip1 = Arrow(
            x.get_right() + DOWN * 0.15,
            add1.get_left() + DOWN * 0.35,
            buff=0.1,
            stroke_width=2,
            color=GREEN,
        )
        self.play(Create(a1), Create(skip1))
        self.play(FadeIn(row2), FadeIn(out))
        a2 = down_arrow(add1, n2, GREEN)
        a3 = arrow(add2, out, GREEN)
        skip2 = Arrow(
            add1.get_bottom(),
            add2.get_left() + LEFT * 0.05,
            buff=0.12,
            stroke_width=2,
            color=GREEN,
        )
        self.play(Create(a2), Create(skip2), Create(a3))

        formula = note("x ← x + Attn(RMS(x));   x ← x + MLP(RMS(x))")
        formula.to_edge(DOWN, buff=0.45)
        self.play(FadeIn(formula))
        self.wait(2)


class SdpaNaive(Scene):
    """Q K V → scores → softmax → weighted V."""

    def construct(self):
        title = caption("Scaled dot-product attention (naive)")
        title.to_edge(UP, buff=0.4)
        self.play(FadeIn(title))

        q = box("Q", ACCENT, 1.6)
        k = box("K  cache", ORANGE, 2.2)
        v = box("V  cache", ORANGE, 2.2)
        qk = box("Q Kᵀ / √d", GREEN, 2.6)
        sm = box("softmax\ncausal", PURPLE, 2.4, 1.15)
        o = box("O = w V", ACCENT, 2.2)

        top = VGroup(q, k, v).arrange(RIGHT, buff=1.0).shift(UP * 1.6)
        qk.next_to(top, DOWN, buff=1.1)
        sm.next_to(qk, DOWN, buff=0.9)
        o.next_to(sm, DOWN, buff=0.9)

        self.play(LaggedStart(FadeIn(q), FadeIn(k), FadeIn(v), lag_ratio=0.2))
        self.play(FadeIn(qk), Create(down_arrow(q, qk)), Create(down_arrow(k, qk, ORANGE)))
        self.play(FadeIn(sm), Create(down_arrow(qk, sm, PURPLE)))
        self.play(FadeIn(o), Create(down_arrow(sm, o)), Create(down_arrow(v, o, ORANGE)))

        foot = note("Causal: position p only sees keys 0…p  (SWA: last W of those)")
        foot.to_edge(DOWN, buff=0.4)
        self.play(FadeIn(foot))
        self.wait(2)


class PrefillVsDecode(Scene):
    def construct(self):
        title = caption("Prefill vs decode")
        title.to_edge(UP, buff=0.35)
        self.play(FadeIn(title))

        left = VGroup(
            Text("Prefill", font_size=26, color=ACCENT),
            box("all prompt tokens", ACCENT, 3.6),
            box("batched GEMM ok", ACCENT, 3.6),
            box("write K,V cache", ORANGE, 3.6),
        ).arrange(DOWN, buff=0.35)
        left.shift(LEFT * 3.4)

        right = VGroup(
            Text("Decode", font_size=26, color=GREEN),
            box("one new token", GREEN, 3.6),
            box("matvecs  W @ x", GREEN, 3.6),
            box("append one K,V row", ORANGE, 3.6),
        ).arrange(DOWN, buff=0.35)
        right.shift(RIGHT * 3.4)

        self.play(FadeIn(left))
        self.wait(0.4)
        self.play(FadeIn(right))
        link = Arrow(
            left[3].get_right(),
            right[3].get_left(),
            buff=0.2,
            color=ORANGE,
            stroke_width=3,
        )
        lab = note("cache reused").next_to(link, UP, buff=0.1)
        self.play(Create(link), FadeIn(lab))
        self.wait(2)


class MetalDispatch(Scene):
    """Grid → threadgroup → simdgroup → thread."""

    def construct(self):
        title = caption("Metal launch hierarchy")
        title.to_edge(UP, buff=0.35)
        self.play(FadeIn(title))

        grid = RoundedRectangle(
            width=12.4,
            height=5.4,
            corner_radius=0.15,
            stroke_color=ACCENT,
            fill_color=BOX_FILL,
            fill_opacity=1,
        )
        grid.shift(DOWN * 0.15)
        gl = Text("dispatch grid  (many threadgroups)", font_size=20, color=ACCENT)
        gl.next_to(grid, UP, buff=0.12)

        tgs = VGroup()
        for i in range(3):
            tg = RoundedRectangle(
                width=3.4,
                height=3.6,
                corner_radius=0.1,
                stroke_color=GREEN,
                fill_opacity=0,
            )
            tgs.add(tg)
        tgs.arrange(RIGHT, buff=0.45)
        tgs.move_to(grid.get_center())

        sg_boxes = VGroup()
        threads = VGroup()
        for tg in tgs:
            sg = RoundedRectangle(
                width=2.8,
                height=1.35,
                corner_radius=0.08,
                stroke_color=ORANGE,
            )
            sg.move_to(tg.get_center() + UP * 0.55)
            sg_boxes.add(sg)
            row = VGroup(
                *[
                    Square(0.28, stroke_color=PURPLE, fill_color=PURPLE, fill_opacity=0.35)
                    for _ in range(8)
                ]
            ).arrange(RIGHT, buff=0.06)
            row.move_to(sg.get_center())
            threads.add(row)

        self.play(FadeIn(grid), FadeIn(gl))
        self.play(LaggedStart(*[Create(tg) for tg in tgs], lag_ratio=0.15))
        tg_lab = note("threadgroup  tg threads, LOCAL memory").next_to(tgs[1], DOWN, buff=0.15)
        self.play(FadeIn(tg_lab), *[Create(s) for s in sg_boxes])
        sg_lab = note("simdgroup = 32 lanes on Apple  (simd_sum)").to_edge(DOWN, buff=0.35)
        self.play(FadeIn(sg_lab), *[FadeIn(r) for r in threads])

        call = Text("gid = which row    lid = which lane in the reduce", font_size=22, color=INK)
        call.next_to(title, DOWN, buff=0.25)
        self.play(FadeIn(call))
        self.wait(2)


class MatvecMulSum(Scene):
    def construct(self):
        title = caption("Matvec is mul + sum  (no Op.MATMUL)")
        title.to_edge(UP, buff=0.4)
        self.play(FadeIn(title))

        w = box("W  [rows × cols]", ACCENT, 3.4, 1.1)
        x = box("x  [cols]", GREEN, 2.6, 1.1)
        y = box("y  [rows]", ORANGE, 2.6, 1.1)
        eq = Text("y[r] = Σ  W[r,c] · x[c]", font_size=28, color=INK)

        row = VGroup(w, x, y).arrange(RIGHT, buff=0.7)
        row.shift(UP * 0.9)
        self.play(FadeIn(w), FadeIn(x))
        times = Text("×", font_size=36, color=MUTED).move_to((w.get_right() + x.get_left()) / 2)
        self.play(FadeIn(times), FadeIn(y))
        eq.next_to(row, DOWN, buff=0.7)
        self.play(FadeIn(eq))

        g = note("Graph:  MulBroadcastRow(W, x)  then  SumReduce(axis=1)")
        k = note("Schedule invents KernelKind::Matvec   renderer prints the K-loop")
        g.to_edge(DOWN, buff=1.1)
        k.next_to(g, DOWN, buff=0.2)
        self.play(FadeIn(g))
        self.play(FadeIn(k))
        self.wait(2)


class CompilerPipeline(Scene):
    def construct(self):
        title = caption("ksearch compiler pipeline")
        title.to_edge(UP, buff=0.4)
        self.play(FadeIn(title))

        names = [
            ("Graph", GREEN),
            ("schedule", ACCENT),
            ("Kernel IR", ORANGE),
            ("MSL", PURPLE),
            ("Metal", ACCENT),
            ("GPU", GREEN),
        ]
        nodes = VGroup(*[box(n, c, 2.15, 1.0) for n, c in names])
        nodes.arrange(RIGHT, buff=0.38)
        nodes.shift(UP * 0.4)

        self.play(LaggedStart(*[FadeIn(n) for n in nodes], lag_ratio=0.18))
        arrows = VGroup(*[arrow(nodes[i], nodes[i + 1]) for i in range(len(nodes) - 1)])
        self.play(LaggedStart(*[Create(a) for a in arrows], lag_ratio=0.12))

        labels = [
            "primitives +\nFuseHint",
            "CALL\nboundary",
            "loops / ALU\nAST",
            "Load expand\nF16/Q4K",
            "compile +\nencode",
            "run",
        ]
        notes = VGroup()
        for n, lab in zip(nodes, labels):
            t = Text(lab, font_size=16, color=MUTED, line_spacing=0.9)
            t.next_to(n, DOWN, buff=0.35)
            notes.add(t)
        self.play(LaggedStart(*[FadeIn(t) for t in notes], lag_ratio=0.1))

        rule = note("Renderer walks the AST only. No named rmsnorm.metal product.")
        rule.to_edge(DOWN, buff=0.4)
        self.play(FadeIn(rule))
        self.wait(2)


class Gemma4Stack(Scene):
    def construct(self):
        title = caption("Gemma 4 E2B  (ksearch path)")
        title.to_edge(UP, buff=0.3)
        self.play(FadeIn(title))

        tok = box("token id", MUTED, 2.2, 0.7)
        emb = box("embed Q4_K gather", ACCENT, 3.2, 0.75)
        ple = box("PLE prepass", PURPLE, 3.2, 0.75)
        swa = box("×5  SWA layers  hd=256", GREEN, 4.4, 0.8)
        full = box("full attn  hd=512", ORANGE, 4.4, 0.8)
        sh = box("shared-KV layers  Q only", MUTED, 4.4, 0.75)
        head = box("RMS + tied lm_head + softcap argmax", ACCENT, 5.6, 0.8)

        col = VGroup(tok, emb, ple, swa, full, sh, head).arrange(DOWN, buff=0.28)
        col.shift(DOWN * 0.15)
        self.play(LaggedStart(*[FadeIn(b) for b in col], lag_ratio=0.12))
        arrows = VGroup(*[down_arrow(col[i], col[i + 1]) for i in range(len(col) - 1)])
        self.play(LaggedStart(*[Create(a) for a in arrows], lag_ratio=0.08))
        self.wait(2)


class GemmaLayer(Scene):
    def construct(self):
        title = caption("One KV-owning layer")
        title.to_edge(UP, buff=0.3)
        self.play(FadeIn(title))

        x = box("x", GREEN, 1.4, 0.7)
        qkv = box("RMS + QKV matvecs", ACCENT, 3.3, 0.8)
        rope = box("per-head RMS + RoPE", ORANGE, 3.3, 0.8)
        pack = box("pack K,V Q4_0", ORANGE, 3.3, 0.8)
        sdpa = box("SDPA hybrid  Q40 KV", PURPLE, 3.3, 0.8)
        o = box("o-proj", ACCENT, 2.2, 0.7)
        mlp = box("gated GELU MLP", GREEN, 3.3, 0.8)
        ple = box("PLE residual", PURPLE, 3.3, 0.8)

        left = VGroup(x, qkv, rope, pack).arrange(DOWN, buff=0.4).shift(LEFT * 3.6)
        right = VGroup(sdpa, o, mlp, ple).arrange(DOWN, buff=0.4).shift(RIGHT * 3.4)

        self.play(FadeIn(left))
        self.play(FadeIn(right))
        a = Arrow(pack.get_right(), sdpa.get_left(), buff=0.15, color=ORANGE, stroke_width=3)
        self.play(Create(a))
        foot = note("Non-owners skip pack; SDPA reads kv_source(layer)")
        foot.to_edge(DOWN, buff=0.4)
        self.play(FadeIn(foot))
        self.wait(2)


class DecodeTokenSeq(Scene):
    """Animated sequence diagram for one decode token."""

    def construct(self):
        title = caption("Sequence: one decode token")
        title.to_edge(UP, buff=0.3)
        self.play(FadeIn(title))

        actors = ["Model", "Eng", "codegen", "Metal", "GPU"]
        xs = [-5.4, -2.7, 0.0, 2.7, 5.4]
        heads = VGroup()
        lines = VGroup()
        for name, x in zip(actors, xs):
            h = box(name, ACCENT, 2.0, 0.65)
            h.move_to([x, 2.6, 0])
            ln = DashedLine(
                [x, 2.2, 0],
                [x, -3.2, 0],
                color=BOX_STROKE,
                dash_length=0.12,
            )
            heads.add(h)
            lines.add(ln)

        self.play(FadeIn(heads), Create(lines))

        steps = [
            (0, 1, "embed / PLE", ACCENT),
            (1, 2, "Graph + lower (first time)", ORANGE),
            (2, 3, "compile MSL", PURPLE),
            (1, 3, "encode dispatch", GREEN),
            (0, 1, "layers: QKV SDPA MLP PLE", ACCENT),
            (1, 3, "more dispatches (same encoder)", GREEN),
            (3, 4, "commit / run", GREEN),
            (4, 0, "argmax token id", ORANGE),
        ]

        y = 1.85
        msgs = VGroup()
        for src, dst, lab, col in steps:
            x0, x1 = xs[src], xs[dst]
            arr = Arrow(
                [x0, y, 0],
                [x1, y, 0],
                buff=0.22,
                stroke_width=2.5,
                color=col,
                max_tip_length_to_length_ratio=0.08,
            )
            t = Text(lab, font_size=16, color=INK)
            t.next_to(arr, UP, buff=0.04)
            if t.width > abs(x1 - x0) - 0.3:
                t.scale_to_fit_width(abs(x1 - x0) - 0.35)
            grp = VGroup(arr, t)
            msgs.add(grp)
            self.play(Create(arr), FadeIn(t), run_time=0.45)
            y -= 0.58

        self.wait(2)
