#!/usr/bin/env python3
"""Generate publication-grade draw.io XML and export PNG/SVG for MMTSFM Stage 1 (Numerical Baseline).

Conference style: NeurIPS / ICLR / ICML
Standards:
- Zero drop shadows, rounded=0 sharp modern academic boxes.
- Print-friendly high-contrast palette: dashed borders for frozen components,
  solid crimson red (#C62828) accent for trainable components.
- Orthogonal routing throughout, straight rectangular residual bypass arrows into (+) circles.
- Strict LaTeX mathematical typesetting enabled via math="1" on mxGraphModel.
- Concise mathematical operators inside boxes; tensor shapes on connecting arrows.
- Integrated compact quantile cone with clearly labeled axes (q10, q50, q90, horizon).
- Explicit Chronos2Encoder visual split: Blocks 1-9 (Frozen) and Blocks 10-12 (Trainable).
- Terminology: Input Patch Embedding (Pretrained), Output Head (Re-initialised).
"""

import subprocess
from pathlib import Path
import xml.etree.ElementTree as ET
from xml.dom import minidom


def build_top_tier_s1_drawio(output_path: str | Path) -> None:
    root = ET.Element("mxGraphModel", math="1", adaptiveColors="auto")
    root_cell_group = ET.SubElement(root, "root")

    ET.SubElement(root_cell_group, "mxCell", id="0")
    ET.SubElement(root_cell_group, "mxCell", id="1", parent="0")

    cell_id = 2

    def get_id() -> str:
        nonlocal cell_id
        cid = str(cell_id)
        cell_id += 1
        return cid

    def add_node(parent: str, val: str, style: str, x: int, y: int, w: int, h: int) -> str:
        nid = get_id()
        cell = ET.SubElement(
            root_cell_group, "mxCell", id=nid, value=val, style=style, vertex="1", parent=parent
        )
        ET.SubElement(
            cell, "mxGeometry", x=str(x), y=str(y), width=str(w), height=str(h), **{"as": "geometry"}
        )
        return nid

    def add_edge_explicit(
        parent: str,
        source_pt: tuple[int, int] | None,
        target_pt: tuple[int, int] | None,
        val: str,
        style: str,
        points=None,
    ) -> str:
        eid = get_id()
        attrs = {"id": eid, "style": style, "edge": "1", "parent": parent}
        if val:
            attrs["value"] = val
        cell = ET.SubElement(root_cell_group, "mxCell", **attrs)
        geo = ET.SubElement(cell, "mxGeometry", relative="1", **{"as": "geometry"})
        if source_pt:
            ET.SubElement(geo, "mxPoint", x=str(source_pt[0]), y=str(source_pt[1]), **{"as": "sourcePoint"})
        if target_pt:
            ET.SubElement(geo, "mxPoint", x=str(target_pt[0]), y=str(target_pt[1]), **{"as": "targetPoint"})
        if points:
            arr = ET.SubElement(geo, "Array", **{"as": "points"})
            for px, py in points:
                ET.SubElement(arr, "mxPoint", x=str(px), y=str(py))
        return eid

    def add_edge(
        parent: str, source: str | None, target: str | None, val: str, style: str, points=None
    ) -> str:
        eid = get_id()
        attrs = {"id": eid, "style": style, "edge": "1", "parent": parent}
        if source:
            attrs["source"] = source
        if target:
            attrs["target"] = target
        if val:
            attrs["value"] = val
        cell = ET.SubElement(root_cell_group, "mxCell", **attrs)
        geo = ET.SubElement(cell, "mxGeometry", relative="1", **{"as": "geometry"})
        if points:
            arr = ET.SubElement(geo, "Array", **{"as": "points"})
            for px, py in points:
                ET.SubElement(arr, "mxPoint", x=str(px), y=str(py))
        return eid

    # Typography & Base styles
    f_academic = "fontFamily=Helvetica,Arial,sans-serif;"

    # Styles
    box_standard = f"rounded=0;strokeWidth=1.5;strokeColor=#1E293B;fillColor=#FFFFFF;html=1;whiteSpace=wrap;align=center;verticalAlign=middle;shadow=0;{f_academic}"
    box_prep = f"rounded=0;strokeWidth=1.5;strokeColor=#334155;fillColor=#FFFFFF;html=1;whiteSpace=wrap;align=center;verticalAlign=middle;shadow=0;{f_academic}"
    box_frozen = f"rounded=0;strokeWidth=1.5;strokeColor=#475569;dashed=1;dashPattern=5 3;fillColor=#F8FAFC;html=1;whiteSpace=wrap;align=center;verticalAlign=middle;shadow=0;{f_academic}"
    box_trainable = f"rounded=0;strokeWidth=1.5;strokeColor=#C62828;fillColor=#FFFFFF;html=1;whiteSpace=wrap;align=center;verticalAlign=middle;shadow=0;{f_academic}"

    box_encoder_container = f"rounded=0;strokeWidth=1.5;strokeColor=#1E293B;fillColor=#FFFFFF;html=1;align=left;verticalAlign=top;shadow=0;{f_academic}"
    box_encoder_block = f"rounded=0;strokeWidth=1.5;strokeColor=#1E293B;fillColor=#FFFFFF;html=1;whiteSpace=wrap;align=center;verticalAlign=middle;shadow=0;{f_academic}"
    style_plus = f"ellipse;whiteSpace=wrap;html=1;aspect=fixed;fillColor=#FFFFFF;strokeColor=#1E293B;strokeWidth=1.5;{f_academic}fontSize=13;fontStyle=1;fontColor=#1E293B;align=center;verticalAlign=middle;shadow=0;"

    # Context patches: white fill with solid dark border
    bar_context = f"rounded=0;strokeWidth=1.5;strokeColor=#1E293B;fillColor=#FFFFFF;html=1;whiteSpace=wrap;align=center;verticalAlign=middle;shadow=0;{f_academic}"
    # Query token: white fill with solid red border
    bar_query = f"rounded=0;strokeWidth=1.5;strokeColor=#C62828;fillColor=#FFFFFF;html=1;whiteSpace=wrap;align=center;verticalAlign=middle;shadow=0;{f_academic}"

    edge_main = "edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;strokeColor=#000000;strokeWidth=1.5;endArrow=block;endFill=1;endSize=5;"
    edge_bypass = "edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;strokeColor=#000000;strokeWidth=1.3;endArrow=block;endFill=1;endSize=5;"
    edge_query = "edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;strokeColor=#C62828;strokeWidth=1.6;endArrow=block;endFill=1;endSize=5;"
    edge_line_only = "edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;strokeColor=#000000;strokeWidth=1.5;endArrow=none;"

    # ================= 1. TOP INPUTS =================
    y_hist = add_node(
        "1",
        '<b style="font-size:13px; color:#0F172A;">PV power history</b><br/>'
        '<span style="font-size:12px;">\\({Y_{\\mathrm{hist}} \\in \\mathbb{R}^{B \\times 672}}\\)</span>',
        box_standard, 90, 30, 340, 50,
    )
    x_cov = add_node(
        "1",
        '<b style="font-size:13px; color:#0F172A;">Known covariates</b><br/>'
        '<span style="font-size:12px;">\\({X_{\\mathrm{cov}} \\in \\mathbb{R}^{B \\times 684 \\times 14}}\\)</span>',
        box_standard, 570, 30, 340, 50,
    )

    # ================= 2. PREPROCESSING BLOCKS =================
    p_hist = add_node(
        "1",
        '<b style="font-size:12px; color:#0F172A;">InstanceNorm(arcsinh) + Patch(\\(P=16\\))</b>',
        box_prep, 90, 108, 340, 42,
    )
    p_cov = add_node(
        "1",
        '<b style="font-size:12px; color:#0F172A;">Per-Channel Patch(\\(P=16\\))</b>',
        box_prep, 570, 108, 340, 42,
    )

    add_edge("1", y_hist, p_hist, "", edge_main)
    add_edge("1", x_cov, p_cov, "", edge_main)

    # ================= 3. INPUT PATCH EMBEDDING (PRETRAINED) =================
    lin_proj = add_node(
        "1",
        '<b style="font-size:12px; color:#1E293B;">Input Patch Embedding (Pretrained)</b>',
        box_frozen, 220, 185, 560, 40,
    )

    add_edge(
        "1", p_hist, lin_proj,
        '<span style="background-color:#FFFFFF; padding:2px; font-size:10px;">\\({\\mathbb{R}^{B \\times 42 \\times 16}}\\)</span>',
        edge_main, points=[(260, 168), (380, 168)],
    )
    add_edge(
        "1", p_cov, lin_proj,
        '<span style="background-color:#FFFFFF; padding:2px; font-size:10px;">\\({14 \\times \\mathbb{R}^{B \\times 43 \\times 16}}\\)</span>',
        edge_main, points=[(740, 168), (620, 168)],
    )

    # ================= 4. MULTIMODAL EMBEDDING + ROPE =================
    # Redundant "(Trainable)" text removed per user request; red border kept
    emb_mm = add_node(
        "1",
        '<b style="font-size:12px; color:#C62828;">MultimodalEmbedding + RoPE</b>',
        box_trainable, 220, 252, 560, 40,
    )

    add_edge(
        "1", lin_proj, emb_mm,
        '<span style="background-color:#FFFFFF; padding:2px; font-size:10px;">\\({\\mathbb{R}^{B \\times 43 \\times 768}}\\) (per row)</span>',
        edge_main,
    )

    # ================= 5. INTERLEAVING SEQUENCE (CONTEXT & QUERY TOKENS) =================
    bar_ctx = add_node(
        "1",
        '<b style="font-size:11px; color:#1E293B;">Context Patches (42)</b>',
        bar_context, 260, 318, 320, 32,
    )
    bar_q = add_node(
        "1",
        '<b style="font-size:11px; color:#C62828;">Query Token (1)</b>',
        bar_query, 595, 318, 145, 32,
    )

    add_edge_explicit("1", (420, 292), (420, 318), "", edge_main)
    add_edge_explicit("1", (667, 292), (667, 318), "", edge_main)

    # Merging Context Patches and Query Token
    add_edge_explicit(
        "1", (420, 350), (500, 375),
        '<span style="background-color:#FFFFFF; padding:2px; font-size:10px;">\\({\\mathbb{R}^{B \\times 42 \\times 768}}\\)</span>',
        edge_line_only, points=[(420, 375)],
    )
    add_edge_explicit(
        "1", (667, 350), (500, 375),
        '<span style="background-color:#FFFFFF; padding:2px; font-size:10px; color:#C62828;">\\({\\mathbb{R}^{B \\times 1 \\times 768}}\\)</span>',
        edge_line_only, points=[(667, 375)],
    )
    add_edge_explicit(
        "1", (500, 375), (500, 405),
        '<span style="background-color:#FFFFFF; padding:2px; font-size:10px; font-weight:bold;">\\({\\mathbb{R}^{B \\times 43 \\times 768}}\\)</span>',
        edge_main,
    )

    # ================= 6. CHRONOS2ENCODER (X 12 BLOCKS) =================
    enc_container = add_node("1", "", box_encoder_container, 40, 405, 920, 265)

    add_node(
        "1",
        '<b style="font-size:13px; color:#0F172A;">Chronos2Encoder (× 12 blocks, Top 3 fine-tuned)</b>',
        f"text;html=1;align=left;verticalAlign=middle;fontSize=13;fontColor=#0F172A;{f_academic}",
        55, 411, 420, 22,
    )
    add_node(
        "1",
        '<span style="font-size:11px; color:#B91C1C; font-weight:bold;">Top 3 blocks fine-tuned (LR 0.1×)</span>',
        f"text;html=1;align=right;verticalAlign=middle;fontSize=11;fontColor=#B91C1C;{f_academic}",
        660, 411, 280, 22,
    )

    # Section A: Blocks 1–9 (Frozen)
    b1_9 = add_node(
        "1",
        '<b>Blocks 1–9 (Frozen)</b>&nbsp;&nbsp;·&nbsp;&nbsp;<span style="font-size:11px; color:#475569;">9 × identical pretrained Transformer blocks</span>',
        box_frozen, 60, 438, 880, 40,
    )

    # Section B: Blocks 10–12 (Trainable) - solid red border, white fill
    b10_12 = add_node("1", "", box_trainable, 60, 508, 880, 150)

    # Title inside Blocks 10-12 positioned at x=160
    add_node(
        "1", '<b style="font-size:12px; color:#C62828;">Blocks 10–12 (Trainable)</b>',
        f"text;html=1;align=left;verticalAlign=middle;fontSize=12;fontColor=#C62828;{f_academic}",
        160, 514, 200, 20,
    )

    # Signal passing from Blocks 1-9 to Blocks 10-12:
    # Straight vertical line at x=98 from Blocks 1-9 down to y=587 into TimeSelfAttention input
    add_edge_explicit("1", (98, 478), (98, 587), "", edge_line_only)

    # Clean label node for the tensor shape between Blocks 1-9 and 10-12
    add_node(
        "1",
        '<span style="background-color:#FFFFFF; padding:1px 3px; font-size:10px;">\\({\\mathbb{R}^{B \\times 43 \\times 768}}\\)</span>',
        f"text;html=1;align=left;verticalAlign=middle;fontSize=10;{f_academic}",
        105, 483, 110, 18,
    )

    # Inside Blocks 10–12: 3 Sub-blocks with straight horizontal lines and clean residual bypasses
    # Sub-block 1: TimeSelfAttention
    b_time = add_node(
        "1",
        '<b style="font-size:12px; color:#0F172A;">TimeSelfAttention + RoPE</b><br/>'
        '<span style="font-size:10px; color:#475569;">temporal axis (\\(T=43\\))</span>',
        box_encoder_block, 150, 555, 185, 65,
    )
    plus1 = add_node("1", "+", style_plus, 355, 577, 21, 21)

    # Sub-block 2: GroupSelfAttention
    b_group = add_node(
        "1",
        '<b style="font-size:12px; color:#0F172A;">GroupSelfAttention</b><br/>'
        '<span style="font-size:10px; color:#475569;">cross-variable axis (\\(V=15\\))</span>',
        box_encoder_block, 395, 555, 190, 65,
    )
    plus2 = add_node("1", "+", style_plus, 605, 577, 21, 21)

    # Sub-block 3: FeedForward
    b_ffn = add_node(
        "1",
        '<b style="font-size:12px; color:#0F172A;">FeedForward</b><br/>'
        '<span style="font-size:10px; color:#475569;">\\(d_{\\mathrm{ff}} = 3072\\)</span>',
        box_encoder_block, 645, 555, 180, 65,
    )
    plus3 = add_node("1", "+", style_plus, 845, 577, 21, 21)

    # Horizontal entry from x=98 into TimeSelfAttention
    add_edge_explicit("1", (98, 587), (150, 587), "", edge_main)

    # Residual Bypass 1 (over TimeSelfAttention)
    add_edge_explicit("1", (98, 538), (365, 577), "", edge_bypass, points=[(365, 538)])
    add_edge("1", b_time, plus1, "", edge_main)

    # From plus1 to GroupSelfAttention & Bypass 2
    add_edge_explicit("1", (376, 587), (395, 587), "", edge_main)
    add_edge_explicit("1", (385, 587), (615, 577), "", edge_bypass, points=[(385, 538), (615, 538)])
    add_edge("1", b_group, plus2, "", edge_main)

    # From plus2 to FeedForward & Bypass 3
    add_edge_explicit("1", (626, 587), (645, 587), "", edge_main)
    add_edge_explicit("1", (635, 587), (855, 577), "", edge_bypass, points=[(635, 538), (855, 538)])
    add_edge("1", b_ffn, plus3, "", edge_main)

    # ================= 7. OUTPUT HEAD & QUANTILE PLOT =================
    out_head = add_node(
        "1",
        '<b style="font-size:12px; color:#C62828;">Output Head (Re-initialised)</b><br/>'
        '<span style="font-size:11px; color:#1E293B;">\\({\\mathrm{Linear}(768 \\rightarrow 12 \\times 9)}\\)</span>',
        box_trainable, 80, 720, 300, 52,
    )

    # Future Query State arrow:
    # Originates from the output of the final residual add (plus3) at x=855, y=598
    # Routes downward out of the encoder to y=685, then left to x=230, pointing down into Output Head
    add_edge_explicit(
        "1", (855, 598), (230, 720), "", edge_query,
        points=[(855, 685), (230, 685)],
    )

    # Annotation for Future Query State placed on the empty segment between x=260 and x=510
    add_node(
        "1",
        '<span style="background-color:#FFFFFF; padding:2px; font-size:11px; font-weight:bold; color:#C62828;">Future Query State: \\({\\mathbb{R}^{B \\times 1 \\times 768}}\\)</span>',
        f"text;html=1;align=center;verticalAlign=middle;fontSize=11;fontColor=#C62828;{f_academic}",
        280, 673, 240, 24,
    )

    # Straight arrow from Output Head to Quantile Plot
    add_edge_explicit(
        "1", (380, 746), (530, 746),
        '<span style="background-color:#FFFFFF; padding:2px; font-size:11px; font-weight:bold; color:#0F172A;">\\({\\hat{Y} \\in \\mathbb{R}^{B \\times 12 \\times 9}}\\)</span>',
        edge_main,
    )

    # Compact Quantile Plot Visual:
    # Outer trapezoid (light red): y from 721 to 771 (height 50), center y = 746
    add_node(
        "1", "",
        "shape=trapezoid;perimeter=trapezoidPerimeter;whiteSpace=wrap;html=1;fixedSize=1;direction=south;fillColor=#FEE2E2;strokeColor=none;opacity=80;size=22;",
        535, 721, 140, 50,
    )
    # Inner trapezoid (medium red): y from 731 to 761 (height 30)
    add_node(
        "1", "",
        "shape=trapezoid;perimeter=trapezoidPerimeter;whiteSpace=wrap;html=1;fixedSize=1;direction=south;fillColor=#FECACA;strokeColor=none;opacity=90;size=14;",
        535, 731, 140, 30,
    )
    # Median line (q50): horizontal red line at y = 746
    add_node("1", "", "line;strokeColor=#DC2626;strokeWidth=2.2;html=1;", 535, 746, 140, 2)

    # Quantile axis labels: properly aligned to y-axis of triangle
    add_node(
        "1", "\\({q_{90}}\\)",
        f"text;html=1;align=left;verticalAlign=middle;fontSize=10;fontColor=#991B1B;{f_academic}",
        682, 713, 35, 16,
    )
    add_node(
        "1", "\\({q_{50}}\\)",
        f"text;html=1;align=left;verticalAlign=middle;fontSize=10;fontColor=#DC2626;fontStyle=1;{f_academic}",
        682, 738, 35, 16,
    )
    add_node(
        "1", "\\({q_{10}}\\)",
        f"text;html=1;align=left;verticalAlign=middle;fontSize=10;fontColor=#991B1B;{f_academic}",
        682, 763, 35, 16,
    )

    # Forecast Title & x-axis label centered below the triangle
    add_node(
        "1", '<b style="font-size:12px; color:#0F172A;">Probabilistic Quantile Forecast</b>',
        f"text;html=1;align=center;verticalAlign=middle;fontSize=12;fontColor:#0F172A;{f_academic}",
        525, 698, 160, 18,
    )
    add_node(
        "1", "\\({t+1 \\;\\cdots\\; t+12\\text{ (6h)}}\\)",
        f"text;html=1;align=center;verticalAlign=middle;fontSize=9;fontColor=#475569;{f_academic}",
        535, 776, 140, 14,
    )

    xml_str = ET.tostring(root, encoding="utf-8")
    parsed = minidom.parseString(xml_str)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(parsed.toxml())


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent.parent
    drawio_path = repo_root / "manuscript" / "figures" / "mmtsfm_s1.drawio"
    png_path = repo_root / "manuscript" / "figures" / "mmtsfm_s1.png"
    svg_path = repo_root / "manuscript" / "figures" / "mmtsfm_s1.svg"
    report_png = repo_root / "report" / "figures" / "mmtsfm_s1.png"
    report_svg = repo_root / "report" / "figures" / "mmtsfm_s1.svg"
    report_drawio = repo_root / "report" / "figures" / "mmtsfm_s1.drawio"

    print(f"Building {drawio_path}...")
    build_top_tier_s1_drawio(drawio_path)

    print("Exporting PNG (with embedded XML, 2x scale, border 15)...")
    subprocess.run(
        ["drawio", "-x", "-f", "png", "-e", "-b", "15", "-s", "2", "-o", str(png_path), str(drawio_path)],
        check=True,
    )

    print("Exporting SVG (with embedded XML)...")
    subprocess.run(
        ["drawio", "-x", "-f", "svg", "-e", "-b", "15", "-o", str(svg_path), str(drawio_path)],
        check=True,
    )

    print("Syncing to report/figures/...")
    report_png.write_bytes(png_path.read_bytes())
    report_svg.write_bytes(svg_path.read_bytes())
    report_drawio.write_bytes(drawio_path.read_bytes())

    print("Done!")


if __name__ == "__main__":
    main()
