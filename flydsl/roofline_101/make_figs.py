"""根据 results/results.json 画 figs/demand_vs_supply.svg。用法：python make_figs.py（先跑 run.py）"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
FONT = 'font-family="Noto Sans CJK SC, DejaVu Sans, Arial, sans-serif"'
K16, K32, CEIL, HIT = "#8ecae6", "#1d6fa5", "#d62828", "#f77f00"


def text(x, y, s, size=12, anchor="middle", color="#222", weight="normal"):
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" text-anchor="{anchor}" fill="{color}" '
            f'font-weight="{weight}" {FONT}>{s}</text>')


def rect(x, y, w, h, fill, stroke="none", dash=""):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" fill="{fill}" stroke="{stroke}" stroke-width="1.5"{d}/>'


def line(x1, y1, x2, y2, color, dash="", width=2):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" stroke-width="{width}"{d}/>'


def main():
    r = json.load(open(os.path.join(HERE, "results", "results.json")))
    need, got = r["demand"], r["measured"]
    ceil_g = got["load_v0"]["global load B/clk/CU"]
    ceil_hit = got["load_hit"]["global load B/clk/CU"]
    ceil_lds = got["lds_read"]["LDS B/clk/CU"]
    groups = [("v0_preshuffle：global load", "v0p", "global B/clk/CU", "global load B/clk/CU", [(ceil_g, CEIL, ""), (ceil_hit, HIT, "")]),
              ("v2：global load", "v2", "global B/clk/CU", "global load B/clk/CU", [(ceil_g, CEIL, ""), (ceil_hit, HIT, "")]),
              ("v2：LDS（读 + 写）", "v2", "LDS B/clk/CU", "LDS B/clk/CU", [(ceil_lds, CEIL, "ds_read_b128 实测上限")])]
    W, H, top, bottom, left = 860, 500, 70, 380, 70
    ymax = 200
    y = lambda v: bottom - (bottom - top) * v / ymax
    body = [text(W / 2, 28, "每个 CU 每周期要搬的字节：MFMA 满速时的需求（虚框）、实测（实心）与上限（横线）", 15, weight="bold")]
    for v in range(0, ymax + 1, 50):                                    # y 轴
        body += [line(left, y(v), W - 20, y(v), "#e5e5e5", width=1), text(left - 8, y(v) + 4, v, 11, "end", "#666")]
    body.append(text(18, (top + bottom) / 2, "B / clk / CU", 12, color="#444").replace("<text", f'<text transform="rotate(-90 18 {(top + bottom) / 2})"'))
    gw = (W - left - 20) / len(groups)
    for g, (title, kern, nk, gk, ceils) in enumerate(groups):
        x0 = left + g * gw
        for i, (sfx, color, label) in enumerate([("k16", K16, "16x16x16"), ("k32", K32, "16x16x32")]):
            case = f"{kern}_{sfx}"
            bx, bw = x0 + gw * (0.2 + 0.32 * i), gw * 0.26
            d, a = need[case][nk], got[case][gk]
            body += [rect(bx, y(d), bw, bottom - y(d), "none", color, "5,3"),
                     rect(bx, y(a), bw, bottom - y(a), color),
                     text(bx + bw / 2, y(max(d, a)) - 6, f"需 {d:g} / 实 {a:g}", 11),
                     text(bx + bw / 2, bottom + 16, label, 11),
                     text(bx + bw / 2, bottom + 31, f"MFMA {got[case]['MFMA 忙碌']}", 11, color="#555")]
        for v, color, label in ceils:
            body.append(line(x0 + gw * 0.12, y(v), x0 + gw * 0.92, y(v), color, "7,4"))
            if label:
                body.append(text(x0 + gw * 0.92, y(v) - 5, f"{label} {v:g}", 11, "end", color))
        body.append(text(x0 + gw / 2, bottom + 52, title, 13, weight="bold"))
    lx, ly = left, H - 42
    body += [rect(lx, ly - 11, 14, 12, K16), text(lx + 20, ly, "16x16x16：每 CU 每周期 2048 FLOP（= MI300X）", 12, "start"),
             rect(lx + 380, ly - 11, 14, 12, K32), text(lx + 400, ly, "16x16x32：4096 FLOP（MI350X 满速）", 12, "start"),
             line(lx, ly + 20, lx + 24, ly + 20, CEIL, "7,4"),
             text(lx + 30, ly + 24, f"只做同样的 global load 的实测上限 {ceil_g:g}", 12, "start"),
             line(lx + 380, ly + 20, lx + 404, ly + 20, HIT, "7,4"),
             text(lx + 410, ly + 24, f"同样的指令全部命中 L1 {ceil_hit:g}", 12, "start")]
    os.makedirs(os.path.join(HERE, "figs"), exist_ok=True)
    with open(os.path.join(HERE, "figs", "demand_vs_supply.svg"), "w") as f:
        f.write(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">'
                f'<rect width="{W}" height="{H}" fill="white"/>{"".join(body)}</svg>\n')


if __name__ == "__main__":
    main()
