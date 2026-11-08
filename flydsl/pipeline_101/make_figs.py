"""根据 results/ 画 figs/*.svg。用法：python make_figs.py（先跑 run.py）"""
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
FIGS = os.path.join(HERE, "figs")
FONT = 'font-family="Noto Sans CJK SC, DejaVu Sans, Arial, sans-serif"'
COL = {"A": "#8ecae6", "B": "#1d6fa5", "M": "#b8b8b8", "W": "#d62828", "sunk": "#f77f00",
       "ok": "#2a9d8f", "warn": "#e9c46a", "bad": "#e76f51", "stall": "#fbd5d5"}


# ---------------------------------------------------------------- 基础

def svg(w, h, body):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">'
            f'<rect width="{w}" height="{h}" fill="white"/>{"".join(body)}</svg>')


def text(x, y, s, size=12, anchor="start", color="#222", weight="normal"):
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" text-anchor="{anchor}" fill="{color}" '
            f'font-weight="{weight}" {FONT}>{s}</text>')


def rect(x, y, w, h, fill, stroke="none", sw=1, rx=0, opacity=1):
    return (f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(w, 0.5):.1f}" height="{h:.1f}" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="{sw}" rx="{rx}" opacity="{opacity}"/>')


def legend(x, y, items):
    out = []
    for color, label in items:
        out += [rect(x, y - 10, 14, 12, color), text(x + 19, y, label, 12)]
        x += 30 + 13 * len(label)
    return out


def save(name, content):
    os.makedirs(FIGS, exist_ok=True)
    with open(os.path.join(FIGS, name), "w") as f:
        f.write(content)
    print("写入", os.path.join("figs", name))


# ---------------------------------------------------------------- 从 ISA 得到事件（和 common.events 相同，这里不 import torch）

def reg_set(s):
    m = re.match(r"([av])\[(\d+):(\d+)\]", s) or re.match(r"([av])(\d+)()", s)
    lo = int(m.group(2))
    return {(m.group(1), r) for r in range(lo, int(m.group(3) or lo) + 1)}


def events(path):
    loop = [l.strip() for l in open(path) if l.strip()]
    loads, kinds, ev = [], [], []
    for l in loop:
        if l.startswith("global_load"):
            loads.append(reg_set(l.split()[1].rstrip(",")))
            kinds.append("?")
            ev.append(["L", len(loads) - 1])
        elif l.startswith("v_mfma"):
            ops = [x.strip() for x in l.split(None, 1)[1].split(",")]
            ra, rb = reg_set(ops[1]), reg_set(ops[2])
            for i, r in enumerate(loads):
                if kinds[i] == "?":
                    kinds[i] = "A" if ra & r else "B" if rb & r else "?"
            used = []
            for rr in (ra, rb):
                idx = [i for i, r in enumerate(loads) if rr & r]
                used.append(idx[-1] if idx else None)
            ev.append(["M", *used])
        else:
            m = re.match(r"s_waitcnt vmcnt\((\d+)\)", l)
            if m:
                ev.append(["W", int(m.group(1))])
    # 给 load 起名：A0..A7、B0..B7（按发出顺序）
    names, cnt = [], {"A": 0, "B": 0, "?": 0}
    for k in kinds:
        names.append(f"{k}{cnt[k]}")
        cnt[k] += 1
    for e in ev:
        if e[0] == "L":
            e += [kinds[e[1]], names[e[1]]]
    return ev, names


def annotate(ev):
    """标出下沉的 load、每次等待等的是哪条 load、是否串行等待"""
    n_mfma, issued_at, n_load = 0, [], 0
    for e in ev:
        if e[0] == "L":
            issued_at.append(n_mfma)
            e.append(n_mfma > 0)                      # 下沉？
            n_load += 1
        elif e[0] == "M":
            n_mfma += 1
        else:
            k = n_load - e[1] - 1
            e.append(k)
            e.append(k >= 0 and n_mfma > 0 and issued_at[k] > 0 and n_mfma - issued_at[k] <= 2)   # 串行等待？
    return ev


# ---------------------------------------------------------------- 简单的时间模型

def simulate(ev, L, t_load=0.25, t_mfma=1.0):
    """按顺序发射：load 占 t_load，MFMA 占 t_mfma；load 发出 L 之后完成，按发出顺序完成；
    s_waitcnt vmcnt(n) 一直等到除最近 n 条以外的 load 都完成。时间单位 = 一条 MFMA"""
    t, issue, done, mf, stalls = 0.0, [], [], [], []
    for e in ev:
        if e[0] == "L":
            issue.append(t)
            done.append(max(t + L, done[-1] if done else 0))
            t += t_load
        elif e[0] == "W":
            k = len(issue) - e[1] - 1
            if k >= 0 and done[k] > t:
                stalls.append((t, done[k], k))
                t = done[k]
        else:
            mf.append((t, e))
            t += t_mfma
    return {"issue": issue, "done": done, "mfma": mf, "stalls": stalls, "total": t}


# ---------------------------------------------------------------- 图 1：13 种遍历顺序

def fig_orders(data):
    rows = data["orders"]
    names = list(rows)
    W, H, x0, y0, bh = 760, 500, 150, 60, 24
    vmax = 500
    out = [text(20, 28, "fx.gemm 的 13 种遍历顺序（BM=BN=128，gfx950 实测）", 16, weight="bold")]
    for i, n in enumerate(names):
        r = rows[n]
        y = y0 + i * (bh + 6)
        s = r["串行等待"]
        color = COL["ok"] if s == 0 else COL["warn"] if s <= 1 else COL["bad"]
        w = (W - x0 - 160) * r["TFLOPS"] / vmax
        out += [text(x0 - 10, y + 17, "None（默认）" if n == "None" else n, 13, anchor="end"),
                rect(x0, y, w, bh, color, rx=3),
                text(x0 + w + 6, y + 17, f'{r["TFLOPS"]} TFLOPS，串行等待 {s} 次', 12)]
    out += legend(x0, H - 20, [(COL["ok"], "串行等待 0 次"), (COL["warn"], "1 次"), (COL["bad"], "≥ 3 次")])
    save("orders.svg", svg(W, H, out))


# ---------------------------------------------------------------- 图 2：主循环的指令顺序（全貌）

ROWS = [("tiled_default__no_sched.s", "① tiled 默认顺序，关掉调度器（≈ 调度前）"),
        ("tiled_default__default.s", "② tiled 默认顺序，默认调度（调度后）"),
        ("tiled_MNK__default.s", "③ tiled MNK，默认调度"),
        ("tiled_default__sched_barrier.s", "④ tiled 默认顺序 + sched_barrier(0)")]


def fig_issue(data):
    WL, WM, WW, GAP = 12, 5, 3, 1
    row_h, x0 = 150, 20
    out = [text(20, 26, "主循环一轮的指令顺序（BM=BN=128，从左到右是发射顺序，gfx942 ISA）", 16, weight="bold")]
    width = 0
    for r, (fname, title) in enumerate(ROWS):
        ev = annotate(events(os.path.join(RES, "isa", fname))[0])
        base = 60 + r * row_h + 95                    # 指令条的上沿
        x, pos = x0, {}
        serial = sum(1 for e in ev if e[0] == "W" and e[3])
        sunk = sum(1 for e in ev if e[0] == "L" and e[4])
        out.append(text(x0, base - 70, f"{title}：下沉 {sunk} 条，串行等待 {serial} 次", 13, weight="bold"))
        load_x, waits = {}, []
        for e in ev:
            if e[0] == "L":
                out.append(rect(x, base, WL, 26, COL[e[2]], stroke=COL["sunk"] if e[4] else "none", sw=2.5))
                load_x[e[1]] = x + WL / 2
                x += WL + GAP
            elif e[0] == "M":
                out.append(rect(x, base + 6, WM, 14, COL["M"]))
                x += WM + GAP
            else:
                out.append(rect(x, base - 4, WW, 34, COL["W"]))
                waits.append((x + WW / 2, e[2], e[3]))
                x += WW + GAP
        for wx, k, ser in waits:                      # 从等待画弧线到被等的 load
            if k < 0 or k not in load_x:
                continue
            lx = load_x[k]
            h = min(60, 12 + abs(wx - lx) / 6)
            color = COL["W"] if ser else "#999"
            out.append(f'<path d="M{wx:.1f},{base - 4} C{wx:.1f},{base - 4 - h} {lx:.1f},{base - 4 - h} {lx:.1f},{base}" '
                       f'fill="none" stroke="{color}" stroke-width="{2 if ser else 1}"/>')
        width = max(width, x)
    H = 60 + len(ROWS) * row_h + 30
    out += legend(x0, H - 14, [(COL["A"], "load A"), (COL["B"], "load B"), (COL["sunk"], "橙框 = 被下沉的 load"),
                               (COL["M"], "MFMA"), (COL["W"], "s_waitcnt vmcnt，弧线指向它等的 load（红 = 串行等待）")])
    save("issue_order.svg", svg(max(width + 30, 1000), H, out))


# ---------------------------------------------------------------- 图 3：放大开头部分，带名字

def fig_issue_zoom(data, n_mfma=20):
    out = [text(20, 26, f"放大每轮开头（前 {n_mfma} 条 MFMA）：每条指令在做什么", 16, weight="bold")]
    rows = ROWS[:3]
    row_h, x0 = 120, 20
    width = 0
    for r, (fname, title) in enumerate(rows):
        ev, names = events(os.path.join(RES, "isa", fname))
        ev = annotate(ev)
        base = 80 + r * row_h
        out.append(text(x0, base - 30, title, 13, weight="bold"))
        x, seen = x0, 0
        for e in ev:
            if e[0] == "M":
                seen += 1
                if seen > n_mfma:
                    break
                label = f"{names[e[1]]}·{names[e[2]]}" if e[1] is not None and e[2] is not None else "MFMA"
                out += [rect(x, base, 44, 26, COL["M"], rx=2), text(x + 22, base + 17, label, 10, anchor="middle")]
                x += 46
            elif e[0] == "L":
                out += [rect(x, base, 26, 26, COL[e[2]], stroke=COL["sunk"] if e[4] else "none", sw=3, rx=2),
                        text(x + 13, base + 17, e[3], 10, anchor="middle", color="white" if e[2] == "B" else "#222")]
                x += 28
            else:
                k = e[2]
                label = f"等{names[k]}" if k >= 0 else "等"
                out += [rect(x, base - 6, 34, 38, "white", stroke=COL["W"], sw=2, rx=3),
                        text(x + 17, base + 17, label, 10, anchor="middle", color=COL["W"], weight="bold" if e[3] else "normal")]
                x += 36
        width = max(width, x)
    H = 80 + len(rows) * row_h
    out.append(text(x0, H - 22, "MFMA 上写的是它用到的 A 块 · B 块；\"等Bk\" = s_waitcnt vmcnt(n)，等到 Bk 及之前发出的 load 都到达；"
                    "橙框 = 被调度器挪到 MFMA 之间的 load", 12, color="#444"))
    save("issue_zoom.svg", svg(max(width + 30, 1100), H, out))


# ---------------------------------------------------------------- 图 4：时间轴（模型）

def calibrate(data):
    """取 L，使模型里 ② / ① 的耗时比等于实测的 TFLOPS 比"""
    t = data["sched"]["tiled 默认"]
    fast = float(t["关 pre-RA + post-RA"].split(" / ")[0])
    slow = float(t["默认"].split(" / ")[0])
    target = fast / slow
    ev_slow = events(os.path.join(RES, "isa", "tiled_default__default.s"))[0]
    ev_fast = events(os.path.join(RES, "isa", "tiled_default__no_sched.s"))[0]
    best = min((abs(simulate(ev_slow, L)["total"] / simulate(ev_fast, L)["total"] - target), L)
               for L in [x / 2 for x in range(2, 200)])
    return best[1], target


def gantt(out, ev, sim, y, scale, x0, title, n_loads=16):
    """画一个时间轴：上面每行一条 load 的在途时间，下面一行 MFMA，红底 = 停顿"""
    kinds = [e for e in ev if e[0] == "L"]
    out.append(text(x0, y - 8, title, 13, weight="bold"))
    lane_h = 5
    ym = y + n_loads * (lane_h + 1) + 6
    for s0, s1, _ in sim["stalls"]:                  # 先画停顿，放在最底层
        out.append(rect(x0 + s0 * scale, y - 2, (s1 - s0) * scale, ym - y + 22, COL["stall"]))
    for i, (t0, t1) in enumerate(zip(sim["issue"], sim["done"])):
        yy = y + i * (lane_h + 1)
        out.append(rect(x0 + t0 * scale, yy, (t1 - t0) * scale, lane_h, COL[kinds[i][2]],
                        stroke=COL["sunk"] if kinds[i][4] else "none", sw=1))
    for t0, _ in sim["mfma"]:
        out.append(rect(x0 + t0 * scale, ym, scale * 0.9, 16, COL["M"]))
    out.append(text(x0 + sim["total"] * scale + 6, ym + 13, f'一轮 ≈ {sim["total"]:.0f}', 12, weight="bold"))
    return ym + 16


def fig_timeline(data):
    L, target = calibrate(data)
    scale, x0 = 5.0, 20
    out = [text(20, 26, "时间轴示意：同一轮循环在模型里怎样执行", 16, weight="bold"),
           text(20, 46, f"模型：按顺序发射，一条 MFMA = 1 个时间单位，load 发出后 L = {L:g} 个单位到达（按发出顺序）。"
                f"L 的取值使 ②/① 的耗时比等于实测的 {target:.2f}", 12, color="#444")]
    y, width = 90, 0
    for fname, title in [ROWS[0], ROWS[1], ROWS[2]]:
        ev = annotate(events(os.path.join(RES, "isa", fname))[0])
        sim = simulate(ev, L)
        stall = sum(s1 - s0 for s0, s1, _ in sim["stalls"])
        y = gantt(out, ev, sim, y, scale, x0, f"{title}：停顿 {stall:.0f}（{len(sim['stalls'])} 次）") + 50
        width = max(width, x0 + sim["total"] * scale + 120)
    out += legend(x0, y - 10, [(COL["A"], "load A 在途"), (COL["B"], "load B 在途"), (COL["sunk"], "橙框 = 下沉的 load"),
                               (COL["M"], "MFMA"), (COL["stall"], "停顿（在等 load）")])
    save("timeline.svg", svg(max(width, 900), y + 10, out))


# ---------------------------------------------------------------- 图 5：原理示意（小例子）

def toy(tokens):
    """tokens：'A0' 发 load，'A0B1' 一条 MFMA（自动在前面插入需要的等待）"""
    ev, idx = [], {}
    for t in tokens:
        if len(t) == 2:
            idx[t] = len(idx)
            ev.append(["L", idx[t], t[0], t, False])
        else:
            need = max(idx[t[:2]], idx[t[2:]])
            ev.append(["W", len(idx) - need - 1, need, False])
            ev.append(["M", idx[t[:2]], idx[t[2:]]])
    first_m = next(i for i, e in enumerate(ev) if e[0] == "M")
    for i, e in enumerate(ev):
        if e[0] == "L":
            e[4] = i > first_m
    return ev


def fig_mechanism():
    A = ["A0", "A1", "A2", "A3"]
    cases = [
        ("(a) 全部 load 先发，再算（N 在最内也没关系）",
         A + ["B0", "B1", "B2", "B3"] + [f"{a}B{j}" for a in A for j in range(4)]),
        ("(b) B1~B3 被挪到 MFMA 之间，且 N 在最内：每个刚发的 B 马上就要用",
         A + ["B0", "A0B0", "B1", "A0B1", "B2", "A0B2", "B3", "A0B3"] + [f"{a}B{j}" for a in A[1:] for j in range(4)]),
        ("(c) 同样被挪，但 M 在最内：每个 B 发出后还有 4 条 MFMA 可做",
         A + ["B0", "A0B0", "B1", "A1B0", "B2", "A2B0", "B3", "A3B0"] + [f"{a}B{j}" for j in range(1, 4) for a in A]),
    ]
    L, scale, x0 = 6, 22, 20
    out = [text(20, 26, "原理示意：load 被挪到 MFMA 之间时，延迟会不会串起来", 16, weight="bold"),
           text(20, 46, f"一条 MFMA = 1 个单位，load 发出后 {L} 个单位到达；每个格子上写的是它用到的 A 块、B 块", 12, color="#444")]
    y, width = 90, 0
    for title, tokens in cases:
        ev = toy(tokens)
        sim = simulate(ev, L, t_load=0.25)
        out.append(text(x0, y - 8, title, 13, weight="bold"))
        loads = [e for e in ev if e[0] == "L"]
        ym = y + len(loads) * 13 + 8
        for s0, s1, k in sim["stalls"]:              # 先画停顿，放在最底层
            out.append(rect(x0 + s0 * scale, y - 2, (s1 - s0) * scale, ym - y + 30, COL["stall"]))
        for i, (t0, t1) in enumerate(zip(sim["issue"], sim["done"])):
            yy = y + i * 13
            out += [rect(x0 + t0 * scale, yy, (t1 - t0) * scale, 11, COL[loads[i][2]],
                         stroke=COL["sunk"] if loads[i][4] else "none", sw=2),
                    text(x0 + t1 * scale + 4, yy + 10, f"{loads[i][3]} 到达", 10, color="#555")]
        for t0, e in sim["mfma"]:
            out += [rect(x0 + t0 * scale + 1, ym, scale - 2, 22, COL["M"], rx=2),
                    text(x0 + t0 * scale + scale / 2, ym + 15, f"{loads[e[1]][3][1]}{loads[e[2]][3][1]}", 9, anchor="middle")]
        stall = sum(s1 - s0 for s0, s1, _ in sim["stalls"])
        out.append(text(x0 + sim["total"] * scale + 8, ym + 15, f"总时间 {sim['total']:.0f}，停顿 {stall:.0f}", 12, weight="bold"))
        width = max(width, x0 + sim["total"] * scale + 180)
        y = ym + 70
    out.append(text(x0, y - 30, "MFMA 格子里的两个数字 = A 块号、B 块号（如 \"02\" = A0·B2）；红底 = 在等 load", 12, color="#444"))
    out += legend(x0, y - 8, [(COL["A"], "load A 在途"), (COL["B"], "load B 在途"), (COL["sunk"], "橙框 = 被挪到 MFMA 之间的 load"),
                              (COL["M"], "MFMA"), (COL["stall"], "停顿")])
    save("mechanism.svg", svg(max(width, 900), y + 6, out))


# ---------------------------------------------------------------- 图 6：关掉调度器 / sched_barrier

def fig_sched(data):
    rows = data["sched"]
    settings = list(next(iter(rows.values())))
    colors = ["#e76f51", "#f4a261", "#2a9d8f"]
    W, x0, y0, bh = 820, 130, 70, 16
    out = [text(20, 28, "关掉 LLVM 的机器指令调度器后（BM=BN=128，gfx950 实测 TFLOPS）", 16, weight="bold")]
    y = y0
    for name, r in rows.items():
        out.append(text(x0 - 10, y + 3 * bh / 2 + 10, name, 13, anchor="end"))
        for j, s in enumerate(settings):
            tf, sunk, ser = r[s].split(" / ")
            w = (W - x0 - 230) * float(tf) / 500
            out += [rect(x0, y + j * (bh + 2), w, bh, colors[j], rx=2),
                    text(x0 + w + 5, y + j * (bh + 2) + 12, f"{tf}（下沉 {sunk}，串行等待 {ser}）", 11)]
        y += 3 * (bh + 2) + 16
    out += legend(x0, y + 14, [(colors[0], "默认"), (colors[1], "关 pre-RA 调度"), (colors[2], "再关 post-RA 调度")])
    save("sched.svg", svg(W, y + 30, out))


if __name__ == "__main__":
    data = json.load(open(os.path.join(RES, "results.json")))
    fig_orders(data)
    fig_issue(data)
    fig_issue_zoom(data)
    fig_timeline(data)
    fig_mechanism()
    fig_sched(data)
