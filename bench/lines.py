#!/usr/bin/env python3
"""Mide dónde corta los versos la hoja (compose.group_lines) contra la letra de Moises (bench/*_letra.txt).

Las palabras de Whisper se alinean con las de la referencia; cada par de palabras seguidas que casa con un par
seguido de la referencia es un ejemplo: "aquí va salto de línea" o "aquí no". Se cuentan aciertos de salto
(precisión, cobertura, F1). Lo que no casa (palabras mal oídas) no cuenta.

Uso (Python de la app, con PYTHONPATH=src):
  python bench/lines.py           F1 por canción y líneas mal cortadas
  python bench/lines.py -v        además, la hoja entera (⟂ = falta un salto, ✗ = salto que sobra)
  python bench/lines.py --train   entrena el modelo de saltos (dejando una canción fuera) y prueba ajustes
"""
from __future__ import annotations

import difflib
import re
import sys
import unicodedata
from pathlib import Path

HERE = Path(__file__).parent
OUT = Path("~/Library/Application Support/audio2midi").expanduser()
SONGS = [("Funeral en el Jardín", "funeral_letra.txt"),
         ("Tu sonrisa sigue aqui-Bm-119bpm-440hz", "sonrisa_letra.txt"),
         ("Obsesion (Edit)", "obsesion_letra.txt")]


def norm(w: str) -> str:
    w = unicodedata.normalize("NFD", w.lower())
    return re.sub(r"[^a-zñ]", "", "".join(c for c in w if unicodedata.category(c) != "Mn"))


def reference(path: Path):
    """Words of the reference and, for each, whether a line ends after it."""
    words, ends = [], []
    for ln in path.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#") or ln.startswith("["):
            continue
        toks = [norm(t) for t in ln.split()]
        toks = [t for t in toks if t]
        if not toks:
            continue
        words += toks
        ends += [False] * (len(toks) - 1) + [True]
    return words, ends


def labels(asr_words, ref_path):
    """{i: True/False}: does a line break go between asr word i and i+1 (only where both align)."""
    rw, rend = reference(ref_path)
    aw = [norm(w["w"]) for w in asr_words]
    sm = difflib.SequenceMatcher(None, aw, rw, autojunk=False)
    a2r = {}
    for blk in sm.get_matching_blocks():
        for k in range(blk.size):
            a2r[blk.a + k] = blk.b + k
    out = {}
    for i in range(len(aw) - 1):
        j, j2 = a2r.get(i), a2r.get(i + 1)
        if j is not None and j2 == j + 1:
            out[i] = rend[j]
    return out


def breaks_of(lines) -> set[int]:
    """Indices i (in the flat word list) after which a line ends."""
    out, n = set(), 0
    for ln in lines[:-1]:
        n += len(ln)
        out.add(n - 1)
    return out


def score(pred: set[int], gold: dict[int, bool]):
    tp = sum(1 for i, b in gold.items() if b and i in pred)
    fp = sum(1 for i, b in gold.items() if not b and i in pred)
    fn = sum(1 for i, b in gold.items() if b and i not in pred)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"p": p, "r": r, "f1": 2 * p * r / (p + r) if p + r else 0.0, "tp": tp, "fp": fp, "fn": fn}


def load(name):
    from audio2midi.compose import load_words
    vocals = next((OUT / "stems" / name / "roformer_inst").glob("*(vocals)*.wav"))
    asr = OUT / "output" / name / "letra" / f"{name}_whisper.json"
    words, _, data = load_words(asr, vocals)
    return words, data, vocals


def fit_logistic(X, y, l2=1.0, iters=3000, lr=0.5):
    """Plain logistic regression (gradient descent, L2 on all but the bias)."""
    import numpy as np
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ w))
        g = X.T @ (p - y) / len(y) + l2 * np.r_[0, w[1:]] / len(y)
        w -= lr * g
    return w


def main():
    import itertools

    import numpy as np

    from audio2midi.compose import LINE_FEATURES, LINE_PARAMS, LINE_WEIGHTS, gap_features, group_lines
    verbose, train = "-v" in sys.argv, "--train" in sys.argv
    data = []
    for name, ref in SONGS:
        words, _, _ = load(name)
        data.append((name, words, labels(words, HERE / ref)))

    def run(weights_for, **params):
        out = []
        for k, (name, words, gold) in enumerate(data):
            out.append(score(breaks_of(group_lines(words, weights_for(k), **params)), gold))
        return out

    if train:
        # break model: leave one song out (weights learned on the other two), then all songs for the app
        XY = []
        for name, words, gold in data:
            X = gap_features(words)
            idx = sorted(gold)
            XY.append((X[idx], np.array([gold[i] for i in idx], dtype=float)))
        def fit(excl=None):
            X = np.vstack([x for k, (x, _) in enumerate(XY) if k != excl])
            y = np.concatenate([y for k, (_, y) in enumerate(XY) if k != excl])
            return fit_logistic(X, y)
        loso = [fit(k) for k in range(len(data))]
        full = fit()
        print("pesos (todas):", ", ".join(f"{n}={v:.2f}" for n, v in zip(LINE_FEATURES, full)))
        best = None
        for lw, hc in itertools.product([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0], [0.0, 0.25, 0.5, 1.0, 2.0]):
            res = run(lambda k: loso[k], length_weight=lw, half_cost=hc)
            f1 = np.mean([r["f1"] for r in res])
            if best is None or f1 > best[0]:
                best = (f1, lw, hc, res)
            print(f"  length_weight={lw:<4} half_cost={hc:<4} F1 medio (sin ver la canción) {100*f1:4.1f}%"
                  f"  [{' '.join(f'{100*r[chr(102)+chr(49)]:.0f}' for r in res)}]")
        print(f"mejor: length_weight={best[1]} half_cost={best[2]} -> {100*best[0]:.1f}%")
        return

    tot = run(lambda k: LINE_WEIGHTS)
    for (name, words, gold), s in zip(data, tot):
        print(f"{name[:28]:28s} F1 {100*s['f1']:3.0f}%  precisión {100*s['p']:3.0f}%  cobertura {100*s['r']:3.0f}%"
              f"  (saltos de referencia {s['tp']+s['fn']}, de más {s['fp']}, faltan {s['fn']})")
        if verbose:
            lines = group_lines(words)
            n = 0
            for ln in lines:
                marks = [w["w"] + (" ⟂" if gold.get(n + k) and k < len(ln) - 1 else "") for k, w in enumerate(ln)]
                last = n + len(ln) - 1
                bad = "   ✗ no va salto" if gold.get(last) is False else ""
                print(f"   {ln[0]['start']:6.1f}  {' '.join(marks)}{bad}")
                n += len(ln)
    print(f"{'media':28s} F1 {100*np.mean([s['f1'] for s in tot]):3.0f}%")


if __name__ == "__main__":
    main()
