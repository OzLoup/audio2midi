#!/usr/bin/env python3
"""Compara los acordes detectados (lista de (ini, fin, acorde) en JSON) con una referencia en texto.
Alinea las dos secuencias (Needleman-Wunsch) porque la referencia no trae tiempos.
Uso: compare.py detectado.json referencia.txt"""
import json, re, sys
PC = {"C":0,"C#":1,"Db":1,"D":2,"D#":3,"Eb":3,"E":4,"F":5,"F#":6,"Gb":6,"G":7,"G#":8,"Ab":8,"A":9,"A#":10,"Bb":10,"B":11}
def parse(ch):
    main, _, bass = ch.partition("/")
    m = re.match(r"([A-G][#b]?)(.*)", main)
    root, rest = m.group(1), m.group(2)
    minor = rest.startswith("m") and not rest.startswith("maj")
    return PC[root], "m" if minor else "", rest, PC[bass] if bass else None
def show(ch):
    return ch
def score(a, b):
    ra, qa, *_ = parse(a); rb, qb, *_ = parse(b)
    return 2 if (ra, qa) == (rb, qb) else 0.5 if ra == rb else -1
def align(x, y, gap=-1):
    n, m = len(x), len(y)
    D = [[0]*(m+1) for _ in range(n+1)]
    for i in range(1, n+1): D[i][0] = i*gap
    for j in range(1, m+1): D[0][j] = j*gap
    for i in range(1, n+1):
        for j in range(1, m+1):
            D[i][j] = max(D[i-1][j-1]+score(x[i-1], y[j-1]), D[i-1][j]+gap, D[i][j-1]+gap)
    i, j, out = n, m, []
    while i or j:
        if i and j and D[i][j] == D[i-1][j-1]+score(x[i-1], y[j-1]): out.append((x[i-1], y[j-1])); i -= 1; j -= 1
        elif i and D[i][j] == D[i-1][j]+gap: out.append((x[i-1], None)); i -= 1
        else: out.append((None, y[j-1])); j -= 1
    return out[::-1]

def evaluate(segments, ref_path):
    """-> dict con aciertos (raiz+mayor/menor, solo raiz, con bajo) y el alineamiento."""
    det = [show(c) for _, _, c in segments]
    ref = [c for l in open(ref_path) if not l.startswith("#") for c in l.split()]
    pairs = align(det, ref)
    both = [(a, b) for a, b in pairs if a and b]
    ok = [(a, b) for a, b in both if parse(a)[:2] == parse(b)[:2]]
    # dividing by the longer sequence penalizes both missed chords and invented ones
    return {"exact": len(ok) / max(len(ref), len(det)), "root": sum(parse(a)[0] == parse(b)[0] for a, b in both) / max(len(ref), len(det)),
            "bass": sum(parse(a)[3] == parse(b)[3] for a, b in ok) / max(len(ref), len(det)),
            "n_det": len(det), "n_ref": len(ref), "pairs": pairs}


def main():
    r = evaluate(json.load(open(sys.argv[1])), sys.argv[2])
    print(f"detectados {r['n_det']} | referencia {r['n_ref']}")
    print(f"raiz + mayor/menor: {100*r['exact']:.0f}% | solo raiz: {100*r['root']:.0f}% | con el bajo (Cm/G): {100*r['bass']:.0f}%")
    print("\n nuestro      moises")
    for a, b in r["pairs"]:
        mark = "" if a and b and parse(a)[:2] == parse(b)[:2] else ("  <- distinto" if a and b else "  <- hueco")
        print(f" {a or '-':10s}  {b or '-':10s}{mark}")


if __name__ == "__main__":
    main()
