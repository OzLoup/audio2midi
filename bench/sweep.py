#!/usr/bin/env python3
"""Mide el detector de acordes contra las referencias de bench/songs.txt y prueba ajustes.
Las caracteristicas se extraen una vez por cancion (quedan en bench/cache); cada ajuste se decodifica en milisegundos.

Uso (Python de la app, con PYTHONPATH=src):
  python bench/sweep.py            mide los ajustes por defecto en cada cancion
  python bench/sweep.py --sweep    ademas: acierto medio por valor de cada ajuste y mejores combinaciones
"""
import itertools
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from compare import evaluate  # noqa: E402

from audio2midi.chords import DEFAULTS, decode, extract  # noqa: E402

songs = []
for line in (HERE / "songs.txt").read_text(encoding="utf-8").splitlines():
    if line.strip() and not line.startswith("#"):
        d, name, ref = (x.strip() for x in line.split("|"))
        d = Path(d).expanduser()
        stem = lambda part, d=d, name=name: d / f"{name}_({part})_htdemucs_ft.wav"  # noqa: E731
        f = extract([stem("Other"), stem("Bass")], stem("Bass"), stem("Drums"), HERE / "cache")
        songs.append((name, f, HERE / ref))


def score(**kw):
    out = []
    for _, f, ref in songs:
        r = decode(f, **kw)
        out.append({**evaluate(r.segments, ref), "key": r.key})
    return out


res = score()
for (name, f, _), e in zip(songs, res):
    print(f"{name[:28]:28s} {e['key']:9s} raiz+mayor/menor {100*e['exact']:3.0f}% | solo raiz {100*e['root']:3.0f}% | "
          f"detectados {e['n_det']} de {e['n_ref']}")
print(f"{'MEDIA':28s} {'':9s} raiz+mayor/menor {100*sum(e['exact'] for e in res)/len(res):3.0f}%")
if "--sweep" not in sys.argv:
    sys.exit()

grid = dict(change=[0.3, 0.5, 0.7], sharpness=[20.0, 30.0, 40.0, 60.0], bass_weight=[0.1, 0.15, 0.25],
            use_repeats=[True, False], sevenths=[False, True], slash=[True, False], key_prior=[0.0, 0.05, 0.1])
acc, rows = defaultdict(list), []
for combo in itertools.product(*grid.values()):
    kw = dict(zip(grid, combo))
    xs = [e["exact"] for e in score(**kw)]
    rows.append((sum(xs) / len(xs), xs, kw))
    for k, v in kw.items():
        acc[(k, v)].append(xs)
print("\nacierto medio por valor (media | por cancion):")
for (k, v), runs in sorted(acc.items(), key=lambda kv: (kv[0][0], str(kv[0][1]))):
    per = [sum(r[i] for r in runs) / len(runs) for i in range(len(songs))]
    mark = "  <- por defecto" if DEFAULTS.get(k) == v else ""
    print(f"  {k:12s} {str(v):6s} {100*sum(per)/len(per):5.1f}%  | " + " ".join(f"{100*x:5.1f}%" for x in per) + mark)
print("\nmejores (media):")
for m, xs, kw in sorted(rows, key=lambda r: -r[0])[:8]:
    print(f"  {100*m:4.0f}%  (" + " ".join(f"{100*x:.0f}%" for x in xs) + f")  {kw}")
