"""
Range-Zonen-Screener
--------------------
Scannt Forex, Indizes, Metalle, Energie und Agrar auf 4h und 1d nach bestätigten
Seitwärts-Ranges (oben und unten je mind. 2 Antests einer Zone) und erzeugt eine
handyfreundliche Webseite in ./site (Tabelle + Charts je Markt: 1d und 4h).

Läuft automatisch über GitHub Actions (siehe .github/workflows/screener.yml).
Keine Anlageberatung – das Skript handelt nicht.
"""
import html
import os
import re
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

# ----------------------------------------------------------------------------
# Einstellungen
# ----------------------------------------------------------------------------
P = dict(
    MIN_KERZEN=10,          # Mindestlänge der Range (Kerzen)
    MAX_KERZEN=60,          # Maximale Länge der Range (Kerzen)
    MIN_ANTESTS=2,          # Mindest-Antests je Zone
    MIN_ABSTAND_KERZEN=2,   # Kerzen außerhalb der Zone zwischen zwei Antests
    ZONE_ATR=0.5,           # Zonentiefe (x ATR)
    ZONE_MAX_PROZENT=30,    # Zonentiefe höchstens (% der Range-Höhe)
    MIN_HOEHE_ATR=1.0,      # Range-Höhe mindestens (x ATR)
    MAX_HOEHE_ATR=8.0,      # Range-Höhe höchstens (x ATR)
    AUSBRUCH_KERZEN=3,      # Ausbrüche der letzten X Kerzen noch erkennen
    ATR_PERIODE=14,         # ATR-Periode
    SL_PUFFER_ATR=0.2,      # Stop-Puffer hinter der Zone (x ATR)
)
CHART_KERZEN = 100
AUSBRUECHE_ZEIGEN = True
AUSGABE = "site"

MAERKTE = {
    "Forex": {"EURUSD=X": "EUR/USD", "GBPUSD=X": "GBP/USD", "USDJPY=X": "USD/JPY", "USDCHF=X": "USD/CHF",
              "AUDUSD=X": "AUD/USD", "USDCAD=X": "USD/CAD", "NZDUSD=X": "NZD/USD", "EURGBP=X": "EUR/GBP",
              "EURJPY=X": "EUR/JPY", "GBPJPY=X": "GBP/JPY", "EURCHF=X": "EUR/CHF", "AUDJPY=X": "AUD/JPY"},
    "Indizes": {"^GDAXI": "DAX", "^GSPC": "S&P 500", "^NDX": "Nasdaq 100", "^DJI": "Dow Jones",
                "^STOXX50E": "Euro Stoxx 50", "^FTSE": "FTSE 100", "^FCHI": "CAC 40",
                "^N225": "Nikkei 225", "^ATX": "ATX"},
    "Metalle": {"GC=F": "Gold", "SI=F": "Silber", "PL=F": "Platin", "PA=F": "Palladium", "HG=F": "Kupfer"},
    "Energie": {"CL=F": "WTI Öl", "BZ=F": "Brent Öl", "NG=F": "Erdgas"},
    "Agrar": {"ZW=F": "Weizen", "ZC=F": "Mais", "ZS=F": "Sojabohnen", "KC=F": "Kaffee",
              "SB=F": "Zucker", "CC=F": "Kakao"},
}
TICKER = {t: name for gruppe in MAERKTE.values() for t, name in gruppe.items()}

# ----------------------------------------------------------------------------
# Zonen-Logik
# ----------------------------------------------------------------------------
def calc_atr(H, L, C, period):
    n = len(C)
    tr = [max(H[n-1-s], C[n-2-s]) - min(L[n-1-s], C[n-2-s]) for s in range(1, period + 1)]
    return float(np.mean(tr))


def count_touches(H, L, from_s, to_s, edge, upper, min_gap):
    """Antest = Kurs läuft in die Zone, nachdem er mind. min_gap Kerzen draußen war."""
    n = len(H)
    touches, outside = 0, min_gap
    for s in range(from_s, to_s - 1, -1):
        i = n - 1 - s
        inside = H[i] >= edge if upper else L[i] <= edge
        if inside:
            if outside >= min_gap:
                touches += 1
            outside = 0
        else:
            outside += 1
    return touches


def find_range(H, L, e, atr):
    n = len(H)
    max_w = min(P["MAX_KERZEN"], n - e - 1)
    best, res = -1, None
    for W in range(P["MIN_KERZEN"], max_w + 1):
        from_s = e + W - 1
        hi = float(H[n-1-from_s: n-e].max())
        lo = float(L[n-1-from_s: n-e].min())
        h = hi - lo
        if h < P["MIN_HOEHE_ATR"] * atr or h > P["MAX_HOEHE_ATR"] * atr:
            continue
        depth = min(P["ZONE_ATR"] * atr, h * P["ZONE_MAX_PROZENT"] / 100)
        up_in, dn_in = hi - depth, lo + depth
        tu = count_touches(H, L, from_s, e, up_in, True, P["MIN_ABSTAND_KERZEN"])
        td = count_touches(H, L, from_s, e, dn_in, False, P["MIN_ABSTAND_KERZEN"])
        if tu < P["MIN_ANTESTS"] or td < P["MIN_ANTESTS"]:
            continue
        if tu + td > best:
            best = tu + td
            res = dict(kerzen=W, tests_oben=tu, tests_unten=td, hi=hi, lo=lo,
                       up_in=up_in, dn_in=dn_in, start_idx=n - 1 - from_s)
    return res


def scan_df(df):
    H, L, C = (df[c].to_numpy(float) for c in ("High", "Low", "Close"))
    if len(C) < P["MIN_KERZEN"] + P["ATR_PERIODE"] + 3:
        return None
    atr = calc_atr(H, L, C, P["ATR_PERIODE"])
    if not atr > 0:
        return None
    rng = None
    for e in range(1, P["AUSBRUCH_KERZEN"] + 1):
        rng = find_range(H, L, e, atr)
        if rng:
            break
    if not rng:
        return None
    p, buf = float(C[-1]), P["SL_PUFFER_ATR"] * atr
    out = dict(rng, kurs=p, atr=atr, sl=np.nan, tp=np.nan, crv=np.nan, abstand_atr=0.0,
               start=df.index[rng["start_idx"]])
    if p > rng["hi"]:
        out.update(status="Ausbruch oben", abstand_atr=(p - rng["hi"]) / atr)
    elif p < rng["lo"]:
        out.update(status="Ausbruch unten", abstand_atr=(rng["lo"] - p) / atr)
    elif p >= rng["up_in"]:
        sl, tp = rng["hi"] + buf, rng["dn_in"]
        out.update(status="Obere Zone → Short", sl=sl, tp=tp,
                   crv=(p - tp) / (sl - p) if sl > p else np.nan)
    elif p <= rng["dn_in"]:
        sl, tp = rng["lo"] - buf, rng["up_in"]
        out.update(status="Untere Zone → Long", sl=sl, tp=tp,
                   crv=(tp - p) / (p - sl) if p > sl else np.nan)
    else:
        out.update(status="Mitte", abstand_atr=min(rng["up_in"] - p, p - rng["dn_in"]) / atr)
    return out

# ----------------------------------------------------------------------------
# Kursdaten (Yahoo Finance)
# ----------------------------------------------------------------------------
OHLC = {"Open": "first", "High": "max", "Low": "min", "Close": "last"}


def history(ticker, **kw):
    for versuch in range(3):
        try:
            df = yf.Ticker(ticker).history(auto_adjust=False, **kw)
            if not df.empty:
                return df
        except Exception as ex:
            print(f"  {ticker}: {ex}")
        time.sleep(3 * (versuch + 1))
    return pd.DataFrame()


def lade_kurse(ticker):
    fx = ticker.endswith("=X")
    out = {}
    h1 = history(ticker, period="730d" if fx else "180d", interval="1h")
    if not h1.empty:
        h1 = h1[["Open", "High", "Low", "Close"]].dropna()
        h1 = h1[(h1["High"] > 0) & (h1["Low"] > 0)]
        out["4h"] = h1.iloc[-180*24:].resample("4h", origin="start_day").agg(OHLC).dropna()
    if fx and not h1.empty:
        # Forex: Yahoo-Tagesdaten haben unzuverlässige Eröffnungskurse ->
        # 1d-Kerzen aus Stundendaten bauen, Tageswechsel 22:00 UTC (wie MT5-Broker)
        u = h1.copy()
        u.index = u.index.tz_convert("UTC") + pd.Timedelta(hours=2)
        d1 = u.resample("1D").agg(OHLC).dropna()
        out["1d"] = d1[d1.index.dayofweek < 5].iloc[-260:]
    else:
        d1 = history(ticker, period="1y", interval="1d")
        if not d1.empty:
            out["1d"] = d1[["Open", "High", "Low", "Close"]].dropna()
    return out


def nk(p):
    return 5 if p < 10 else 3 if p < 300 else 2

# ----------------------------------------------------------------------------
# Charts
# ----------------------------------------------------------------------------
FARBE = {"Obere Zone → Short": "#e05a4f", "Untere Zone → Long": "#3aa66b",
         "Ausbruch oben": "#d4a017", "Ausbruch unten": "#d4a017", "Mitte": "#9a9a9a"}
CHANCE = ["Obere Zone → Short", "Untere Zone → Long"]


def chart_datei(ticker, tf):
    return f"charts/{re.sub(r'[^A-Za-z0-9]', '_', ticker)}_{tf}.png"


def zeichne(ticker, markt, tf, daten, ranges):
    df = daten.get((ticker, tf))
    if df is None or df.empty:
        return None
    df = df.iloc[-CHART_KERZEN:]
    o, h, l, c = (df[k].to_numpy(float) for k in ("Open", "High", "Low", "Close"))
    x = np.arange(len(df))
    n = len(df)
    eigen = ranges.get((ticker, tf))
    d = nk(c[-1])

    with plt.style.context("dark_background"):
        fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=120)
        fig.patch.set_facecolor("#111316")
        ax.set_facecolor("#111316")
        farbe = np.where(c >= o, "#26a69a", "#ef5350")
        ax.vlines(x, l, h, color=farbe, linewidth=0.9)
        ax.bar(x, np.maximum(np.abs(c - o), (h.max() - l.min()) * 0.002),
               bottom=np.minimum(o, c), color=farbe, width=0.65)

        if eigen is not None:
            s = max(int(df.index.searchsorted(eigen["start"])), 0)
            ax.fill_between([s - 0.5, n + 1], eigen["up_in"], eigen["hi"], color="#e05a4f", alpha=0.3, linewidth=0)
            ax.fill_between([s - 0.5, n + 1], eigen["lo"], eigen["dn_in"], color="#3aa66b", alpha=0.3, linewidth=0)
            if not pd.isna(eigen["sl"]):
                for y, txt, col in ((eigen["sl"], "SL", "#e05a4f"), (eigen["tp"], "TP", "#3aa66b")):
                    ax.axhline(y, ls="--", lw=1, color=col)
                    ax.text(n + 1.5, y, f"{txt} {y:.{d}f}", color=col, va="center", fontsize=10)
        ax.axhline(c[-1], ls=":", lw=0.8, color="#bbbbbb")

        ticks = np.linspace(0, n - 1, 5).astype(int)
        fmt = "%d.%m." if tf == "1d" else "%d.%m. %H:%M"
        ax.set_xticks(ticks)
        ax.set_xticklabels([df.index[i].strftime(fmt) for i in ticks], fontsize=10)
        ax.set_xlim(-1, n + 14)
        ax.tick_params(axis="y", labelsize=10)
        ax.grid(color="#262a30", linewidth=0.5)
        for sp in ax.spines.values():
            sp.set_color("#333840")

        if eigen is not None:
            titel, col = f"{markt} · {tf} · {eigen['status']}", FARBE.get(eigen["status"], "white")
        else:
            titel, col = f"{markt} · {tf} · keine Range", "#9a9a9a"
        ax.set_title(titel, fontsize=13, loc="left", color=col, fontweight="bold")
        fig.tight_layout()
        pfad = chart_datei(ticker, tf)
        fig.savefig(os.path.join(AUSGABE, pfad), facecolor=fig.get_facecolor())
        plt.close(fig)
    return pfad

# ----------------------------------------------------------------------------
# Webseite
# ----------------------------------------------------------------------------
CSS = """
:root{--bg:#0b0d10;--card:#16191e;--line:#262a30;--text:#e8eaed;--mute:#9aa0a6;
--short:#e05a4f;--long:#3aa66b;--brk:#d4a017}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);
font:15px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
main{max-width:1100px;margin:0 auto;padding:16px}
h1{font-size:22px;margin:4px 0}
.stand{color:var(--mute);margin-bottom:12px}
.summary{display:flex;gap:8px;flex-wrap:wrap;margin:12px 0 4px}
.summary a{flex:1 1 30%;min-width:96px;background:var(--card);border:1px solid var(--line);
border-radius:12px;padding:10px;text-decoration:none;color:var(--text);text-align:center}
.summary b{display:block;font-size:22px}
h2{font-size:19px;margin:28px 0 10px;scroll-margin-top:12px}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:12px;margin:0 0 14px}
.head{display:flex;align-items:baseline;gap:8px;margin-bottom:6px}
.nr{color:var(--mute);font-weight:600}
.head h3{margin:0;font-size:18px}
.tf{border-top:1px solid var(--line);padding:8px 0}
.tf:first-of-type{border-top:0}
.badge{display:inline-block;font-weight:700;margin-left:6px}
.tfl{display:inline-block;min-width:28px;font-weight:700;color:var(--mute)}
.kv{display:grid;grid-template-columns:auto 1fr;gap:2px 10px;margin-top:4px;font-size:14px}
.kv span:nth-child(odd){color:var(--mute)}
.charts{display:grid;grid-template-columns:1fr;gap:8px;margin-top:8px}
.charts img{width:100%;height:auto;border-radius:8px;border:1px solid var(--line);display:block}
@media(min-width:820px){.charts{grid-template-columns:1fr 1fr}}
.watch{background:var(--card);border:1px solid var(--line);border-radius:14px;overflow:hidden}
.watch div{padding:9px 12px;border-top:1px solid var(--line);font-size:14px}
.watch div:first-child{border-top:0}
.watch b{font-size:15px}
.mute{color:var(--mute)}
footer{color:var(--mute);font-size:12px;margin:30px 0 10px}
"""


def tf_block(z):
    d = nk(z["kurs"])
    f = lambda v: "–" if pd.isna(v) else f"{v:.{d}f}"
    col = FARBE.get(z["status"], "#9a9a9a")
    zeilen = [("Kurs", f(z["kurs"])),
              ("Obere Zone", f"{f(z['up_in'])} – {f(z['hi'])}"),
              ("Untere Zone", f"{f(z['lo'])} – {f(z['dn_in'])}"),
              ("Antests o/u", f"{z['tests_oben']} / {z['tests_unten']}")]
    if not pd.isna(z["sl"]):
        zeilen += [("Stop / Ziel", f"{f(z['sl'])} / {f(z['tp'])}"),
                   ("CRV", f"{z['crv']:.1f}" if not pd.isna(z["crv"]) else "–")]
    kv = "".join(f"<span>{k}</span><span>{v}</span>" for k, v in zeilen)
    return (f"<div class='tf'><span class='tfl'>{z['TF']}</span>"
            f"<span class='badge' style='color:{col}'>{html.escape(z['status'])}</span>"
            f"<div class='kv'>{kv}</div></div>")


def karten(ticker_liste, ranges, charts):
    out = []
    for nr, t in enumerate(ticker_liste, 1):
        markt = html.escape(TICKER[t])
        bloecke = "".join(tf_block(ranges[(t, tf)]) for tf in ("1d", "4h") if (t, tf) in ranges)
        bilder = "".join(f"<img loading='lazy' src='{charts[(t, tf)]}' alt='{markt} {tf}'>"
                         for tf in ("1d", "4h") if charts.get((t, tf)))
        out.append(f"<div class='card'><div class='head'><span class='nr'>#{nr}</span><h3>{markt}</h3></div>"
                   f"{bloecke}<div class='charts'>{bilder}</div></div>")
    return "".join(out) or "<p class='mute'>Keine.</p>"


def beobachtung(ticker_liste, ranges):
    if not ticker_liste:
        return "<p class='mute'>Keine.</p>"
    rows = []
    for t in ticker_liste:
        teile = []
        for tf in ("1d", "4h"):
            z = ranges.get((t, tf))
            if z is None:
                continue
            d = nk(z["kurs"])
            teile.append(f"<span class='tfl'>{tf}</span> {z['lo']:.{d}f} – {z['hi']:.{d}f} "
                         f"<span class='mute'>({html.escape(z['status'])})</span>")
        rows.append(f"<div><b>{html.escape(TICKER[t])}</b><br>{'<br>'.join(teile)}</div>")
    return f"<div class='watch'>{''.join(rows)}</div>"


def seite(stand, chancen, ausbr, mitte, ranges, charts, fehler):
    abschnitt_ausbr = (f"<h2 id='ausbr'>🚀 Ausbrüche ({len(ausbr)})</h2>{karten(ausbr, ranges, charts)}"
                       if AUSBRUECHE_ZEIGEN else "")
    fehltext = f"<p class='mute'>Keine Daten für: {html.escape(', '.join(fehler))}</p>" if fehler else ""
    return f"""<!doctype html>
<html lang="de"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="apple-mobile-web-app-capable" content="yes">
<title>Range-Screener</title><style>{CSS}</style></head>
<body><main>
<h1>📊 Range-Screener</h1>
<div class="stand">Stand {stand} · {len(TICKER)} Märkte · 1d + 4h</div>
<div class="summary">
<a href="#chancen"><b style="color:var(--long)">{len(chancen)}</b>Chancen</a>
<a href="#ausbr"><b style="color:var(--brk)">{len(ausbr)}</b>Ausbrüche</a>
<a href="#watch"><b>{len(mitte)}</b>Beobachtung</a>
</div>
<h2 id="chancen">🎯 Tradingchancen ({len(chancen)})</h2>
{karten(chancen, ranges, charts)}
{abschnitt_ausbr}
<h2 id="watch">👀 Beobachtungsliste ({len(mitte)})</h2>
<p class="mute">Bestätigte Ranges, Kurs aktuell in der Mitte.</p>
{beobachtung(mitte, ranges)}
{fehltext}
<footer>Kurse: Yahoo Finance (Gold, Öl usw. als Futures – können vom CFD abweichen).
Range bestätigt = jede Zone mind. {P['MIN_ANTESTS']}× angetestet. Keine Anlageberatung.</footer>
</main></body></html>"""

# ----------------------------------------------------------------------------
# Telegram (optional)
# ----------------------------------------------------------------------------
def telegram(chancen, ausbr, ranges, stand):
    token, chat = os.environ.get("TELEGRAM_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")
    if not (token and chat):
        return
    import requests
    zeilen = [f"📊 Range-Screener · {stand}", ""]
    for t in chancen:
        for tf in ("1d", "4h"):
            z = ranges.get((t, tf))
            if z is not None and z["status"] in CHANCE:
                crv = f" · CRV {z['crv']:.1f}" if not pd.isna(z["crv"]) else ""
                zeilen.append(f"🎯 {TICKER[t]} {tf}: {z['status']}{crv}")
    if not chancen:
        zeilen.append("Heute keine Tradingchancen.")
    if ausbr:
        zeilen.append(f"🚀 Ausbrüche: {', '.join(TICKER[t] for t in ausbr)}")
    url = os.environ.get("PAGE_URL")
    if url:
        zeilen += ["", url]
    try:
        requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                      data={"chat_id": chat, "text": "\n".join(zeilen)}, timeout=20)
    except Exception as ex:
        print("Telegram-Fehler:", ex)

# ----------------------------------------------------------------------------
# Ablauf
# ----------------------------------------------------------------------------
def main():
    os.makedirs(os.path.join(AUSGABE, "charts"), exist_ok=True)
    daten, ranges, fehler = {}, {}, []
    for i, (t, name) in enumerate(TICKER.items(), 1):
        print(f"{i}/{len(TICKER)} {name}")
        k = lade_kurse(t)
        if not k:
            fehler.append(name)
            continue
        for tf, df in k.items():
            daten[(t, tf)] = df
            r = scan_df(df)
            if r:
                ranges[(t, tf)] = dict(r, Ticker=t, Markt=name, TF=tf)
        time.sleep(0.5)

    erg = pd.DataFrame(list(ranges.values()))

    def maerkte(maske, spalte, absteigend=False):
        if erg.empty:
            return []
        sub = erg[maske].sort_values(spalte, ascending=not absteigend)
        return list(dict.fromkeys(sub["Ticker"]))

    if erg.empty:
        chancen = ausbr = mitte = []
    else:
        chancen = maerkte(erg["status"].isin(CHANCE), "crv", True)
        ausbr = [t for t in maerkte(erg["status"].str.startswith("Ausbruch"), "Markt") if t not in chancen]
        mitte = [t for t in maerkte(erg["status"] == "Mitte", "abstand_atr") if t not in chancen and t not in ausbr]

    charts = {}
    for t in chancen + (ausbr if AUSBRUECHE_ZEIGEN else []):
        for tf in ("1d", "4h"):
            charts[(t, tf)] = zeichne(t, TICKER[t], tf, daten, ranges)

    stand = datetime.now(ZoneInfo("Europe/Vienna")).strftime("%d.%m.%Y %H:%M")
    with open(os.path.join(AUSGABE, "index.html"), "w", encoding="utf-8") as f:
        f.write(seite(stand, chancen, ausbr, mitte, ranges, charts, fehler))
    print(f"Fertig: {len(chancen)} Chancen, {len(ausbr)} Ausbrüche, {len(mitte)} Beobachtung, "
          f"{len(fehler)} ohne Daten")
    telegram(chancen, ausbr, ranges, stand)


if __name__ == "__main__":
    main()
