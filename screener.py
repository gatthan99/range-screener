"""
Markt-Übersicht
---------------
Lädt Forex, Indizes, Metalle, Energie und Agrar und erzeugt eine handyfreundliche
Webseite in ./site:
  * je Markt der 1d-Chart mit Volumenprofil und der 4h-Chart (ohne Zonen)

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
KERZEN_1D = 100          # sichtbare Tageskerzen (= Zeitraum des Volumenprofils)
KERZEN_4H = 120          # sichtbare 4h-Kerzen
KORR_TAGE = 60           # Zeitraum der Korrelation (Handelstage, tägliche Renditen)
KORR_SCHWELLE = 0.70     # ab diesem Wert (positiv oder negativ) gilt ein Paar als stark korreliert
VP_STUFEN = 48           # Preisstufen im Volumenprofil
VALUE_AREA = 0.70        # Anteil des Volumens in der Value Area
AUSGABE = "site"
BROKER_TZ = "Europe/Athens"  # MT5-Serverzeit (GMT+2 / Sommer GMT+3) = Wien + 1 h
VERSION = str(int(time.time()))  # hängt an Bildadressen, damit der Browser keine alten Charts zeigt

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

# Positionsrechner: Kurswährung und Kontraktgröße je Lot (typische MT5-CFD-Werte).
# Die Werte lassen sich auf der Seite anpassen (werden im Browser gespeichert).
# "USc" = Kurs in US-Cent (z. B. Agrar).
KONTRAKT = {
    "^GDAXI": ("EUR", 1), "^GSPC": ("USD", 1), "^NDX": ("USD", 1), "^DJI": ("USD", 1),
    "^STOXX50E": ("EUR", 1), "^FTSE": ("GBP", 1), "^FCHI": ("EUR", 1), "^N225": ("JPY", 1), "^ATX": ("EUR", 1),
    "GC=F": ("USD", 100), "SI=F": ("USD", 5000), "PL=F": ("USD", 100), "PA=F": ("USD", 100),
    "HG=F": ("USD", 25000),
    "CL=F": ("USD", 1000), "BZ=F": ("USD", 1000), "NG=F": ("USD", 10000),
    "ZW=F": ("USc", 5000), "ZC=F": ("USc", 5000), "ZS=F": ("USc", 5000), "KC=F": ("USc", 37500),
    "SB=F": ("USc", 112000), "CC=F": ("USD", 10),
}


def kontrakt(ticker):
    if ticker.endswith("=X"):
        return ticker[3:6], 100000
    return KONTRAKT.get(ticker, ("USD", 1))

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
        if "Volume" not in h1:
            h1["Volume"] = 0.0
        h1 = h1[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Open", "High", "Low", "Close"])
        h1["Volume"] = h1["Volume"].fillna(0)
        h1 = h1[(h1["High"] > 0) & (h1["Low"] > 0)]
        out["1h"] = h1
        # Kerzen wie beim MT5-Broker (Serverzeit = Wiener Zeit + 1 h, ganzjährig):
        # 4h-Wechsel um 23, 03, 07, 11, 15, 19 Uhr Wiener Zeit, Tageswechsel um 23 Uhr.
        # Index der Kerzen = Wiener Ortszeit (ohne Zeitzone), für die Chart-Beschriftung.
        srv = h1.copy()
        srv.index = srv.index.tz_convert(BROKER_TZ).tz_localize(None)
        h4 = srv.iloc[-180*24:].resample("4h", origin="epoch").agg(OHLC).dropna()
        h4.index = h4.index - pd.Timedelta(hours=1)
        out["4h"] = h4
    if fx and not h1.empty:
        # Forex: Yahoo-Tagesdaten haben unzuverlässige Eröffnungskurse -> aus Stundendaten bauen
        d1 = srv.resample("1D").agg(OHLC).dropna()
        out["1d"] = d1[d1.index.dayofweek < 5].iloc[-260:]
    else:
        d1 = history(ticker, period="1y", interval="1d")
        if not d1.empty:
            out["1d"] = d1[["Open", "High", "Low", "Close"]].dropna()
    return out


def nk(p):
    return 5 if p < 10 else 3 if p < 300 else 2

# ----------------------------------------------------------------------------
# Volumenprofil
# ----------------------------------------------------------------------------
def volumenprofil(h1, t0, lo, hi):
    """Volumenprofil aus Stundenkerzen ab t0 im Preisbereich lo..hi.
    Ohne Volumen (z. B. Forex bei Yahoo) wird ein Zeit-Profil berechnet."""
    if h1 is None or h1.empty or not hi > lo:
        return None
    t0 = pd.Timestamp(t0)
    if t0.tzinfo is None:
        t0 = t0.tz_localize("Europe/Vienna", nonexistent="shift_forward", ambiguous=False)
    sub = h1[h1.index >= t0]
    if len(sub) < 20:
        return None
    vol = sub["Volume"].to_numpy(float)
    echt = bool(np.nansum(vol) > 0 and (vol > 0).mean() > 0.5)
    gew = vol if echt else np.ones(len(sub))
    edges = np.linspace(lo, hi, VP_STUFEN + 1)
    prof = np.zeros(VP_STUFEN)
    skala = VP_STUFEN / (hi - lo)
    for bl, bh, w in zip(sub["Low"].to_numpy(float), sub["High"].to_numpy(float), gew):
        bl, bh = max(bl, lo), min(bh, hi)
        if not w > 0 or bh < bl:
            continue
        i0 = min(int((bl - lo) * skala), VP_STUFEN - 1)
        i1 = min(int((bh - lo) * skala), VP_STUFEN - 1)
        prof[i0:i1 + 1] += w / (i1 - i0 + 1)
    if prof.sum() <= 0:
        return None
    mitte = (edges[:-1] + edges[1:]) / 2
    poc_i = int(prof.argmax())
    # Value Area: vom POC aus jeweils zur volumenstärkeren Seite erweitern, bis 70 % erreicht sind
    a = b = poc_i
    summe, ziel = prof[poc_i], VALUE_AREA * prof.sum()
    while summe < ziel and (a > 0 or b < VP_STUFEN - 1):
        unten = prof[a - 1] if a > 0 else -1
        oben = prof[b + 1] if b < VP_STUFEN - 1 else -1
        if oben >= unten:
            b += 1; summe += prof[b]
        else:
            a -= 1; summe += prof[a]
    return dict(edges=edges, mitte=mitte, prof=prof, poc=float(mitte[poc_i]), poc_i=poc_i,
                va=set(range(a, b + 1)), vah=float(edges[b + 1]), val=float(edges[a]), echt=echt)

# ----------------------------------------------------------------------------
# Charts
# ----------------------------------------------------------------------------
HG = "#111316"


def chart_datei(ticker, tf):
    return f"charts/{re.sub(r'[^A-Za-z0-9]', '_', ticker)}_{tf}.png"


def kerzen(ax, df):
    o, h, l, c = (df[k].to_numpy(float) for k in ("Open", "High", "Low", "Close"))
    x = np.arange(len(df))
    farbe = np.where(c >= o, "#26a69a", "#ef5350")
    ax.vlines(x, l, h, color=farbe, linewidth=0.9)
    ax.bar(x, np.maximum(np.abs(c - o), (h.max() - l.min()) * 0.002),
           bottom=np.minimum(o, c), color=farbe, width=0.65)
    ax.axhline(c[-1], ls=":", lw=0.8, color="#bbbbbb")


def achsen(ax, df, fmt, rechts=4):
    n = len(df)
    ticks = np.linspace(0, n - 1, 5).astype(int)
    ax.set_xticks(ticks)
    ax.set_xticklabels([df.index[i].strftime(fmt) for i in ticks], fontsize=10)
    ax.set_xlim(-1, n + rechts)
    ax.tick_params(axis="y", labelsize=10)
    ax.grid(color="#262a30", linewidth=0.5)
    ax.set_facecolor(HG)
    for sp in ax.spines.values():
        sp.set_color("#333840")


def speichern(fig, ticker, tf):
    fig.tight_layout(w_pad=0.3)
    pfad = chart_datei(ticker, tf)
    fig.savefig(os.path.join(AUSGABE, pfad), facecolor=HG)
    plt.close(fig)
    return pfad


def chart_1d(ticker, markt, df, vp):
    with plt.style.context("dark_background"):
        if vp:
            fig, (ax, axp) = plt.subplots(1, 2, figsize=(7.2, 4.2), dpi=120, sharey=True,
                                          gridspec_kw={"width_ratios": [5, 1.1]})
        else:
            fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=120)
        fig.patch.set_facecolor(HG)
        kerzen(ax, df)
        if vp:
            farben = ["#e3a33b" if i == vp["poc_i"] else "#4f75a3" if i in vp["va"] else "#343b46"
                      for i in range(len(vp["prof"]))]
            axp.set_facecolor(HG)
            axp.barh(vp["mitte"], vp["prof"], height=(vp["edges"][1] - vp["edges"][0]) * 0.92,
                     color=farben, linewidth=0)
            axp.set_xticks([])
            axp.tick_params(left=False)
            axp.set_title("Volumen" if vp["echt"] else "Zeit-Profil", fontsize=10, color="#9aa0a6", loc="left")
            for sp in axp.spines.values():
                sp.set_color("#333840")
            ax.axhline(vp["poc"], ls="-.", lw=1, color="#e3a33b", alpha=0.9)
            ax.text(0, vp["poc"], " POC", color="#e3a33b", fontsize=9, va="bottom")
        achsen(ax, df, "%d.%m.")
        ax.set_title(f"{markt} · 1d", fontsize=13, loc="left", fontweight="bold")
        return speichern(fig, ticker, "1d")


def chart_4h(ticker, markt, df):
    with plt.style.context("dark_background"):
        fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=120)
        fig.patch.set_facecolor(HG)
        kerzen(ax, df)
        achsen(ax, df, "%d.%m. %H:%M")
        ax.set_title(f"{markt} · 4h", fontsize=13, loc="left", fontweight="bold")
        return speichern(fig, ticker, "4h")

# ----------------------------------------------------------------------------
# Korrelation
# ----------------------------------------------------------------------------
def korrelation(schluss):
    """schluss: {Marktname: Serie der Tagesschlusskurse}. Liefert Korrelationsmatrix der
    täglichen Renditen über die letzten KORR_TAGE gemeinsamen Handelstage."""
    reihen = {}
    for name, ser in schluss.items():
        idx = pd.DatetimeIndex(ser.index)
        if idx.tz is not None:
            idx = idx.tz_localize(None)
        r = pd.Series(ser.to_numpy(float), index=idx.normalize())
        reihen[name] = r[~r.index.duplicated(keep="last")]
    df = pd.DataFrame(reihen).sort_index()
    df = df[df.index.dayofweek < 5].ffill(limit=2)
    rend = df.pct_change(fill_method=None).iloc[-KORR_TAGE:]
    return rend.corr(min_periods=int(KORR_TAGE * 0.6))


def chart_korrelation(k):
    n = len(k)
    with plt.style.context("dark_background"):
        groesse = max(6.0, 0.24 * n + 1.8)
        fig, ax = plt.subplots(figsize=(groesse, groesse), dpi=130)
        fig.patch.set_facecolor(HG)
        ax.set_facecolor(HG)
        werte = k.to_numpy(float)
        bild = ax.imshow(np.ma.masked_invalid(werte), cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(n)); ax.set_yticks(range(n))
        ax.set_xticklabels(k.columns, rotation=90, fontsize=7)
        ax.set_yticklabels(k.index, fontsize=7)
        for i in range(n):
            for j in range(n):
                v = werte[i, j]
                if i != j and np.isfinite(v):
                    ax.text(j, i, f"{v*100:.0f}", ha="center", va="center", fontsize=4.6,
                            color="white" if abs(v) > 0.55 else "#1b1f24")
        # Trennlinien zwischen den Gruppen
        pos = 0
        for g in MAERKTE:
            anz = sum(1 for name in k.columns if name in GRUPPE_VON and GRUPPE_VON[name] == g)
            pos += anz
            if 0 < pos < n:
                ax.axhline(pos - 0.5, color="#0b0d10", lw=1.5); ax.axvline(pos - 0.5, color="#0b0d10", lw=1.5)
        for sp in ax.spines.values():
            sp.set_color("#333840")
        cb = fig.colorbar(bild, ax=ax, fraction=0.035, pad=0.02)
        cb.ax.tick_params(labelsize=7)
        ax.set_title(f"Korrelation tägl. Renditen · letzte {KORR_TAGE} Handelstage (Werte in %)",
                     fontsize=9, loc="left", color="#c8ccd2")
        fig.tight_layout()
        pfad = "charts/korrelation.png"
        fig.savefig(os.path.join(AUSGABE, pfad), facecolor=HG)
        plt.close(fig)
    return pfad


GRUPPE_VON = {name: g for g, m in MAERKTE.items() for name in m.values()}

# ----------------------------------------------------------------------------
# Kennzahlen für die Übersicht
# ----------------------------------------------------------------------------
def kennzahlen(d1, vp):
    c = d1["Close"].to_numpy(float)
    p = c[-1]
    tag = (p / c[-2] - 1) * 100 if len(c) > 1 else np.nan
    woche = (p / c[-6] - 1) * 100 if len(c) > 5 else np.nan
    lage, lage_kl = "–", ""
    if vp:
        if p > vp["vah"]:
            lage, lage_kl = "über VA", "up"
        elif p < vp["val"]:
            lage, lage_kl = "unter VA", "dn"
        else:
            lage, lage_kl = "in VA", ""
    return dict(kurs=p, tag=tag, woche=woche, lage=lage, lage_kl=lage_kl)

# ----------------------------------------------------------------------------
# Webseite
# ----------------------------------------------------------------------------
CSS = """
:root{--bg:#0b0d10;--card:#16191e;--line:#262a30;--text:#e8eaed;--mute:#9aa0a6;
--up:#3aa66b;--dn:#e05a4f;--poc:#e3a33b}
*{box-sizing:border-box}
html{scroll-behavior:smooth}
body{margin:0;background:var(--bg);color:var(--text);
font:15px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
main{max-width:1100px;margin:0 auto;padding:16px}
h1{font-size:22px;margin:4px 0}
.stand{color:var(--mute);margin-bottom:10px}
nav{position:sticky;top:0;z-index:5;background:var(--bg);padding:8px 0;display:flex;gap:6px;
overflow-x:auto;border-bottom:1px solid var(--line);margin-bottom:8px}
nav a{flex:0 0 auto;padding:6px 12px;border-radius:999px;background:var(--card);border:1px solid var(--line);
color:var(--text);text-decoration:none;font-size:14px}
h2{font-size:18px;margin:22px 0 8px;scroll-margin-top:56px}
table{width:100%;border-collapse:collapse;background:var(--card);border:1px solid var(--line);
border-radius:12px;overflow:hidden;font-size:14px}
th,td{padding:8px 10px;text-align:right;white-space:nowrap}
th{color:var(--mute);font-weight:600;font-size:12px;border-bottom:1px solid var(--line)}
td:first-child,th:first-child{text-align:left}
tr+tr td{border-top:1px solid var(--line)}
td a{color:var(--text);text-decoration:none;font-weight:600}
.up{color:var(--up)}.dn{color:var(--dn)}.mute{color:var(--mute)}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:12px;margin:0 0 14px;
scroll-margin-top:56px}
.head{display:flex;align-items:baseline;justify-content:space-between;gap:8px;flex-wrap:wrap}
.head h3{margin:0;font-size:18px}
.info{color:var(--mute);font-size:14px}
.info b{color:var(--text);font-weight:600}
.charts{display:grid;grid-template-columns:1fr;gap:8px;margin-top:10px}
.charts img{width:100%;height:auto;border-radius:8px;border:1px solid var(--line);display:block}
@media(min-width:820px){.charts{grid-template-columns:1fr 1fr}}
.top{display:inline-block;margin-top:6px;font-size:13px;color:var(--mute);text-decoration:none}
@media(max-width:480px){.hide-s{display:none}}
footer{color:var(--mute);font-size:12px;margin:30px 0 10px}
#rechner h2{margin:0 0 10px}
.form{display:grid;grid-template-columns:1fr 1fr;gap:10px}
.form label{display:flex;flex-direction:column;gap:4px;font-size:13px;color:var(--mute)}
.form .voll{grid-column:1/-1}
.form input,.form select{font:inherit;font-size:16px;color:var(--text);background:#0f1215;
border:1px solid var(--line);border-radius:10px;padding:9px 10px;width:100%}
.ergebnis{margin-top:12px;border-top:1px solid var(--line);padding-top:10px}
.gross{font-size:30px;font-weight:700;line-height:1.1}
.gross small{font-size:15px;color:var(--mute);font-weight:400}
.kv{display:grid;grid-template-columns:auto 1fr;gap:3px 12px;margin-top:8px;font-size:14px}
.kv span:nth-child(odd){color:var(--mute)}
.warn{color:var(--poc);font-size:13px;margin-top:8px}
details{margin-top:10px;font-size:14px}
summary{color:var(--mute);cursor:pointer}
.calc-link{float:right;font-size:13px;color:var(--poc);text-decoration:none;margin-top:6px}
.korr img{width:100%;height:auto;border-radius:8px;border:1px solid var(--line);display:block;margin-top:8px}
.paare{display:grid;grid-template-columns:1fr;gap:0;margin-top:6px}
.paare div{display:flex;justify-content:space-between;gap:10px;padding:7px 0;border-top:1px solid var(--line);font-size:14px}
.paare div:first-child{border-top:0}
.paare b{font-variant-numeric:tabular-nums}
.kpos{color:#e0705f}.kneg{color:#5b9bd5}
@media(min-width:820px){.paare{grid-template-columns:1fr 1fr;column-gap:24px}}
"""


def anker(ticker):
    return "m-" + re.sub(r"[^A-Za-z0-9]", "", ticker)


def pct(v):
    if pd.isna(v):
        return "<span class='mute'>–</span>"
    kl = "up" if v > 0 else "dn" if v < 0 else "mute"
    return f"<span class='{kl}'>{v:+.2f}%</span>"


def karte(z):
    d = nk(z["kurs"])
    vp = z["vp"]
    info = f"Kurs <b>{z['kurs']:.{d}f}</b> · Tag {pct(z['tag'])}"
    if vp:
        art = "POC" if vp["echt"] else "POC (Zeit)"
        info += (f"<br>{art} <b>{vp['poc']:.{d}f}</b> · VA <b>{vp['val']:.{d}f} – {vp['vah']:.{d}f}</b>"
                 f" · <span class='{z['lage_kl']}'>{z['lage']}</span>")
    bilder = "".join(f"<img loading='lazy' src='{p}?v={VERSION}' alt='{html.escape(z['markt'])} {tf}'>"
                     for tf, p in (("1d", z["c1d"]), ("4h", z["c4h"])) if p)
    return (f"<div class='card' id='{anker(z['ticker'])}'><div class='head'><h3>{html.escape(z['markt'])}</h3>"
            f"</div><div class='info'>{info}</div><div class='charts'>{bilder}</div>"
            f"<a class='top' href='#top'>↑ nach oben</a>"
            f"<a class='calc-link' href='#rechner' onclick=\"waehle('{z['ticker']}')\">🧮 Position berechnen</a></div>")


RECHNER_HTML = """
<div class="card" id="rechner">
<h2>🧮 Positionsrechner</h2>
<div class="form">
  <label>Kontostand<input id="konto" inputmode="decimal" placeholder="z. B. 10000"></label>
  <label>Kontowährung<select id="kwaehr"><option>EUR</option><option>USD</option><option>CHF</option><option>GBP</option></select></label>
  <label>Risiko in %<input id="risiko" inputmode="decimal" value="1"></label>
  <label>Markt<select id="markt"></select></label>
  <label>Einstieg<input id="einstieg" inputmode="decimal"></label>
  <label>Stop-Loss<input id="stop" inputmode="decimal"></label>
  <label class="voll">Ziel (optional, für CRV)<input id="ziel" inputmode="decimal"></label>
</div>
<details><summary>Kontraktdaten dieses Markts (bitte einmal mit MT5 abgleichen)</summary>
<div class="form" style="margin-top:8px">
  <label>Kontraktgröße je Lot<input id="kgroesse" inputmode="decimal"></label>
  <label>Kurswährung<select id="qwaehr"><option>USD</option><option>EUR</option><option>GBP</option>
  <option>JPY</option><option>CHF</option><option>CAD</option><option>AUD</option><option>NZD</option>
  <option value="USc">USc (US-Cent)</option></select></label>
  <label>Lot-Schritt<input id="lotstep" inputmode="decimal" value="0.01"></label>
  <label>Mindest-Lot<input id="lotmin" inputmode="decimal" value="0.01"></label>
</div>
<p class="mute" style="font-size:13px">MT5 → Kurse → Symbol antippen → <i>Details/Spezifikation</i> → „Kontraktgröße“.
Änderungen werden auf diesem Gerät gespeichert.</p>
</details>
<div class="ergebnis" id="ergebnis"></div>
</div>
"""

RECHNER_JS = """
<script>
const M = __DATEN__;
const FX = __FX__;
const $ = id => document.getElementById(id);
const zahl = v => { const n = parseFloat(String(v).replace(/\\s/g,'').replace(',', '.')); return isFinite(n) ? n : NaN; };
const ls = { get(k){ try { return localStorage.getItem(k); } catch(e) { return null; } },
             set(k,v){ try { localStorage.setItem(k, v); } catch(e) {} } };
const fmt = (v, d=2) => isFinite(v) ? v.toLocaleString('de-AT', {minimumFractionDigits:d, maximumFractionDigits:d}) : '–';

// USD-Wert einer Währungseinheit
function usdJe(w){
  if (w === 'USD') return 1;
  if (w === 'USc') return 0.01;
  const r = FX[w]; return r ? r : NaN;
}

function specLaden(m){
  let s = null; try { s = JSON.parse(ls.get('spec-' + m.t) || 'null'); } catch(e) {}
  $('kgroesse').value = s ? s.k : m.k;
  $('qwaehr').value = s ? s.w : m.w;
  $('lotstep').value = s && s.step ? s.step : '0.01';
  $('lotmin').value = s && s.min ? s.min : '0.01';
}
function specSpeichern(){
  const m = M[$('markt').value]; if (!m) return;
  ls.set('spec-' + m.t, JSON.stringify({k: $('kgroesse').value, w: $('qwaehr').value,
                                         step: $('lotstep').value, min: $('lotmin').value}));
}

function rechnen(){
  ls.set('konto', $('konto').value); ls.set('risiko', $('risiko').value); ls.set('kwaehr', $('kwaehr').value);
  const m = M[$('markt').value]; const out = $('ergebnis');
  if (!m) { out.innerHTML = ''; return; }
  const konto = zahl($('konto').value), risiko = zahl($('risiko').value);
  const ein = zahl($('einstieg').value), stop = zahl($('stop').value), ziel = zahl($('ziel').value);
  const k = zahl($('kgroesse').value), qw = $('qwaehr').value, kw = $('kwaehr').value;
  const step = zahl($('lotstep').value) || 0.01, min = zahl($('lotmin').value) || 0.01;
  if (!(konto > 0 && risiko > 0 && ein > 0 && stop > 0 && k > 0) || ein === stop) {
    out.innerHTML = '<p class="mute">Kontostand, Risiko, Einstieg und Stop eingeben.</p>'; return;
  }
  const umr = usdJe(qw) / usdJe(kw);          // Kurswährung -> Kontowährung
  if (!isFinite(umr)) { out.innerHTML = '<p class="warn">Kein Umrechnungskurs für ' + qw + '/' + kw + ' verfügbar.</p>'; return; }
  const richtung = ein > stop ? 'Long' : 'Short';
  const abstand = Math.abs(ein - stop);
  const betrag = konto * risiko / 100;
  const verlustJeLot = abstand * k * umr;
  const roh = betrag / verlustJeLot;
  const lots = Math.floor(roh / step + 1e-9) * step;
  const dec = (String(step).split('.')[1] || '').length;
  const echt = lots * verlustJeLot;
  let html = '<div class="gross">' + fmt(lots, dec) + ' <small>Lot · ' + richtung + '</small></div>';
  html += '<div class="kv">';
  html += '<span>Risikobetrag</span><span>' + fmt(betrag) + ' ' + kw + '</span>';
  html += '<span>Tatsächliches Risiko</span><span>' + fmt(echt) + ' ' + kw + ' (' + fmt(echt / konto * 100) + ' %)</span>';
  if (m.t.endsWith('=X')) {
    const pip = m.t.indexOf('JPY') >= 0 ? 0.01 : 0.0001;
    html += '<span>Stop-Abstand</span><span>' + fmt(abstand / pip, 1) + ' Pips</span>';
    html += '<span>Pip-Wert</span><span>' + fmt(pip * k * umr) + ' ' + kw + ' pro Lot</span>';
  } else {
    html += '<span>Stop-Abstand</span><span>' + fmt(abstand, m.d) + ' Punkte</span>';
    html += '<span>Wert je Punkt</span><span>' + fmt(k * umr) + ' ' + kw + ' pro Lot</span>';
  }
  if (ziel > 0) {
    const okZiel = richtung === 'Long' ? ziel > ein : ziel < ein;
    if (okZiel) {
      const gew = Math.abs(ziel - ein) * k * umr * lots;
      html += '<span>Gewinn bei Ziel</span><span>' + fmt(gew) + ' ' + kw + '</span>';
      html += '<span>CRV</span><span>' + fmt(Math.abs(ziel - ein) / abstand, 1) + '</span>';
    } else {
      html += '<span>Ziel</span><span class="dn">liegt auf der falschen Seite</span>';
    }
  }
  html += '</div>';
  if (lots < min) html += '<p class="warn">Unter der Mindest-Lotgröße (' + min + ') – das Risiko ist für diesen Stop zu klein.</p>';
  html += '<p class="warn">Kontraktgröße ' + fmt(k, 0) + ' · Kurswährung ' + qw + ' – bitte einmal mit deinem MT5 prüfen.</p>';
  out.innerHTML = html;
}

function waehle(t){
  const i = M.findIndex(m => m.t === t); if (i < 0) return;
  $('markt').value = i; specLaden(M[i]);
  $('einstieg').value = M[i].p; $('stop').value = ''; $('ziel').value = '';
  rechnen();
}

(function(){
  const sel = $('markt');
  let gruppe = '', og = null;
  M.forEach((m, i) => {
    if (m.g !== gruppe) { og = document.createElement('optgroup'); og.label = m.g; sel.appendChild(og); gruppe = m.g; }
    const o = document.createElement('option'); o.value = i; o.textContent = m.n; og.appendChild(o);
  });
  $('konto').value = ls.get('konto') || '';
  $('risiko').value = ls.get('risiko') || '1';
  $('kwaehr').value = ls.get('kwaehr') || 'EUR';
  if (M.length) waehle(M[0].t);
  sel.addEventListener('change', () => { const m = M[sel.value]; specLaden(m); $('einstieg').value = m.p;
                                         $('stop').value = ''; $('ziel').value = ''; rechnen(); });
  ['kgroesse','qwaehr','lotstep','lotmin'].forEach(id => $(id).addEventListener('input', () => { specSpeichern(); rechnen(); }));
  ['konto','risiko','kwaehr','einstieg','stop','ziel'].forEach(id => $(id).addEventListener('input', rechnen));
  $('kwaehr').addEventListener('change', rechnen); $('qwaehr').addEventListener('change', () => { specSpeichern(); rechnen(); });
})();
</script>
"""


def rechner(ergebnis):
    import json
    daten, fx = [], {}
    for g in MAERKTE:
        for z in ergebnis.get(g, []):
            w, k = kontrakt(z["ticker"])
            d = nk(z["kurs"])
            daten.append(dict(t=z["ticker"], n=z["markt"], g=g, w=w, k=k, d=d, p=round(z["kurs"], d)))
            if z["ticker"].endswith("=X"):
                fx[z["ticker"][:6]] = z["kurs"]
    usd = {}
    if "EURUSD" in fx: usd["EUR"] = fx["EURUSD"]
    if "GBPUSD" in fx: usd["GBP"] = fx["GBPUSD"]
    if "AUDUSD" in fx: usd["AUD"] = fx["AUDUSD"]
    if "NZDUSD" in fx: usd["NZD"] = fx["NZDUSD"]
    if "USDJPY" in fx: usd["JPY"] = 1 / fx["USDJPY"]
    if "USDCHF" in fx: usd["CHF"] = 1 / fx["USDCHF"]
    if "USDCAD" in fx: usd["CAD"] = 1 / fx["USDCAD"]
    js = RECHNER_JS.replace("__DATEN__", json.dumps(daten, ensure_ascii=False)).replace("__FX__", json.dumps(usd))
    return RECHNER_HTML, js


def korr_abschnitt(k, bild):
    if k is None or k.empty:
        return ""
    paare = []
    namen = list(k.columns)
    for i in range(len(namen)):
        for j in range(i + 1, len(namen)):
            v = k.iat[i, j]
            if np.isfinite(v) and abs(v) >= KORR_SCHWELLE:
                paare.append((v, namen[i], namen[j]))
    pos = sorted([p for p in paare if p[0] > 0], reverse=True)
    neg = sorted([p for p in paare if p[0] < 0])

    def zeilen(pp):
        return "".join(f"<div><span>{html.escape(a)} ↔ {html.escape(b)}</span>"
                       f"<b class='{'kpos' if v > 0 else 'kneg'}'>{v:+.2f}</b></div>" for v, a, b in pp)

    def liste(pp, top=10):
        if not pp:
            return "<p class='mute'>Keine.</p>"
        out = f"<div class='paare'>{zeilen(pp[:top])}</div>"
        if len(pp) > top:
            out += (f"<details><summary>{len(pp) - top} weitere anzeigen</summary>"
                    f"<div class='paare'>{zeilen(pp[top:])}</div></details>")
        return out

    return f"""<h2 id="korrelation">🔗 Korrelation</h2>
<div class="card korr">
<p class="info">Wie ähnlich sich die Märkte in den letzten {KORR_TAGE} Handelstagen bewegt haben
(tägliche Renditen). <b>+1</b> = laufen gleich, <b>−1</b> = laufen entgegengesetzt, 0 = kein Zusammenhang.
Zwei Trades in stark positiv korrelierten Märkten in dieselbe Richtung sind praktisch ein doppeltes Risiko –
bei stark negativ korrelierten gilt das für entgegengesetzte Richtungen.</p>
<h3 style="margin:14px 0 2px;font-size:16px">Stark positiv (≥ {KORR_SCHWELLE:.2f})</h3>{liste(pos)}
<h3 style="margin:14px 0 2px;font-size:16px">Stark negativ (≤ −{KORR_SCHWELLE:.2f})</h3>{liste(neg)}
<img loading="lazy" src="{bild}?v={VERSION}" alt="Korrelations-Heatmap">
<p class="mute" style="font-size:13px">Bild zum Vergrößern mit zwei Fingern aufziehen. Rot = gleichläufig, Blau = gegenläufig.</p>
</div>"""


def seite(stand, ergebnis, fehler, korr_html=""):
    gruppen = [g for g in MAERKTE if ergebnis.get(g)]
    nav = ("<a href='#rechner'>🧮 Rechner</a>" + "".join(f"<a href='#g-{g}'>{g}</a>" for g in gruppen)
           + ("<a href='#korrelation'>🔗 Korrelation</a>" if korr_html else ""))
    rechner_html, rechner_js = rechner(ergebnis)
    karten = "".join(f"<h2 id='g-{g}'>{g}</h2>" + "".join(karte(z) for z in ergebnis[g]) for g in gruppen)
    fehltext = f"<p class='mute'>Keine Daten für: {html.escape(', '.join(fehler))}</p>" if fehler else ""
    anzahl = sum(len(v) for v in ergebnis.values())
    return f"""<!doctype html>
<html lang="de"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="apple-mobile-web-app-capable" content="yes">
<title>Markt-Übersicht</title><style>{CSS}</style></head>
<body><main id="top">
<h1>📊 Markt-Übersicht</h1>
<div class="stand">Stand {stand} · {anzahl} Märkte · 1d + 4h</div>
<nav>{nav}</nav>
{fehltext}
{rechner_html}
{karten}
{korr_html}
<footer>Volumenprofil über die letzten {KERZEN_1D} Tage, POC orange, Value Area (70 %) blau.
Forex ohne Volumen bei Yahoo → Zeit-Profil.<br>
Kurse: Yahoo Finance (Gold, Öl usw. als Futures – können vom CFD abweichen). Keine Anlageberatung.</footer>
</main>{rechner_js}</body></html>"""

# ----------------------------------------------------------------------------
# Ablauf
# ----------------------------------------------------------------------------
def main():
    os.makedirs(os.path.join(AUSGABE, "charts"), exist_ok=True)
    ergebnis, fehler, schluss = {}, [], {}
    alle = [(g, t, n) for g, m in MAERKTE.items() for t, n in m.items()]
    for i, (gruppe, t, name) in enumerate(alle, 1):
        print(f"{i}/{len(alle)} {name}")
        k = lade_kurse(t)
        d1 = k.get("1d")
        if d1 is None or d1.empty:
            fehler.append(name)
            continue
        d1 = d1.iloc[-KERZEN_1D:]
        vp = volumenprofil(k.get("1h"), d1.index[0], float(d1["Low"].min()), float(d1["High"].max()))
        h4 = k.get("4h")
        z = dict(ticker=t, markt=name, vp=vp, **kennzahlen(k["1d"], vp))
        z["c1d"] = chart_1d(t, name, d1, vp)
        z["c4h"] = chart_4h(t, name, h4.iloc[-KERZEN_4H:]) if h4 is not None and not h4.empty else None
        ergebnis.setdefault(gruppe, []).append(z)
        schluss[name] = k["1d"]["Close"]
        time.sleep(0.5)

    stand = datetime.now(ZoneInfo("Europe/Vienna")).strftime("%d.%m.%Y %H:%M")
    korr_html = ""
    try:
        k = korrelation(schluss)
        k = k.dropna(how="all").dropna(axis=1, how="all")
        if len(k) > 1:
            korr_html = korr_abschnitt(k, chart_korrelation(k))
    except Exception as ex:
        print("Korrelation fehlgeschlagen:", ex)
    with open(os.path.join(AUSGABE, "index.html"), "w", encoding="utf-8") as f:
        f.write(seite(stand, ergebnis, fehler, korr_html))
    print(f"Fertig: {sum(len(v) for v in ergebnis.values())} Märkte, {len(fehler)} ohne Daten")


if __name__ == "__main__":
    main()
