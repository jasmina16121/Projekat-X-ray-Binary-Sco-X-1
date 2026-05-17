import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. UČITAVANJE PODATAKA I IZBOR FOTOMETRIJSKOG FILTERA
# ============================================================

print("\n**** Izbor filtera ****")
def ucitaj_krivu(fajl, ime_objekta, filter_kanal="g"):
    df = pd.read_csv(fajl, comment="#")  # učitavanje CSV fajla; redovi koji počinju sa # se preskaču

    df = df.replace([np.inf, -np.inf], np.nan)  # beskonačne vrednosti se pretvaraju u NaN

    df = df[df["Filter"] == filter_kanal]  # koristi se samo jedan filter zbog konzistentnog poređenja

    df = df.dropna(subset=["JD", "Flux", "Flux Error", "Mag", "Mag Error"])  # uklanjanje nepotpunih merenja

    df = df.sort_values("JD").reset_index(drop=True)  # sortiranje po vremenu

    vreme = df["JD"]  # vreme posmatranja u Julijanskim danima
    fluks = df["Flux"]  # instrumentalni fluks
    greska_fluksa = df["Flux Error"]  # greška merenja fluksa
    
    print(f"\n{ime_objekta}")
    print("Broj tačaka posle izbora filtera:", len(vreme))
    print("JD opseg:", np.min(vreme), "-", np.max(vreme))

    return df, vreme, fluks, greska_fluksa


# ============================================================
# 2. UČITAVANJE KRIVIH SJAJA ZA SVA TRI SISTEMA
# ============================================================

df_sco, vreme_sco, fluks_sco, greska_sco = ucitaj_krivu("sco.csv", "Sco X-1")
df_her, vreme_her, fluks_her, greska_her = ucitaj_krivu("her.csv", "Her X-1")
df_cyg, vreme_cyg, fluks_cyg, greska_cyg = ucitaj_krivu("cyg.csv", "Cyg X-2")


# ============================================================
# 3. FUNKCIJA ZA PRIKAZ KRIVE SJAJA U FLUKSU
# ============================================================

def nacrtaj_fluks(vreme, fluks, greska, ime_objekta):
    jd0 = 2450000
    plt.figure(figsize=(8, 5))

    plt.errorbar(
        vreme - jd0, fluks,
        yerr=greska,             # greške se ne izbacuju; prikazuju se kao errorbar
        fmt=".",                 # tačke
        markersize=2,            # veličina tačke
        elinewidth=0.5,          # širina linije greške
        capsize=1,               # veličina tačke na liniji greške
        alpha=0.6,               # transparentnost
        label=ime_objekta
    )
    plt.xlabel(f"JD - {jd0}")    # JD 2450000.0=1995−10−09 12:00 UT
    plt.ylabel(r" g Fluks [mJy]")
    plt.title(f"Kriva sjaja: {ime_objekta}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# ============================================================
# 4. FUNKCIJA ZA PRIKAZ KRIVE SJAJA U MAGNITUDAMA
# ============================================================

def nacrtaj_magnitudu(df, ime_objekta):
    jd0 = 2450000
    vreme = df["JD"]
    magnituda = df["Mag"]
    greska_magnitude = df["Mag Error"]

    plt.figure(figsize=(8, 5))

    plt.errorbar(
        vreme - jd0, magnituda,
        yerr=greska_magnitude,
        fmt=".",
        markersize=2,
        elinewidth=0.5,
        capsize=1,
        alpha=0.6,
        label=ime_objekta
    )

    plt.gca().invert_yaxis()  # veća magnitudama manji sjaj
    plt.xlabel(f"JD - {jd0}")
    plt.ylabel(r"$g$ magnituda")
    plt.title(f"Kriva sjaja (magnituda): {ime_objekta}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# ============================================================
# 5. POČETNI VIZUELNI PREGLED SIROVIH KRIVIH SJAJA
# ============================================================

nacrtaj_fluks(vreme_sco, fluks_sco, greska_sco, "Sco X-1")
nacrtaj_fluks(vreme_her, fluks_her, greska_her, "Her X-1")
nacrtaj_fluks(vreme_cyg, fluks_cyg, greska_cyg, "Cyg X-2")

# Po potrebi se mogu prikazati i magnitude:
#nacrtaj_magnitudu(df_sco, "Sco X-1")
#nacrtaj_magnitudu(df_her, "HZ Her")
#nacrtaj_magnitudu(df_cyg, "Cyg X-2")


# ============================================================
# 6. ROBUSNI SIGMA-CLIPPING OUTLIERA POMOĆU MEDIJANE I MAD
# ============================================================

def sigma_clipping_mad(vreme, fluks, greska, prag=5):
    medijana = np.median(fluks)  # medijana je stabilnija od srednje vrednosti kod outliera

    mad = np.median(np.abs(fluks - medijana))  # MAD = median absolute deviation

    sigma_rob = 1.4826 * mad  # pretvaranje MAD u procenu standardne devijacije

    donja_granica = medijana - prag * sigma_rob  # donja granica prihvatljivog fluksa
    gornja_granica = medijana + prag * sigma_rob  # gornja granica prihvatljivog fluksa

    maska = (fluks > donja_granica) & (fluks < gornja_granica)  # zadržavaju se samo normalne tačke

    return vreme[maska], fluks[maska], greska[maska]


vreme_sco_cisto1, fluks_sco_cist1, greska_sco_cista1 = sigma_clipping_mad(
    vreme_sco, fluks_sco, greska_sco
)

vreme_her_cisto1, fluks_her_cist1, greska_her_cista1 = sigma_clipping_mad(
    vreme_her, fluks_her, greska_her
)

vreme_cyg_cisto1, fluks_cyg_cist1, greska_cyg_cista1 = sigma_clipping_mad(
    vreme_cyg, fluks_cyg, greska_cyg
)


# ============================================================
# 7. UKLANJANJE TAČAKA SA PREVELIKOM FOTOMETRIJSKOM GREŠKOM
# ============================================================

def ukloni_velike_greske(vreme, fluks, greska, procenat=95):
    granica_greske = np.percentile(greska, procenat)  # uzima se 95. percentil greške

    maska = greska < granica_greske  # uklanja se najgorih 5% merenja po grešci

    return vreme[maska], fluks[maska], greska[maska]


vreme_sco_cisto2, fluks_sco_cist2, greska_sco_cista2 = ukloni_velike_greske(
    vreme_sco_cisto1, fluks_sco_cist1, greska_sco_cista1
)

vreme_her_cisto2, fluks_her_cist2, greska_her_cista2 = ukloni_velike_greske(
    vreme_her_cisto1, fluks_her_cist1, greska_her_cista1
)

vreme_cyg_cisto2, fluks_cyg_cist2, greska_cyg_cista2 = ukloni_velike_greske(
    vreme_cyg_cisto1, fluks_cyg_cist1, greska_cyg_cista1
)





# ============================================================
# 8. VIZUELNA PROVERA KONAČNO FILTRIRANIH KRIVIH SJAJA
# ============================================================

nacrtaj_fluks(vreme_sco_cisto2, fluks_sco_cist2, greska_sco_cista2, "Sco X-1 nakon čišćenja")
nacrtaj_fluks(vreme_her_cisto2, fluks_her_cist2, greska_her_cista2, "Her X-1 nakon čišćenja")
nacrtaj_fluks(vreme_cyg_cisto2, fluks_cyg_cist2, greska_cyg_cista2, "Cyg X-2 nakon čišćenja")


# ============================================================
# 9. PROVERA BROJA TAČAKA PRE I POSLE FILTRIRANJA
# ============================================================

print("\n**** Broj tačaka pre i posle čišćenja ****\n")

print("Sco X-1:", len(fluks_sco), "->", len(fluks_sco_cist2))
print("Her X-1:", len(fluks_her), "->", len(fluks_her_cist2))
print("Cyg X-2:", len(fluks_cyg), "->", len(fluks_cyg_cist2))


# ============================================================
# 10. OSNOVNA STATISTIKA VARIJABILNOSTI 
# ============================================================

from scipy.stats import skew, kurtosis
print("\n**** Osnovna statistika ****")
def osnovna_statistika(vreme, fluks, greska, ime_objekta):
    # srednja vrednost fluksa
    srednja = np.mean(fluks)
    
    # standardna devijacija
    std = np.std(fluks)
    
    # RMS (root mean square)
    rms = np.sqrt(np.mean(fluks**2))
    
    # amplituda (raspon)
    amplituda = np.max(fluks) - np.min(fluks)
    
    # ========================================================
    # ROBUSNA AMPLITUDA (P95 - P5)
    # ========================================================
    p95 = np.percentile(fluks, 95)
    p5 = np.percentile(fluks, 5)
    amplituda_rob = p95 - p5
    
    # ========================================================
    # OBLIK RASPODELE
    # ========================================================
    skewness = skew(fluks)
    kurt = kurtosis(fluks)  # Fisher (0 = normalna raspodela)
    
    # srednja kvadratna greška
    srednja_greska2 = np.mean(greska**2)
    
    # Fvar (frakcijska varijabilnost)
    if std**2 > srednja_greska2:
        fvar = np.sqrt(std**2 - srednja_greska2) / srednja
    else:
        fvar = np.nan
    
    # ========================================================
    # ISPIS
    # ========================================================
    print(f"\n{ime_objekta}")
    print(f"Srednja vrednost: {srednja:.3f}")
    print(f"Standardna devijacija: {std:.3f}")
    print(f"RMS: {rms:.3f}")
    print(f"Amplituda: {amplituda:.3f}")
    print(f"Robusna amplituda (P95-P5): {amplituda_rob:.3f}")
    print(f"Fvar: {fvar:.3f}")
    print(f"Skewness: {skewness:.3f}")
    print(f"Kurtosis: {kurt:.3f}")

    return {
        "sistem": ime_objekta,
        "srednja": srednja,
        "std": std,
        "rms": rms,
        "amplituda": amplituda,
        "amplituda_rob": amplituda_rob,
        "fvar": fvar,
        "skewness": skewness,
        "kurtosis": kurt
    }


# ============================================================
# PRIMENA NA FILTRIRANE PODATKE
# ============================================================

stat_sco = osnovna_statistika(
    vreme_sco_cisto2, fluks_sco_cist2, greska_sco_cista2, "Sco X-1"
)

stat_her = osnovna_statistika(
    vreme_her_cisto2, fluks_her_cist2, greska_her_cista2, "Her X-1"
)

stat_cyg = osnovna_statistika(
    vreme_cyg_cisto2, fluks_cyg_cist2, greska_cyg_cista2, "Cyg X-2"
)







# ============================================================
# 11. HISTOGRAMI RASPODELE FLUKSA
# ============================================================

def nacrtaj_histogram(fluks, ime_objekta):
    plt.figure(figsize=(8,5))

    # histogram
    plt.hist(fluks, bins=30, density=True, alpha=0.6, label="Histogram")

    # KDE (glatka kriva)
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(fluks)
    x = np.linspace(min(fluks), max(fluks), 500)
    kde_vrednosti = kde(x)
    plt.plot(x, kde_vrednosti, label="KDE")

    plt.xlabel("Fluks [mJy]")
    plt.ylabel(r"$p(F)$ [$\mathrm{mJy}^{-1}$]")
    plt.title(f"Raspodela fluksa: {ime_objekta}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    return {
        "sistem": ime_objekta,
        "fluks": fluks,
        "x_kde": x,
        "kde": kde_vrednosti
    }


# Sco
hist_sco = nacrtaj_histogram(fluks_sco_cist2, "Sco X-1")

# Her
hist_her = nacrtaj_histogram(fluks_her_cist2, "Her X-1")

# Cyg
hist_cyg = nacrtaj_histogram(fluks_cyg_cist2, "Cyg X-2")


#--------------------------------------------------------------------------


# ============================================================
# 12. LOMB–SCARGLE PERIODOGRAM
# ============================================================

from astropy.timeseries import LombScargle
print("\n**** Lomb - Scargle ****")
def nacrtaj_periodogram(vreme, fluks, ime_objekta):
    # frekvencije
    frekvencija, snaga = LombScargle(vreme, fluks).autopower()

    # pretvaranje u period
    period = 1 / frekvencija

    plt.figure(figsize=(8,5))
    plt.plot(period, snaga)

    plt.xlabel("Period (days)")
    plt.ylabel("Lomb-Scargle Snaga")
    plt.title(f"Lomb–Scargle periodogram: {ime_objekta}")

    plt.xscale("log")  # bolje se vide različite skale
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # dominantni period
    indeks_max = np.argmax(snaga)
    print(f"\n{ime_objekta}")
    print("Dominantni period:", period[indeks_max])

    return {
        "sistem": ime_objekta,
        "frekvencija": frekvencija,
        "period": period,
        "snaga": snaga,
        "dominantni_period": period[indeks_max]
    }


# Sco
period_sco = nacrtaj_periodogram(vreme_sco_cisto2, fluks_sco_cist2, "Sco X-1")

# Her
period_her = nacrtaj_periodogram(vreme_her_cisto2, fluks_her_cist2, "Her X-1")

# Cyg
period_cyg = nacrtaj_periodogram(vreme_cyg_cisto2, fluks_cyg_cist2, "Cyg X-2")


# ============================================================
# 13. AUTOKORELACIONA FUNKCIJA (DCF – binovana ACF)
# ============================================================

def dcf(vreme, fluks, max_lag=50, bin_width=1.0):
    # sortiranje
    idx = np.argsort(vreme)
    t = np.array(vreme)[idx]
    f = np.array(fluks)[idx]

    f_mean = np.mean(f)
    f_std = np.std(f)

    lags = []
    udcf = []

    # sve parove
    for i in range(len(t)):
        for j in range(i+1, len(t)):
            dt = t[j] - t[i]
            if dt > max_lag:
                break
            val = ((f[i]-f_mean)*(f[j]-f_mean)) / (f_std**2)
            lags.append(dt)
            udcf.append(val)

    lags = np.array(lags)
    udcf = np.array(udcf)

    # binovanje po lag-u
    bins = np.arange(0, max_lag+bin_width, bin_width)
    bin_centers = 0.5*(bins[:-1] + bins[1:])
    dcf_vals = []

    for k in range(len(bins)-1):
        mask = (lags >= bins[k]) & (lags < bins[k+1])
        if np.any(mask):
            dcf_vals.append(np.mean(udcf[mask]))
        else:
            dcf_vals.append(np.nan)

    return bin_centers, np.array(dcf_vals)


def nacrtaj_acf(vreme, fluks, ime):
    lag, acf = dcf(vreme, fluks, max_lag=50, bin_width=1.0)

    plt.figure(figsize=(8,5))
    plt.plot(lag, acf, marker='o', ms=3)
    plt.axhline(0, linestyle='--', linewidth=1)
    plt.xlabel(r"Vremenski pomak $\tau$ [days]")
    plt.ylabel("DCF")
    plt.title(f"Autokorelaciona funkcija: {ime}")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    return {
        "sistem": ime,
        "lag": lag,
        "acf": acf
    }


# pozivi
acf_sco = nacrtaj_acf(vreme_sco_cisto2, fluks_sco_cist2, "Sco X-1")
acf_her = nacrtaj_acf(vreme_her_cisto2, fluks_her_cist2, "Her X-1")
acf_cyg = nacrtaj_acf(vreme_cyg_cisto2, fluks_cyg_cist2, "Cyg X-2")






# ============================================================
# 14. MIKROVARIJABILNOST 
# ============================================================
print("\n**** Mikrovarijabilnost ****")
def mikrovarijabilnost(vreme, fluks, ime_objekta):
    vreme = np.array(vreme)
    fluks = np.array(fluks)

    # sortiranje po vremenu
    redosled = np.argsort(vreme)
    vreme = vreme[redosled]
    fluks = fluks[redosled]

    # razlike
    delta_vreme = np.diff(vreme)   # Δt
    delta_fluks = np.diff(fluks)   # ΔF

    # ======================================================
    # KLJUČNO: uklanjanje malih Δt 
    # ======================================================
    maska_dt = delta_vreme > 0.01   # ~15 minuta

    delta_vreme = delta_vreme[maska_dt]
    delta_fluks = delta_fluks[maska_dt]

    # brzina promene
    brzina_promene = delta_fluks / delta_vreme

    # ======================================================
    # uklanjanje NaN i inf vrednosti (dodatna sigurnost)
    # ======================================================
    maska_valid = np.isfinite(brzina_promene)
    brzina_promene = brzina_promene[maska_valid]

    # ======================================================
    # uklanjanje ekstremnih outliera (robusnije)
    # ======================================================
    med = np.median(brzina_promene)
    mad = np.median(np.abs(brzina_promene - med))
    sigma_rob = 1.4826 * mad

    maska_out = (brzina_promene > med - 5*sigma_rob) & (brzina_promene < med + 5*sigma_rob)
    brzina_promene = brzina_promene[maska_out]

    # ======================================================
    # statistika
    # ======================================================
    srednja = np.mean(brzina_promene)
    std = np.std(brzina_promene)
    max_abs = np.max(np.abs(brzina_promene))

    print(f"\n{ime_objekta}")
    print(f"Srednja ΔF/Δt: {srednja:.3f}")
    print(f"Std ΔF/Δt: {std:.3f}")
    print(f"Max |ΔF/Δt|: {max_abs:.3f}")
    print(f"Broj tačaka: {len(brzina_promene)}")

    # ======================================================
    # histogram
    # ======================================================
    plt.figure(figsize=(8,5))
    plt.hist(brzina_promene, bins=40, alpha=0.7)
    plt.xlabel(r"$\Delta F / \Delta t \, [mJy / day]$")
    plt.ylabel("Broj tačaka")
    plt.title(f"Mikrovarijabilnost: {ime_objekta}")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    return {
        "sistem": ime_objekta,
        "brzina_promene": brzina_promene,
        "srednja": srednja,
        "std": std,
        "max_abs": max_abs
    }


# ============================================================
# POZIV
# ============================================================

mikro_sco = mikrovarijabilnost(vreme_sco_cisto2, fluks_sco_cist2, "Sco X-1")
mikro_her = mikrovarijabilnost(vreme_her_cisto2, fluks_her_cist2, "Her X-1")
mikro_cyg = mikrovarijabilnost(vreme_cyg_cisto2, fluks_cyg_cist2, "Cyg X-2")


# ============================================================
# 15. DODATNA VREMENSKA ANALIZA (SEGMENTACIJA)
# ============================================================
print("\n**** Segmentacija ****")
def segmentacija(vreme, fluks, ime_objekta, broj_segmenata=3):
    vreme = np.array(vreme)
    fluks = np.array(fluks)

    # sortiranje
    redosled = np.argsort(vreme)
    vreme = vreme[redosled]
    fluks = fluks[redosled]

    # delimo na segmente
    segmenti = np.array_split(np.arange(len(vreme)), broj_segmenata)

    rezultati_segmenata = []
    
    jd0 = 2450000
    plt.figure(figsize=(8,5))

    for i, seg in enumerate(segmenti):
        plt.scatter(vreme[seg]-jd0, fluks[seg], s=5, label=f"Segment {i+1}")

        # osnovna statistika po segmentu
        srednja = np.mean(fluks[seg])
        std = np.std(fluks[seg])

        print(f"\n{ime_objekta} - Segment {i+1}")
        print(f"Srednja vrednost: {srednja:.3f}")
        print(f"Std: {std:.3f}")

        rezultati_segmenata.append({
            "segment": i + 1,
            "srednja": srednja,
            "std": std
        })
    plt.xlabel(f"JD - {jd0}")
    plt.ylabel("Fluks [mJy]")
    plt.title(f"Segmentacija svetlosne krive: {ime_objekta}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    return {
        "sistem": ime_objekta,
        "segmenti": rezultati_segmenata
    }


# poziv
seg_sco = segmentacija(vreme_sco_cisto2, fluks_sco_cist2, "Sco X-1")
seg_her = segmentacija(vreme_her_cisto2, fluks_her_cist2, "Her X-1")
seg_cyg = segmentacija(vreme_cyg_cisto2, fluks_cyg_cist2, "Cyg X-2")



# ============================================================
# 16. FIZIČKI SMISLENO POREĐENJE SISTEMA
#     UPOREDNA ANALIZA OSNOVNIH PARAMETARA VARIJABILNOSTI
# ============================================================
print("\n**** Poređenje parametara ****")
tabela_uporedna = pd.DataFrame({
    "Sistem": ["Sco X-1", "Her X-1", "Cyg X-2"],

    "Fvar": [
        stat_sco["fvar"],
        stat_her["fvar"],
        stat_cyg["fvar"]
    ],

    "Relativna amplituda": [
        stat_sco["amplituda"] / stat_sco["srednja"],
        stat_her["amplituda"] / stat_her["srednja"],
        stat_cyg["amplituda"] / stat_cyg["srednja"]
    ],

    "Mikrovarijabilnost": [
        mikro_sco["std"],
        mikro_her["std"],
        mikro_cyg["std"]
    ]
})




for i, sistem in tabela_uporedna.iterrows():

    ime = sistem["Sistem"]

    fvar = sistem["Fvar"]
    amp = sistem["Relativna amplituda"]
    mikro = sistem["Mikrovarijabilnost"]

    odnos_brzine = mikro / fvar
    odnos_amplitude = amp / fvar

    print(f"\n{ime}")
    print(f"Odnos brzine i varijabilnosti (mikro/Fvar): {odnos_brzine:.3f}")
    print(f"Odnos amplitude i varijabilnosti (A/Fvar): {odnos_amplitude:.3f}")






# ============================================================
# 17. KS TEST 
# ============================================================
print("\n**** KS Test ****")
def relativni_fluks(fluks):
    return fluks / np.mean(fluks)

sco_rel = relativni_fluks(fluks_sco_cist2)
her_rel = relativni_fluks(fluks_her_cist2)
cyg_rel = relativni_fluks(fluks_cyg_cist2)



from scipy.stats import ks_2samp

def ks_test(a, b, ime1, ime2):
    stat, p = ks_2samp(a, b)

    print(f"\n{ime1} vs {ime2}")
    print(f"KS statistika: {stat:.4f}")
    print(f"p-vrednost: {p:.4e}")

ks_test(sco_rel, her_rel, "Sco X-1", "Her X-1")
ks_test(sco_rel, cyg_rel, "Sco X-1", "Cyg X-2")
ks_test(her_rel, cyg_rel, "Her X-1", "Cyg X-2")




# ============================================================
# 18. KULLBACK-LEIBLER DIVERGENCIJA
#    POREĐENJE RASPODELA RELATIVNOG FLUKSA
# ============================================================

from scipy.stats import entropy

print("\n**** KL Divergencija ****")

def kl_divergencija(a, b, ime1, ime2, bins=40):
    # zajednički opseg da bi histogrami bili uporedivi
    xmin = min(np.min(a), np.min(b))
    xmax = max(np.max(a), np.max(b))

    # histogrami kao procena raspodele verovatnoće
    p, binovi = np.histogram(a, bins=bins, range=(xmin, xmax), density=False)
    q, _ = np.histogram(b, bins=binovi, density=False)

    

# ============================================================
# PLOT HISTOGRAMA
# ============================================================

    plt.figure(figsize=(8,5))

    plt.hist(a,
         bins=binovi,
         density=True,
         alpha=0.45,
         label=ime1,
         histtype="stepfilled")

    plt.hist(b,
         bins=binovi,
         density=True,
         alpha=0.45,
         label=ime2,
         histtype="stepfilled")

    plt.xlabel("Fluks [mJy]")
    plt.ylabel(r"$p(F)$ [$\mathrm{mJy}^{-1}$]")

    plt.title(f"KL poređenje: {ime1} vs {ime2}")

    plt.legend()
    plt.grid(alpha=0.3)

    plt.show()


# pretvaranje u verovatnoće
    p = p / np.sum(p)
    q = q / np.sum(q)

    # mala vrednost da se izbegne deljenje nulom i log(0)
    eps = 1e-12
    p = p + eps
    q = q + eps

    # ponovna normalizacija
    p = p / np.sum(p)
    q = q / np.sum(q)

    # KL divergencija
    dkl_pq = entropy(p, q)
    dkl_qp = entropy(q, p)
      
    print(f"\n{ime1} vs {ime2}")
    print(f"D_KL({ime1} || {ime2}) = {dkl_pq:.4f}")
    print(f"D_KL({ime2} || {ime1}) = {dkl_qp:.4f}")

    return {
        "poredjenje": f"{ime1} vs {ime2}",
        "D_KL_1_2": dkl_pq,
        "D_KL_2_1": dkl_qp
    }


kl_sco_her = kl_divergencija(sco_rel, her_rel, "Sco X-1", "Her X-1")
kl_sco_cyg = kl_divergencija(sco_rel, cyg_rel, "Sco X-1", "Cyg X-2")
kl_her_cyg = kl_divergencija(her_rel, cyg_rel, "Her X-1", "Cyg X-2")









