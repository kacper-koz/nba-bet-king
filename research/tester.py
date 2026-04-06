import pandas as pd
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

print("Wczytywanie bazy Elo...")
dane = pd.read_excel("nba_elo_full.xlsx")
dane["Date"] = pd.to_datetime(dane["Date"])
dane["win_num"] = (dane["Rslt"] == "W").astype(int)
dane["net_rating"] = dane["Pkt"] - dane["Opp_Pkt"]

dane['days_rest'] = dane.groupby('Team')['Date'].diff().dt.days
dane['is_b2b'] = (dane['days_rest'] == 1).astype(int)

# Ogólny numer meczu w sezonie (żeby wiedzieć, kiedy jest połowa sezonu)
dane["game_number"] = dane.groupby(["Team", "Season"]).cumcount() + 1

# ==========================================
# 1. LICZYMY STATYSTYKI Z PODZIAŁEM NA DOM/WYJAZD
# ==========================================
# Tutaj dzieje się magia: dodajemy "Home" do grupowania!
statystyki = ["eFG%", "net_rating"]

for stat in statystyki:
    dane[f"{stat}_split"] = (
        dane.groupby(["Team", "Season", "Home"])[stat]
        .expanding().mean().shift(1)
        .reset_index(level=[0,1,2], drop=True)
    )

# ==========================================
# 2. ŁĄCZENIE W PARY
# ==========================================
dane["game_id"] = dane["Date"].astype(str) + "_" + dane[["Team", "Opp"]].apply(lambda x: "_".join(sorted(x)), axis=1)
mecze = dane.merge(dane, on="game_id", suffixes=("_A", "_B"))
mecze = mecze[mecze["Team_A"] < mecze["Team_B"]].copy()

# Drużyna A zawsze jest u nas gospodarzem w tabeli mecze
mecze["Home"] = mecze["Home_A"] 

# Filtr: tylko druga połowa sezonu (>40 meczów ogółem dla obu drużyn)
mecze_pozne = mecze[(mecze["game_number_A"] > 40) & (mecze["game_number_B"] > 40)].copy()

# Upewniamy się, że nie ma braków danych
mecze_pozne = mecze_pozne.dropna(subset=["eFG%_split_A", "eFG%_split_B", "Elo_Pre_A"])

print(f"\nAnalizujemy późny sezon (N = {len(mecze_pozne)}) - PORÓWNANIE DOM vs WYJAZD")

# Różnice kontekstowe: 
# (Historyczna skuteczność A W DOMU) minus (Historyczna skuteczność B NA WYJEŹDZIE)
for stat in statystyki:
    mecze_pozne[f"{stat}_split_diff"] = mecze_pozne[f"{stat}_split_A"] - mecze_pozne[f"{stat}_split_B"]

mecze_pozne["elo_diff"] = mecze_pozne["Elo_Pre_A"] - mecze_pozne["Elo_Pre_B"]
mecze_pozne["b2b_diff"] = mecze_pozne["is_b2b_A"] - mecze_pozne["is_b2b_B"]
mecze_pozne["Y"] = mecze_pozne["win_num_A"]

# ==========================================
# 3. EKONOMETRIA (MODEL LOGITOWY)
# ==========================================
# Wrzucamy Elo, Zmęczenie oraz nasze nowe zmienne "Home/Away Splits"
X = mecze_pozne[["elo_diff", "b2b_diff", "eFG%_split_diff", "net_rating_split_diff"]]
y = mecze_pozne["Y"]

X_sm = sm.add_constant(X)
logit_model = sm.Logit(y, X_sm)
logit_results = logit_model.fit(disp=0)

print("\n" + "="*55)
print("🎯 WYNIKI DLA ROZKŁADÓW DOM/WYJAZD (2. połowa sezonu)")
print("="*55)
print(f"{'Zmienna':<25} | {'Współczynnik':<15} | {'p-value':<10} | {'Wniosek'}")
print("-" * 70)

for var in X.columns:
    p_val = logit_results.pvalues[var]
    coef = logit_results.params[var]
    
    if p_val < 0.05:
        wniosek = "✅ ZŁOTO (Dajemy do Głównego Modelu!)"
    elif p_val < 0.15:
        wniosek = "⚠️ SREBRO (Idealne do Betting Filtra)"
    else:
        wniosek = "❌ SZUM"
        
    print(f"{var:<25} | {coef:>15.4f} | {p_val:>10.4f} | {wniosek}")