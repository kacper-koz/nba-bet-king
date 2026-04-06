import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

print(" INICJALIZACJA SYSTEMU BUKMACHERSKIEGO...")


# 1. PRZYGOTOWANIE BAZY I WYLICZANIE STATYSTYK

dane = pd.read_excel("nba_elo_full.xlsx")
dane.columns = dane.columns.str.strip()
dane = dane.dropna(subset=["Team", "Opp", "Date"])

dane["Date"] = pd.to_datetime(dane["Date"])
dane = dane.sort_values(["Team", "Season", "Date"])
dane["win_num"] = (dane["Rslt"] == "W").astype(int)
dane["net_rating"] = dane["Pkt"] - dane["Opp_Pkt"]

# Numer meczu w sezonie (żeby wiedzieć, kiedy statystyki są stabilne)
dane["game_number"] = dane.groupby(["Team", "Season"]).cumcount() + 1

# Forma krótkoterminowa (10 meczów)
dane["win_num_10"] = (
    dane.groupby(["Team", "Season"])["win_num"]
    .rolling(10).mean().shift(1)
    .reset_index(level=[0,1], drop=True)
)

# Nasz bezpiecznik do filtra: Net Rating w Domu / na Wyjeździe (Expanding)
dane["net_rating_split"] = (
    dane.groupby(["Team", "Season", "Home"])["net_rating"]
    .expanding().mean().shift(1)
    .reset_index(level=[0,1,2], drop=True)
)

# Zmęczenie (B2B)
dane['days_rest'] = dane.groupby('Team')['Date'].diff().dt.days
dane['is_b2b'] = (dane['days_rest'] == 1).astype(int)

# 2. ŁĄCZENIE W PARY I BUDOWA WERSJI ROBOCZEJ

dane["game_id"] = dane["Date"].astype(str) + "_" + dane[["Team", "Opp"]].apply(lambda x: "_".join(sorted(x)), axis=1)

mecze = dane.merge(dane, on="game_id", suffixes=("_A", "_B"))
mecze = mecze[mecze["Team_A"] < mecze["Team_B"]].copy()
mecze["Home"] = mecze["Home_A"]

# Odrzucamy tylko te mecze, gdzie nie ma historii 10 spotkań do Formy i wyliczonego Elo
mecze = mecze.dropna(subset=["win_num_10_A", "win_num_10_B", "Elo_Pre_A", "Elo_Pre_B"])

# Różnice
mecze["elo_diff"] = mecze["Elo_Pre_A"] - mecze["Elo_Pre_B"]
mecze["win_num_diff"] = mecze["win_num_10_A"] - mecze["win_num_10_B"]
mecze["b2b_diff"] = mecze["is_b2b_A"] - mecze["is_b2b_B"]

# Sortowanie CHRONOLOGICZNE (Krytyczne dla braku oszukiwania w zakładach)
mecze = mecze.sort_values("Date_A").reset_index(drop=True)
mecze["Y"] = mecze["win_num_A"]


# 3. SILNIK MODELU (CZYSTY LOGIT)
print(" TRENOWANIE SILNIKA ML NA DANYCH HISTORYCZNYCH...")
# Używamy tylko niezawodnej "Świętej Czwórki"
X = mecze[["elo_diff", "win_num_diff", "b2b_diff", "Home"]]
y = mecze["Y"]

# Podział w czasie (Uczymy się na starszych sezonach, gramy na 20% najnowszych)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Generujemy prawdopodobieństwa dla najnowszych meczów
y_prob = model.predict_proba(X_test)[:, 1]

# Przypisujemy wyniki testowe do czytelnej tabeli, żeby filtr mógł na nich operować
wyniki_testowe = mecze.iloc[X_test.index].copy()
wyniki_testowe["Prob_A"] = y_prob

print("\n" + "="*65)
print(" WIRTUALNY BUKMACHER - WYNIKI SYMULACJI (OSTATNIE 20% MECZÓW)")
print("="*65)


# 4. FILTR PODSTAWOWY (SAME PRAWDOPODOBIEŃSTWA)

print(" STRATEGIA 1: CZYSTE PRAWDOPODOBIEŃSTWO MODELU")
progi = [0.60, 0.65, 0.70, 0.75, 0.80]

print(f"{'Wymagana Pewność':<18} | {'Zostaje Meczów':<15} | {'Skuteczność (Accuracy)':<20}")
print("-" * 65)

total_test_games = len(wyniki_testowe)

for p in progi:
    # Gramy tylko gdy model jest pewny na > P (na A) ALBO < (1-P) (czyli pewny na B)
    zagrane = wyniki_testowe[(wyniki_testowe["Prob_A"] >= p) | (wyniki_testowe["Prob_A"] <= (1-p))]
    
    if len(zagrane) > 0:
        typy = (zagrane["Prob_A"] > 0.5).astype(int)
        acc = accuracy_score(zagrane["Y"], typy)
        proc_meczow = (len(zagrane) / total_test_games) * 100
        print(f"Pewność > {p*100:.0f}%      | {proc_meczow:>5.1f}% ({len(zagrane):>4} gier) | {acc:>10.2%}")
    else:
        print(f"Pewność > {p*100:.0f}%      | Brak meczów spełniających warunek.")


# 5. FILTR ZAAWANSOWANY (SNAJPER + NET RATING SPLIT)

print(" STRATEGIA 2: SNAJPER (Pewność > 65% + Dodatkowe Zabezpieczenia)")
print("Warunki: Model jest w miarę pewny (65%), a my odrzucamy typy z pułapką (np. oszuści własnej hali).")

# Przygotowujemy logikę: Typujemy A (Prob > 0.65) lub Typujemy B (Prob < 0.35)
typ_na_A = wyniki_testowe["Prob_A"] >= 0.65
typ_na_B = wyniki_testowe["Prob_A"] <= 0.35


bezpieczne_A = typ_na_A & (
    (wyniki_testowe["game_number_A"] <= 20) | 
    ((wyniki_testowe["game_number_A"] > 20) & (wyniki_testowe["net_rating_split_A"] > 0)) 
)


bezpieczne_B = typ_na_B & (
    (wyniki_testowe["game_number_B"] <= 20) |
    ((wyniki_testowe["game_number_B"] > 20) & (wyniki_testowe["net_rating_split_B"] > 0))
)   

snajper_mecze = wyniki_testowe[bezpieczne_A | bezpieczne_B]

if len(snajper_mecze) > 0:
    typy_snajpera = (snajper_mecze["Prob_A"] > 0.5).astype(int)
    acc_snajpera = accuracy_score(snajper_mecze["Y"], typy_snajpera)
    proc_snajpera = (len(snajper_mecze) / total_test_games) * 100
    
    print("-" * 65)
    print(f"Wynik Snajpera     | {proc_snajpera:>5.1f}% ({len(snajper_mecze):>4} gier) | {acc_snajpera:>10.2%}")
    print("-" * 65)
 