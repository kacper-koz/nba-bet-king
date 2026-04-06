import pandas as pd
from sklearn.linear_model import LogisticRegression
import subprocess
import sys
import warnings
warnings.filterwarnings('ignore')


print("="*60)
print(" NBA PREDICTOR PRO - LEVEL 1.1")
print("="*60)


# 0. AUTOMATYCZNA AKTUALIZACJA ELO D

print("\n ETAP 1: Aktualizacja bazy danych i przeliczanie systemu Elo...")
try:
    
    subprocess.run([sys.executable, "nba_elo.py"], check=True)
    print(" System Elo zaktualizowany pomyślnie!")
except FileNotFoundError:
    print(" BŁĄD: Nie znaleziono pliku 'nba_elo.py'. Upewnij się, że jest w tym samym folderze.")
    sys.exit(1)
except subprocess.CalledProcessError:
    print(" BŁĄD: Skrypt 'nba_elo.py' napotkał problem (np. masz otwarty plik Excel). Zamknij Excela i spróbuj ponownie.")
    sys.exit(1)


# 1. TRENOWANIE GŁÓWNEGO MODELU NA CAŁEJ BAZIE

print("\n ETAP 2: Inicjalizacja Głównego Silnika ML...")

dane = pd.read_excel("nba_elo_full.xlsx")
dane.columns = dane.columns.str.strip()
dane = dane.dropna(subset=["Team", "Opp", "Date"])
dane["Date"] = pd.to_datetime(dane["Date"])
dane = dane.sort_values(["Team", "Season", "Date"])
dane["win_num"] = (dane["Rslt"] == "W").astype(int)

dane["win_num_10"] = (
    dane.groupby(["Team", "Season"])["win_num"]
    .rolling(10).mean().shift(1)
    .reset_index(level=[0,1], drop=True)
)

dane['days_rest'] = dane.groupby('Team')['Date'].diff().dt.days
dane['is_b2b'] = (dane['days_rest'] == 1).astype(int)
dane["game_id"] = dane["Date"].astype(str) + "_" + dane[["Team", "Opp"]].apply(lambda x: "_".join(sorted(x)), axis=1)

mecze = dane.merge(dane, on="game_id", suffixes=("_A", "_B"))
mecze = mecze[mecze["Team_A"] < mecze["Team_B"]].copy()
mecze["Home"] = mecze["Home_A"]
mecze = mecze.dropna(subset=["win_num_10_A", "win_num_10_B", "Elo_Pre_A", "Elo_Pre_B"])

mecze["elo_diff"] = mecze["Elo_Pre_A"] - mecze["Elo_Pre_B"]
mecze["win_num_diff"] = mecze["win_num_10_A"] - mecze["win_num_10_B"]
mecze["b2b_diff"] = mecze["is_b2b_A"] - mecze["is_b2b_B"]
mecze["Y"] = mecze["win_num_A"]

# Trenujemy na 100% danych
X = mecze[["elo_diff", "win_num_diff", "b2b_diff", "Home"]]
y = mecze["Y"]

model = LogisticRegression(max_iter=1000)
model.fit(X, y)

print(" Model nauczony! Gotowy do pracy.")
print("="*60)

# 2. FUNKCJE POMOCNICZE (TERAZ Z DATĄ OSTATNIEGO MECZU)

def get_current_stats(team_name, df):
    team_df = df[df["Team"] == team_name].sort_values("Date")
    if len(team_df) == 0:
        return None
    
    ostatnie_10 = team_df.tail(10)
    aktualna_forma = ostatnie_10["win_num"].mean()
    aktualne_elo = team_df.iloc[-1]["Elo_Pre"]
    ostatnia_data = team_df.iloc[-1]["Date"] 
    
    return {"elo": aktualne_elo, "forma_10": aktualna_forma, "ostatnia_data": ostatnia_data}


# 3. INTERFEJS UŻYTKOWNIKA 

from datetime import datetime

while True:
    print(" WPROWADŹ NOWY MECZ DO ANALIZY (lub wpisz 'exit' aby wyjść):")
    gospodarz = input("Podaj skrót GOSPODARZA (np. BOS, LAL, MIA): ").strip().upper()
    if gospodarz == 'EXIT': break
    
    gosc = input("Podaj skrót GOŚCIA (np. NYK, DEN, GSW): ").strip().upper()
    if gosc == 'EXIT': break

    #  Automatyzacja B2B: Pytamy o datę meczu 
    data_input = input("Podaj datę meczu (RRRR-MM-DD) [Wciśnij ENTER jeśli mecz jest dzisiaj]: ").strip()
    if not data_input:
        data_meczu = pd.Timestamp.today().normalize()
    else:
        try:
            data_meczu = pd.to_datetime(data_input)
        except ValueError:
            print(" Błąd: Niepoprawny format daty. Użyj formatu RRRR-MM-DD.")
            continue

    stats_A = get_current_stats(gospodarz, dane)
    stats_B = get_current_stats(gosc, dane)

    if not stats_A or not stats_B:
        print(" Błąd: Nie znaleziono jednej z drużyn w bazie danych. Sprawdź skróty!")
        continue

    
    
    b2b_A = 1 if (data_meczu - stats_A["ostatnia_data"]).days == 1 else 0
    b2b_B = 1 if (data_meczu - stats_B["ostatnia_data"]).days == 1 else 0

    print(f"\n[INFO B2B]: Ostatni mecz {gospodarz}: {stats_A['ostatnia_data'].date()} | B2B: {'TAK' if b2b_A else 'NIE'}")
    print(f"[INFO B2B]: Ostatni mecz {gosc}: {stats_B['ostatnia_data'].date()} | B2B: {'TAK' if b2b_B else 'NIE'}")

    # Obliczamy różnice tak jak w modelu
    elo_diff = stats_A["elo"] - stats_B["elo"]
    win_num_diff = stats_A["forma_10"] - stats_B["forma_10"]
    b2b_diff = b2b_A - b2b_B
    home = 1 

    # Formatowanie wektora dla sklearn
    mecz_do_typowania = pd.DataFrame([[elo_diff, win_num_diff, b2b_diff, home]], 
                                     columns=["elo_diff", "win_num_diff", "b2b_diff", "Home"])

    # PRZEWIDYWANIE
    prawd_A = model.predict_proba(mecz_do_typowania)[0][1]
    prawd_B = 1 - prawd_A

    print("\n" + "="*50)
    print(f" Wynik Analizy ML: {gospodarz} (Dom) vs {gosc} (Wyjazd)")
    print("="*50)
    print(f"Szansa na wygraną {gospodarz}: {prawd_A:.1%}")
    print(f"Szansa na wygraną {gosc}: {prawd_B:.1%}")
    
    # REKOMENDACJA BUKMACHERSKA
    pewnosc = max(prawd_A, prawd_B)
    faworyt = gospodarz if prawd_A > prawd_B else gosc
    
    print(" REKOMENDACJA SYSTEMU (BET KING):")
    if pewnosc >= 0.80:
        print(f" ZŁOTY TYP (DIAMOND): Stawiaj na {faworyt}.")
        print("   Historyczna trafność: ~85%. Mecz łapie się do Top 25% najlepszych okazji.")
    elif pewnosc >= 0.70:
        print(f" SREBRNY TYP (GOLD): Stawiaj na {faworyt}.")
        print("   Historyczna trafność: >76%. Solidny typ.")
    elif pewnosc >= 0.60:
        print(f" BRĄZOWY TYP (SILVER): Lekka przewaga {faworyt}.")
        print("   Historyczna trafność: ~69%. Można zagrać za mniejszą stawkę.")
    else:
        print(f" NO BET (ODRZUCONE): Mecz nie łapie się do modelu.")
        print("   Pewność poniżej 60%. Zbyt duże ryzyko, tzw. rzut monetą.")
    print("="*50)