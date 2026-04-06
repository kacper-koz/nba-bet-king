import pandas as pd
import numpy as np
import math


# 1. WCZYTANIE DANYCH

dane = pd.read_excel("nba_dane.xlsx")

# Czyszczenie
dane.columns = dane.columns.str.strip()
dane["Date"] = pd.to_datetime(dane["Date"])
dane = dane.dropna(subset=["Team", "Opp", "Date", "Rslt", "Pkt", "Opp_Pkt", "Season"])

# Sortowanie 
dane = dane.sort_values("Date").reset_index(drop=True)


# 2. BACK-TO-BACK 

# Tworzymy pełny kalendarz każdej drużyny 
wszystkie_mecze = pd.concat([
    dane[['Date', 'Team']].rename(columns={'Team': 'Klub'}),
    dane[['Date', 'Opp']].rename(columns={'Opp': 'Klub'})
]).drop_duplicates().sort_values(['Klub', 'Date'])

# Liczymy dni odpoczynku dla każdego występu
wszystkie_mecze['days_rest'] = wszystkie_mecze.groupby('Klub')['Date'].diff().dt.days
wszystkie_mecze['is_b2b'] = (wszystkie_mecze['days_rest'] == 1).astype(int)

# Podpinamy wynik do głównej tabeli dla Gospodarza (Team) i Gościa (Opp)
dane = dane.merge(wszystkie_mecze, left_on=['Date', 'Team'], right_on=['Date', 'Klub'], how='left')
dane.rename(columns={'is_b2b': 'is_b2b_team'}, inplace=True)
dane = dane.drop(columns=['Klub', 'days_rest'])

dane = dane.merge(wszystkie_mecze, left_on=['Date', 'Opp'], right_on=['Date', 'Klub'], how='left')
dane.rename(columns={'is_b2b': 'is_b2b_opp'}, inplace=True)
dane = dane.drop(columns=['Klub', 'days_rest'])


# 3. ZOPTYMALIZOWANE PARAMETRY MODELU

BASE_ELO = 1500
K_BASE = 24      
HOME_ADV = 65     
B2B_PENALTY = 60  


# 4. STRUKTURY

elo_dict = {}
last_season = {}

elo_pre_list = []
elo_opp_pre_list = []
exp_win_list = []


# 5. FUNKCJE POMOCNICZE

def expected_win(elo_a, elo_b):
    return 1 / (1 + 10 ** ((elo_b - elo_a) / 400))


def margin_multiplier(margin, elo_diff):
    # Oficjalna formuła FiveThirtyEight dla NBA
    return ((margin + 3) ** 0.8) / (7.5 + 0.006 * elo_diff)


# 6. GŁÓWNA PĘTLA

dane = dane.sort_values("Date").reset_index(drop=True)

for idx, row in dane.iterrows():
    team = row["Team"]
    opp = row["Opp"]
    season = row["Season"]

    # inicjalizacja 
    if team not in elo_dict:
        elo_dict[team] = BASE_ELO
    if opp not in elo_dict:
        elo_dict[opp] = BASE_ELO

    #  regresja sezonowa
    if team in last_season and last_season[team] != season:
        elo_dict[team] = 0.75 * elo_dict[team] + 0.25 * BASE_ELO
    if opp in last_season and last_season[opp] != season:
        elo_dict[opp] = 0.75 * elo_dict[opp] + 0.25 * BASE_ELO

    last_season[team] = season
    last_season[opp] = season

    #Elo przed meczem
    team_elo_pre = elo_dict[team]
    opp_elo_pre = elo_dict[opp]

    elo_pre_list.append(team_elo_pre)
    elo_opp_pre_list.append(opp_elo_pre)

    #  ADJUSTMENTS 
    team_adj = (
        team_elo_pre
        + (HOME_ADV if row["Home"] == 1 else 0)
        - (B2B_PENALTY if row["is_b2b_team"] == 1 else 0)
    )

    opp_adj = (
        opp_elo_pre
        + (HOME_ADV if row["Home"] == 0 else 0)
        - (B2B_PENALTY if row["is_b2b_opp"] == 1 else 0)
    )

    
    exp_team = expected_win(team_adj, opp_adj)
    exp_win_list.append(exp_team)

    # --- wynik ---
    actual = 1 if row["Rslt"] == "W" else 0

    margin = abs(row["Pkt"] - row["Opp_Pkt"])
    
    # Różnica Elo z perspektywy drużyny, która WYGRAŁA mecz
    if actual == 1:
        elo_diff_winner = team_elo_pre - opp_elo_pre
    else:
        elo_diff_winner = opp_elo_pre - team_elo_pre
        
    mult = margin_multiplier(margin, elo_diff_winner)

    #  update 
    change = K_BASE * mult * (actual - exp_team)

    elo_dict[team] += change
    elo_dict[opp] -= change


# 7. ZAPIS

dane["Elo_Pre"] = elo_pre_list
dane["Opp_Elo_Pre"] = elo_opp_pre_list
dane["Exp_Win"] = exp_win_list

# GAME ID
dane["game_id"] = (
    dane["Date"].astype(str)
    + "_"
    + dane[["Team", "Opp"]].apply(lambda x: "_".join(sorted(x)), axis=1)
)

# zapis
dane.to_excel("nba_elo_full.xlsx", index=False)

print("Zoptymalizowany system Elo zapisany!")




