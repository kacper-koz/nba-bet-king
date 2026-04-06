




# 3. STAŁE PARAMETRY MODELU
BASE_ELO = 1500
K_BASE = 20

# Listy wartości, które chcemy przetestować
wartosci_home_adv = [40, 50, 60, 65, 70]
wartosci_b2b = [20, 30, 40, 50, 60]

dane = dane.sort_values("Date").reset_index(drop=True)


# 4. FUNKCJA SYMULUJĄCA CAŁY SEZON DLA DWÓCH PARAMETRÓW
def przetestuj_parametry(df, test_home_adv, test_b2b):
    elo_dict = {}
    last_season = {}
    season_game_count = {}
    
    poprawne_typy = 0
    wszystkie_mecze = 0

    for idx, row in df.iterrows():
        team = row["Team"]
        opp = row["Opp"]
        season = row["Season"]

        # --- inicjalizacja ---
        if team not in elo_dict: elo_dict[team] = BASE_ELO
        if opp not in elo_dict: elo_dict[opp] = BASE_ELO

        # --- regresja sezonowa ---
        if team in last_season and last_season[team] != season:
            elo_dict[team] = 0.75 * elo_dict[team] + 0.25 * BASE_ELO
        if opp in last_season and last_season[opp] != season:
            elo_dict[opp] = 0.75 * elo_dict[opp] + 0.25 * BASE_ELO

        last_season[team] = season
        last_season[opp] = season

        # --- progress sezonu ---
        if season not in season_game_count: season_game_count[season] = 0
        season_game_count[season] += 1
        season_progress = season_game_count[season] / 1230

        K = dynamic_k(season_progress)

        team_elo_pre = elo_dict[team]
        opp_elo_pre = elo_dict[opp]

        # --- ADJUSTMENTS (Tutaj wrzucamy oba testowane parametry) ---
        team_adj = (
            team_elo_pre 
            + (test_home_adv if row["Home"] == 1 else 0) 
            - (test_b2b if row["is_b2b_team"] == 1 else 0)
        )
        
        opp_adj = (
            opp_elo_pre 
            + (test_home_adv if row["Home"] == 0 else 0) 
            - (test_b2b if row["is_b2b_opp"] == 1 else 0)
        )

        # --- expected ---
        exp_team = expected_win(team_adj, opp_adj)
        actual = 1 if row["Rslt"] == "W" else 0

        # --- SPRAWDZENIE SKUTECZNOŚCI ---
        if (exp_team > 0.5 and actual == 1) or (exp_team < 0.5 and actual == 0):
            poprawne_typy += 1
        wszystkie_mecze += 1

        # --- margin of victory & update ---
        margin = abs(row["Pkt"] - row["Opp_Pkt"])
        elo_diff_winner = (team_elo_pre - opp_elo_pre) if actual == 1 else (opp_elo_pre - team_elo_pre)
        
        mult = margin_multiplier(margin, elo_diff_winner)
        change = K * mult * (actual - exp_team)

        elo_dict[team] += change
        elo_dict[opp] -= change

    # Zwracamy % poprawnych typowań
    skutecznosc = (poprawne_typy / wszystkie_mecze) * 100
    return skutecznosc


# 5. GŁÓWNA PĘTLA TESTUJĄCA (GRID SEARCH DLA 2 ZMIENNYCH)
print("Rozpoczynam testowanie kombinacji Home Advantage i B2B Penalty...\n")

najlepszy_wynik = 0
najlepsze_home_adv = 0
najlepsze_b2b = 0

# Zagnieżdżona pętla: przechodzi przez każdą kombinację
for h_adv in wartosci_home_adv:
    for b2b in wartosci_b2b:
        # Wywołujemy symulację z parą parametrów
        wynik = przetestuj_parametry(dane, h_adv, b2b)
        
        print(f"Test: HOME = {h_adv:2d}, B2B = {b2b:2d} | Skuteczność = {wynik:.2f}%")
        
        # Zapisujemy najlepszy wynik
        if wynik > najlepszy_wynik:
            najlepszy_wynik = wynik
            najlepsze_home_adv = h_adv
            najlepsze_b2b = b2b

print("\n" + "=" * 50)
print(f"🏆 ZWYCIĘZCA: Najlepsza kombinacja to:")
print(f"   Przewaga parkietu (HOME_ADV)  = {najlepsze_home_adv}")
print(f"   Kara za zmęczenie (B2B)       = {najlepsze_b2b}")
print(f"   Skuteczność modelu            = {najlepszy_wynik:.2f}%")
print("=" * 50)
# Zwycięskie parametry z poprzedniego testu
HOME_ADV = 65
B2B_PENALTY = 60
BASE_ELO = 1500

# Lista strategii K do przetestowania w formacie: (K_START, K_MID, K_END)
# Start: 0-30% sezonu | Mid: 30-70% sezonu | End: 70-100% sezonu
strategie_k = [
    (20, 20, 20),  # TEST 1: Brak zmian, stałe K przez cały rok
    (24, 24, 24),  # TEST 2: Stałe, ale wyższe K przez cały rok
    (30, 20, 15),  # TEST 3: Twój obecny, klasyczny model
    (40, 20, 10),  # TEST 4: Agresywny początek (szybka nauka), mocna stabilizacja na koniec
    (35, 25, 15),  # TEST 5: Szybsze tempo zmian ogólnie
    (25, 20, 15),  # TEST 6: Delikatniejszy start niż obecnie
    (30, 25, 20)   # TEST 7: Podniesione wartości dla środka i końca sezonu
]

dane = dane.sort_values("Date").reset_index(drop=True)

def przetestuj_wspolczynnik_k(df, k_start, k_mid, k_end):
    elo_dict = {}
    last_season = {}
    season_game_count = {}
    
    poprawne_typy = 0
    wszystkie_mecze = 0

    for idx, row in df.iterrows():
        team = row["Team"]
        opp = row["Opp"]
        season = row["Season"]

        if team not in elo_dict: elo_dict[team] = BASE_ELO
        if opp not in elo_dict: elo_dict[opp] = BASE_ELO

        if team in last_season and last_season[team] != season:
            elo_dict[team] = 0.75 * elo_dict[team] + 0.25 * BASE_ELO
        if opp in last_season and last_season[opp] != season:
            elo_dict[opp] = 0.75 * elo_dict[opp] + 0.25 * BASE_ELO

        last_season[team] = season
        last_season[opp] = season

        if season not in season_game_count: season_game_count[season] = 0
        season_game_count[season] += 1
        season_progress = season_game_count[season] / 1230

        # --- DYNAMICZNE K NA BAZIE TESTOWANEJ STRATEGII ---
        if season_progress < 0.3:
            K = k_start
        elif season_progress < 0.7:
            K = k_mid
        else:
            K = k_end

        team_elo_pre = elo_dict[team]
        opp_elo_pre = elo_dict[opp]

        team_adj = team_elo_pre + (HOME_ADV if row["Home"] == 1 else 0) - (B2B_PENALTY if row["is_b2b_team"] == 1 else 0)
        opp_adj = opp_elo_pre + (HOME_ADV if row["Home"] == 0 else 0) - (B2B_PENALTY if row["is_b2b_opp"] == 1 else 0)

        exp_team = expected_win(team_adj, opp_adj)
        actual = 1 if row["Rslt"] == "W" else 0

        if (exp_team > 0.5 and actual == 1) or (exp_team < 0.5 and actual == 0):
            poprawne_typy += 1
        wszystkie_mecze += 1

        margin = abs(row["Pkt"] - row["Opp_Pkt"])
        elo_diff_winner = (team_elo_pre - opp_elo_pre) if actual == 1 else (opp_elo_pre - team_elo_pre)
        
        mult = margin_multiplier(margin, elo_diff_winner)
        change = K * mult * (actual - exp_team)

        elo_dict[team] += change
        elo_dict[opp] -= change

    return (poprawne_typy / wszystkie_mecze) * 100


# GŁÓWNA PĘTLA TESTUJĄCA STRATEGIE K
print("Rozpoczynam testowanie strategii współczynnika K...\n")

najlepszy_wynik = 0
najlepsza_strategia = None

for strat in strategie_k:
    k_s, k_m, k_e = strat
    wynik = przetestuj_wspolczynnik_k(dane, k_s, k_m, k_e)
    
    print(f"Test K = ({k_s:2d}, {k_m:2d}, {k_e:2d}) | Skuteczność = {wynik:.2f}%")
    
    if wynik > najlepszy_wynik:
        najlepszy_wynik = wynik
        najlepsza_strategia = strat

print("\n" + "=" * 50)
print(f"🏆 ZWYCIĘZCA: Najlepsza strategia K to:")
print(f"   K Start (0-30%)   = {najlepsza_strategia[0]}")
print(f"   K Mid (30-70%)    = {najlepsza_strategia[1]}")
print(f"   K End (70-100%)   = {najlepsza_strategia[2]}")
print(f"   Skuteczność       = {najlepszy_wynik:.2f}%")
print("=" * 50)
