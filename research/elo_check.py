import pandas as pd

# 1. Wczytujemy plik wygenerowany przez system Elo
df_elo = pd.read_excel("nba_elo_full.xlsx")

# 2. Wyciągamy ostatni ranking Elo dla każdej drużyny w każdym sezonie
# Sortujemy po dacie, grupujemy po Sezonie i Drużynie, bierzemy ostatni wpis
rankingi_sezonowe = (
    df_elo.sort_values("Date")
    .groupby(["Season", "Team"])
    .tail(1)[["Season", "Team", "Elo_Pre"]]
)

# 3. Wyświetlamy rankingi dla każdego sezonu
for sezon in sorted(rankingi_sezonowe["Season"].unique()):
    print(f"\n🏆 RANKING ELO NA KONIEC SEZONU: {sezon}")
    top_sezon = rankingi_sezonowe[rankingi_sezonowe["Season"] == sezon].sort_values("Elo_Pre", ascending=False)
    
    print("--- TOP 5 (Liderzy) ---")
    print(top_sezon.head(5).to_string(index=False))
    
    print("\n--- BOTTOM 5 (Autsajderzy) ---")
    print(top_sezon.tail(5).to_string(index=False))
    print("-" * 30)