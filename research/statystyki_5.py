import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

# =========================
# 1. PRZYGOTOWANIE DANYCH
# =========================
# Wczytujemy plik wygenerowany przez nasz system Elo
dane = pd.read_excel("nba_elo_full.xlsx")
    
dane.columns = dane.columns.str.strip()
dane = dane.dropna(subset=["Team", "Opp", "Date"])

dane["Team"] = dane["Team"].astype(str)
dane["Opp"] = dane["Opp"].astype(str)
dane["Date"] = pd.to_datetime(dane["Date"])
dane = dane.sort_values(["Team", "Season", "Date"])

dane["win_num"] = (dane["Rslt"] == "W").astype(int)
dane["net_rating"] = dane["Pkt"] - dane["Opp_Pkt"]

# Zostawiliśmy tylko najmocniejsze statystyki "krótkoterminowe"
statystyki = ["win_num", "eFG%", "TOV", "net_rating"]

for stat in statystyki:
    nazwa_kolumny = f"{stat}_5"
    dane[nazwa_kolumny] = (
        dane.groupby(["Team", "Season"])[stat]
        .rolling(5).mean().shift(1)
        .reset_index(level=[0,1], drop=True)
    )

# =========================
# 2. ZMĘCZENIE (B2B)
# =========================
dane['days_rest'] = dane.groupby('Team')['Date'].diff().dt.days
dane['is_b2b'] = (dane['days_rest'] == 1).astype(int)

# =========================
# 3. ŁĄCZENIE W PARY (MECZE)
# =========================
dane["game_id"] = (
    dane["Date"].astype(str) + "_" +
    dane[["Team", "Opp"]].apply(lambda x: "_".join(sorted(x)), axis=1)
)

mecze = dane.merge(dane, on="game_id", suffixes=("_A", "_B"))
mecze = mecze[mecze["Team_A"] < mecze["Team_B"]]
mecze["Home"] = mecze["Home_A"]

mecze = mecze.dropna(subset=[
    "win_num_5_A", "win_num_5_B",
    "eFG%_5_A", "eFG%_5_B",
    "Elo_Pre_A", "Elo_Pre_B" # Upewniamy się, że Elo nie ma braków
])

# =========================
# 4. RÓŻNICE (DIFF)
# =========================
for stat in statystyki:
    mecze[f"{stat}_diff"] = mecze[f"{stat}_5_A"] - mecze[f"{stat}_5_B"]

mecze["b2b_diff"] = mecze["is_b2b_A"] - mecze["is_b2b_B"]
mecze["elo_diff"] = mecze["Elo_Pre_A"] - mecze["Elo_Pre_B"] # NASZA NOWA POTĘŻNA ZMIENNA

# =========================
# 5. MODEL LOGITOWY
# =========================
mecze["Y"] = mecze["win_num_A"]

# Zwróć uwagę jak elitarna jest teraz ta lista!
X = mecze[[
    "elo_diff",
    "win_num_diff",
    "eFG%_diff",
    "TOV_diff",
    "net_rating_diff",
    "b2b_diff",
    "Home"
]]
y = mecze["Y"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:,1]

print(f"🎯 Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"📊 ROC AUC: {roc_auc_score(y_test, y_prob):.4f}")

coeffs = pd.DataFrame({
    "zmienna": X.columns,
    "wspolczynnik": model.coef_[0]
})
coeffs["abs"] = abs(coeffs["wspolczynnik"])
coeffs = coeffs.sort_values("abs", ascending=False)
print("\nWspółczynniki modelu:")
print(coeffs)

# Statsmodels dla p-value
X_train_sm = sm.add_constant(X_train)
X_test_sm = sm.add_constant(X_test)

logit_model = sm.Logit(y_train, X_train_sm)
logit_results = logit_model.fit()
print("\nRAPORT EKONOMETRYCZNY:")
print(logit_results.summary())

# =========================
# 6. ANALIZA WRAŻLIWOŚCI (PROGI)
# =========================
# Zaktualizowany plan testów z włączonym Elo
test_plan = {
    "elo_diff": [0, 50, 100, 150, 200, 250],        # Różnica klas długoterminowa
    "net_rating_diff": [0, 2, 4, 6, 8, 10],      
    "eFG%_diff": [0, 0.01, 0.02, 0.03, 0.04, 0.05], 
    "win_num_diff": [0, 0.1, 0.2, 0.3, 0.4, 0.5],   
    "TOV_diff": [0, 1, 2, 3, 4]                     
}

def analyze_all_thresholds(df, plan):
    for var, thresholds in plan.items():
        print(f"\n🚀 ANALIZA DLA: {var}")
        print(f"{'Próg |abs|':<12} | {'Mecze (%)':<10} | {'Liczba meczów':<15} | {'Accuracy':<10}")
        print("-" * 60)
        
        total = len(df)
        for t in thresholds:
            sub = df[abs(df[var]) >= t].copy()
            if len(sub) > 0:
                if var in ["TOV_diff"]:
                    preds = (sub[var] < 0).astype(int)
                else:
                    preds = (sub[var] > 0).astype(int)
                
                acc = (preds == sub['win_num_A']).mean()
                perc = (len(sub) / total) * 100
                print(f"{t:<12} | {perc:>8.1f}% | {len(sub):>13} | {acc:>8.2%}")

analyze_all_thresholds(mecze, test_plan)

print("\n🔋 ANALIZA DLA: b2b_diff (Zmęczenie)")
for val in [0, 1]:
    sub = mecze[abs(mecze['b2b_diff']) == val]
    if val == 1:
        preds = (sub['b2b_diff'] < 0).astype(int)
        acc = (preds == sub['win_num_A']).mean()
        print(f"Gdy jedna drużyna jest bardziej zmęczona: Accuracy = {acc:.2%} (N={len(sub)})")