#   Machine Learning Predictor

## O projekcie
**NBA Predictor** to zautomatyzowany system analityczny do przewidywania wyników meczów NBA, natomiast stworzony głównie w celach naukowych by zwiekszyć swoje umiejętnopści programowania oraz poznania nowych metod analizy danych. Projekt łączy własną implementację algorytmu Elo Rating z uczeniem maszynowym (Logistic Regression), aby oceniać prawdopodobieństwo wygranej i kategoryzować mecze pod kątem ryzyka i opłacalności.

## Droga Analityczna (Research Journey)
Projekt zaczynałem od:
1. **Faza testów:**  weryfikacji dziesiątek statystyk (m.in. eFG%, Net Rating, TOV, Zbiórki,). Testowałem je w różnych oknach czasowych (Rolling Windows z 3, 5, 10, 15 meczów) oraz na rozkładach Dom/Wyjazd (Home/Away splits).
2. **Wykrycie Szumu :** Ekonometria (p-values w bibliotece `statsmodels`) udowodniła, że w obecności potężnego systemu Elo, standardowe statystyki koszykarskie stają się szumem. System Elo z sukcesem absorbuje informacje o skuteczności rzutowej i przewadze parkietu.
3. **Redukcja wymiarowości:** Usunąłem statystyki rzutowe, pozostawiając model w najczystszej możliwej postaci, co paradoksalnie podniosło jego Accuracy oraz odporność na przeuczenie (Overfitting).

##  Zmienne w Ostatecznym Modelu 
Obecnie w głównym silniku `bet_king.py` używam tylko 4 rygorystycznie wyselekcjonowanych parametrów:
1.  **`elo_diff`**: Historyczna i długoterminowa siła zespołu.
2.  **`win_num_diff`**: Krótkoterminowa forma / Hot streak (Rolling window z 10 ostatnich meczów).
3.  **`b2b_diff`**: Wskaźnik zmęczenia (czy drużyna gra drugi mecz w ciągu 2 dni).
4.  **`Home`**: Waga własnego parkietu w kontekście starcia.

## Struktura Projektu
* `nba_elo.py` - Silnik bazowy. Oblicza rankingi Elo i generuje plik główny.
* `statystyki_10.py` - Serce badawcze projektu. Skrypt walidacyjny pokazujący pełną regresję logistyczną, p-values, ROC AUC oraz testy progów (Sensitivity Analysis).
* `bet_king.py` - Skrypt produkcyjny. Uruchamia interfejs użytkownika, posiada wbudowany Pipeline i kategoryzuje typy (Złote, Srebrne, Brązowe).
* `nba_dane.xlsx` - Surowa baza danych z wynikami meczów.
* `nba_elo_full.xlsx` - zaktualizowana baza danych z wynikami meczów i rankingiem elo.
* `research` - folder z plikami gdzie testowane, sprawdzane były różne rzeczy które pomogły ulepszyć projekt

## Krok po kroku: Jak uruchomić projekt?

## KROK1:Pobierz repozytorium i zainstaluj wymagane biblioteki:
pip install pandas scikit-learn statsmodels openpyxl
## KROK2:Na samym początku trzeba uruchomić skrypt nba_elo.py, aby wygenerować pełną baze.
## KROK3:(opcjonalinie) można uruchomić  statystyki_10.py, aby zobaczyć jak algorytm dobiera wagi i dlaczego ostateczne parametry mają p-value < 0.05.
## KROK4: uruchomić bet_king.py:
Instrukcja obsługi w konsoli:

Program zapyta o gospodarza. Wpisz oficjalny 3-literowy skrót (np. SAS dla San Antonio Spurs, LAL, NYK,).

Wpisz 3-literowy skrót gościa (np. BOS, GSW).

Program zapyta o datę meczu w formacie RRRR-MM-DD (np. 2026-04-06). Jeśli mecz odbywa się dzisiaj, po prostu wciśnij ENTER.

System automatycznie sprawdzi zmęczenie (B2B) względem podanej daty, policzy model i wyrzuci ostatecznego faworyta.
