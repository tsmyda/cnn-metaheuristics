#let project_report(
  title: "Metaheurystyczne strojenie hiperparametrów CNN",
  subtitle: "",
  authors: (),
  stage: 1,
  date: none,
  body
) = {
  set document(title: title, author: authors)
  set text(font: "Linux Libertine", lang: "pl", size: 11pt)
  set heading(numbering: "1.")

  set par(justify: true)
  
  set page(
    margin: (top: 2cm, bottom: 2cm, x: 2.5cm),
    numbering: "1",
    header: context {
      if counter(page).get().first() > 1 {
        align(right)[#text(size: 8pt, fill: gray)[#title - #authors.join(", ")]]
      }
    }
  )

  align(center)[
    #grid(
      columns: (1fr, 1fr),
      align(left)[#text(size: 10pt)[AGH University of Science and Technology \ Kraków, Poland]],
      align(right)[#text(size: 10pt)[Wydział Informatyki]]
    )
    #v(1em)
    #text(size: 18pt, weight: "bold")[#title] \
    #v(0.4em)
    #text(size: 13pt, style: "italic")[#subtitle] \
    #v(0.8em)
    #text(size: 11pt)[
      *Autorzy:* #authors.join(", ")
    ]
    #v(0.4em)
    #if date != none [#date] else [#datetime.today().display()]
    #v(0.2em)
    #line(length: 100%, stroke: 0.5pt)
  ]

  body

  v(2em)
  line(length: 100%, stroke: 0.5pt)
  v(0.5em)
  bibliography("bibliography.bib", title: "Bibliografia", style: "ieee")
}

#show: project_report.with(
  title: "Metaheurystyczne strojenie hiperparametrów CNN",
  subtitle: "Porównanie GA, PSO, ACO i Harmony Search z wyszukiwaniem losowym i ręcznym 
    na zbiorach FashionMNIST, CIFAR-10 i CIFAR-100",
  authors: ("Jakub Grześ", "Tomasz Smyda"),
  stage: 3,
)

// ============ STRESZCZENIE ============

#align(center)[
  #box(width: 85%)[
    #align(left)[
      *Streszczenie.* Praca porównuje cztery metaheurystyki -- algorytm genetyczny (GA),
      optymalizację rojem cząstek (PSO), optymalizację kolonią mrówek (ACO) oraz
      wyszukiwanie harmoniczne (Harmony Search) -- z bazowymi metodami doboru
      hiperparametrów (wyszukiwanie ręczne i losowe) w zadaniu strojenia
      konwolucyjnej sieci neuronowej. Eksperymenty przeprowadzono na trzech standardowych zbiorach obrazów
      (FashionMNIST, CIFAR-10, CIFAR-100). Metody automatyczne porównano przy budżecie
        20 ewaluacji konfiguracji, gdzie każda ewaluacja obejmowała 10 epok treningu.
      Manual search potraktowano jako ekspercki punkt odniesienia złożony z 5 ręcznie
      dobranych konfiguracji. Następnie najlepszą konfigurację każdej metody
      dotrenowano przez 20 epok.
      
      Wyniki wskazują, że po 20-epokowym dotrenowaniu najlepszą dokładność testową
      uzyskało manual search na CIFAR-10 (0.8482) i FashionMNIST (0.9383), natomiast
      GA najlepiej poradził sobie na CIFAR-100 (0.5189). Jednocześnie ręczne strojenie
      pozostało bardzo silnym punktem odniesienia przy małym budżecie eksperymentalnym,
      a PSO w przyjętej implementacji uzyskało słabsze wyniki, co sugeruje, że jego
      klasyczna, ciągła postać może być niedopasowana do mieszanej przestrzeni
      hiperparametrów zawierającej wiele zmiennych dyskretnych i kategorycznych.
    ]
  ]
]

#v(0.8cm)

// ============ WPROWADZENIE ============

= Wprowadzenie

Konwolucyjne sieci neuronowe (CNN) są obecnie dominującą rodziną modeli w zadaniach
klasyfikacji obrazów. Ich skuteczność silnie zależy od doboru hiperparametrów:
tempa uczenia, wielkości batcha, liczby filtrów konwolucyjnych w kolejnych blokach,
rozmiaru jądra splotu, prawdopodobieństwa dropoutu, liczby warstw w bloku, a także
wyboru optymalizatora oraz regularyzacji. 

Klasyczne przeszukiwanie siatkowe szybko staje się w takim problemie niepraktyczne,
ponieważ liczba możliwych kombinacji rośnie bardzo szybko wraz z liczbą hiperparametrów.
Z tego powodu w literaturze stosuje się metody heurystyczne i metaheurystyczne,
które próbują znaleźć dobrą konfigurację bez konieczności sprawdzania całej przestrzeni
przeszukiwań @optimization_cnn.

Celem niniejszego raportu jest empiryczne porównanie wybranych metod strojenia
hiperparametrów CNN: algorytmu genetycznego (GA), optymalizacji rojem cząstek (PSO),
optymalizacji kolonią mrówek (ACO), wyszukiwania harmonicznego (Harmony Search),
wyszukiwania losowego oraz ręcznie dobranych konfiguracji referencyjnych. Wszystkie
metody automatyczne testowano w tej samej przestrzeni hiperparametrów i przy tym samym
budżecie 20 ewaluacji. Manual search potraktowano jako ekspercki punkt odniesienia,
a nie jako metodę o identycznym budżecie.

Kluczowe pytania badawcze są następujące:

#set list(indent: 0.6em)
- Czy którakolwiek z metaheurystyk daje systematyczny zysk jakościowy względem random search / manual search?
- Jak kształtuje się relacja jakości do czasu przeszukiwania (time-to-best)?
- Czy konfiguracja wybrana w fazie szybkiego przeszukiwania (10 epok) przekłada się na jakość po pełniejszym dotrenowaniu (20 epok)?

// ============ METODYKA ============

= Metodyka

== Przestrzeń przeszukiwań

Eksperyment przeprowadzono w 12-wymiarowej przestrzeni hiperparametrów,
obejmującej zarówno parametry architektury CNN, jak i parametry procesu uczenia.
Do parametrów architektonicznych należą: liczba bloków konwolucyjnych, liczba
filtrów w kolejnych blokach, rozmiar jądra splotu, rozmiar warstwy gęstej oraz
użycie batch normalization. Do parametrów treningowych należą: tempo uczenia,
wielkość batcha, prawdopodobieństwo dropoutu, optymalizator oraz współczynnik
weight decay. Szczegóły przestrzeni zestawiono w tabeli @tab:space.

#figure(
  caption: [Przestrzeń hiperparametrów użyta w eksperymencie.],
  table(
    columns: (auto, auto, auto),
    align: (left, center, left),
    stroke: 0.5pt,
    table.header([*Hiperparametr*], [*Typ*], [*Dziedzina*]),
    [`learning_rate`], [log-float], [$[10^(-4), 10^(-2)]$],
    [`batch_size`], [kategoryczny], [{32, 64, 128, 256}],
    [`num_blocks`], [całkowity], [{1, 2, 3}],
    [`filters_1`], [kategoryczny], [{16, 32, 64}],
    [`filters_2`], [kategoryczny], [{32, 64, 128}],
    [`filters_3`], [kategoryczny], [{64, 128, 256}],
    [`kernel_size`], [kategoryczny], [{3, 5}],
    [`dropout`], [float], [$[0.0, 0.5]$],
    [`dense_units`], [kategoryczny], [{64, 128, 256}],
    [`optimizer`], [kategoryczny], [{adam, sgd, adamw}],
    [`weight_decay`], [log-float], [$[10^(-6), 10^(-3)]$],
    [`use_batch_norm`], [binarny], [{0, 1}],
  )
) <tab:space>

Wspólnym modelem bazowym była konfigurowalna sieć `TunableCNN`. Model składa się
z `num_blocks` bloków konwolucyjnych. Każdy blok zawiera dwie warstwy konwolucyjne,
opcjonalną normalizację batchową po każdej konwolucji, aktywacje ReLU, operację
max pooling oraz dropout. Po części konwolucyjnej następuje spłaszczenie reprezentacji
i klasyfikator z jedną ukrytą warstwą gęstą o rozmiarze `dense_units`.

Ten sam szablon architektury był używany zarówno w fazie krótkiego przeszukiwania,
jak i podczas późniejszego dotrenowania najlepszych konfiguracji. Dzięki temu różnice
między metodami wynikają z wyboru hiperparametrów, a nie ze zmiany samego modelu.

== Algorytmy przeszukiwania

Zaimplementowano i porównano sześć metod:

- *Manual search* — 5 ręcznie dobranych konfiguracji referencyjnych. Nie jest to metoda
  o takim samym budżecie jak pozostałe algorytmy, lecz baseline do oceny pozostałych metod.
- *Random search* — 20 niezależnych losowań z przestrzeni hiperparametrów.
  Metoda ta stanowi prosty punkt odniesienia, pokazujący, ile można osiągnąć
  bez wykorzystywania informacji z poprzednich ewaluacji.
- *GA* — algorytm genetyczny z populacją 5 osobników przez 4 generacje. Zastosowano
  selekcję turniejową o rozmiarze 3, krzyżowanie wartości parametrów między rodzicami,
  mutację z prawdopodobieństwem 0.20 oraz elityzm o rozmiarze 1.
- *PSO* — optymalizacja rojem 5 cząstek przez 4 iteracje. Użyto parametrów
  $w = 0.7$ oraz $c_1 = c_2 = 1.5$. Ponieważ część hiperparametrów ma charakter
  kategoryczny lub dyskretny, pozycje cząstek były naprawiane przez ograniczanie
  wartości do dozwolonych zakresów i zaokrąglanie do najbliższych poprawnych wartości.
- *ACO* — optymalizacja kolonią 5 mrówek przez 4 iteracje. Rozkład feromonów prowadzono
  osobno dla poszczególnych wymiarów przestrzeni. Po każdej iteracji stosowano parowanie
  feromonów z parametrem $rho = 0.2$ oraz wzmacnianie najlepszych konfiguracji.
- *Harmony Search* — metoda z pamięcią harmonii o rozmiarze 5 oraz 15 kolejnymi
  improwizacjami. Zastosowano parametry HMCR $= 0.9$ oraz PAR $= 0.3$.

Dla metod automatycznych zastosowano jednakowy budżet 20 ewaluacji konfiguracji.
W przypadku GA i PSO odpowiada to schematowi $5 times 4$, dla ACO liczbie
$5$ mrówek przez $4$ iteracje, a dla Harmony Search: $5$ konfiguracjom inicjalnym
w pamięci harmonii oraz $15$ improwizacjom. Manual search ma mniejszy budżet
i dlatego jest interpretowany osobno jako punkt odniesienia, a nie jako metoda
bezpośrednio równoważna pod względem liczby ewaluacji.

== Procedura ewaluacji

Po zakończeniu przeszukiwania dla każdej pary (zbiór danych, metoda) wybierano
konfigurację o najwyższej wartości `val_accuracy`. Następnie wybraną konfigurację
trenowano ponownie przez 20 epok, zapisując najlepszy stan modelu względem zbioru
walidacyjnego. Dopiero dokładność na zbiorze testowym po tym etapie traktowano jako
wynik końcowy danej metody.

Wszystkie eksperymenty uruchamiano z ustalonym ziarnem losowym, aby ograniczyć
wpływ losowości na porównanie metod. Należy jednak podkreślić, że pojedyncze ziarno
nie wystarcza do oceny istotności statystycznej różnic między metodami; dlatego
wyniki należy traktować jako porównanie w jednym kontrolowanym scenariuszu
eksperymentalnym.

== Zbiory danych i sprzęt

Eksperymenty przeprowadzono na trzech standardowych benchmarkach klasyfikacji obrazów:
*FashionMNIST* @fashion_mnist, *CIFAR-10* @cifar oraz *CIFAR-100* @cifar. FashionMNIST składa się z obrazów
w skali szarości o rozmiarze 28$times$28 i 10 klas, natomiast CIFAR-10 i CIFAR-100
zawierają kolorowe obrazy RGB o rozmiarze 32$times$32. Zbiory CIFAR różnią się
liczbą klas: CIFAR-10 obejmuje 10 klas, a CIFAR-100 obejmuje 100 klas.

Raport został przygotowany na podstawie wyników wygenerowanych w notatniku `experiments_full_retraining.ipynb`, a tabele i wykresy zapisano w katalogu `notebooks/results/`. Kod źródłowy implementacji metod przeszukiwania, modelu CNN, procedury ewaluacji oraz skryptów pomocniczych udostępniono w repozytorium projektu @cnn_metaheuristics_repo.

// ============ WYNIKI ============

= Wyniki

== Najlepsza dokładność walidacyjna w fazie przeszukiwania

Tabela @tab:cross-val oraz rysunek @fig:cross podsumowują najlepszą wartość
`val_accuracy` osiągniętą przez każdą metodę w fazie przeszukiwania. Każda
ewaluacja konfiguracji obejmowała krótki, 10-epokowy trening, dlatego wyniki
z tej części należy interpretować jako miarę jakości konfiguracji w warunkach
ograniczonego budżetu treningowego.

#figure(
  caption: [Najlepsza `val_accuracy` uzyskana w fazie przeszukiwania (10 epok / ewaluację).],
  table(
    columns: (auto, auto, auto, auto),
    align: (left, center, center, center),
    stroke: 0.5pt,
    table.header([*Metoda*], [*CIFAR-10*], [*CIFAR-100*], [*FashionMNIST*]),
    [ACO], [0.7922], [0.3652], [0.9262],
    [GA], [*0.8178*], [*0.4432*], [0.9322],
    [Harmony Search], [0.7740], [0.4034], [0.9342],
    [Manual search], [0.7980], [0.4288], [*0.9365*],
    [PSO], [0.7984], [0.3138], [0.9197],
    [Random search], [0.8166], [0.4094], [0.9328],
  )
) <tab:cross-val>

#figure(
  image("figures/cross_dataset_best_val_accuracy.png", width: 95%),
  caption: [Najlepsza `val_accuracy` per metoda na każdym zbiorze w fazie przeszukiwania.],
) <fig:cross>

Na zbiorze CIFAR-10 najwyższą dokładność walidacyjną uzyskał GA (0.8178), bardzo
nieznacznie wyprzedzając random search (0.8166) i manual search (0.7980).
Na zbiorze CIFAR-100 ponownie najlepszy był GA (0.4432), a kolejne miejsca zajęły
manual search (0.4288) i random search (0.4094). Na FashionMNIST najwyższą
`val_accuracy` osiągnęło manual search (0.9365), przy czym Harmony Search,
random search i GA również uplasowały się bardzo blisko siebie w zakresie
0.932--0.934. Sugeruje to, że dla tego zbioru badana architektura stosunkowo
łatwo osiąga wysoki poziom jakości, a wybór metody strojenia ma mniejsze znaczenie
niż w przypadku trudniejszych zbiorów CIFAR.

== Krzywe best-so-far

Wykresy @fig:bsf-cifar10, @fig:bsf-cifar100 i @fig:bsf-fashion pokazują przebieg
najlepszej dotychczas znalezionej wartości `val_accuracy` w funkcji numeru
ewaluacji. Taka prezentacja pozwala ocenić nie tylko końcową jakość, ale również
tempo znajdowania dobrych konfiguracji.

#figure(
  image("figures/best_so_far_CIFAR10.png", width: 90%),
  caption: [Best-so-far `val_accuracy` na CIFAR-10. GA i ACO wyraźnie
  przewyższają pozostałe metody po ok. 10 ewaluacjach.],
) <fig:bsf-cifar10>

#figure(
  image("figures/best_so_far_CIFAR100.png", width: 90%),
  caption: [Best-so-far `val_accuracy` na CIFAR-100. ACO osiąga
  maksimum już w 3. ewaluacji; GA dogania w końcowej fazie.],
) <fig:bsf-cifar100>

#figure(
  image("figures/best_so_far_FashionMNIST.png", width: 90%),
  caption: [Best-so-far `val_accuracy` na FashionMNIST. Wszystkie
  metody zbiegają do zbliżonego poziomu.],
) <fig:bsf-fashion>

Na CIFAR-10 random search i manual search szybko osiągają poziom zbliżony do
końcowego optimum, ale GA ostatecznie uzyskuje minimalnie lepszy wynik. Na
CIFAR-100 GA poprawia się najbardziej w końcowych ewaluacjach i wyraźnie
wyprzedza pozostałe metody. Na FashionMNIST wszystkie metody dochodzą do bardzo
podobnego poziomu, przy czym manual search pozostaje najlepszym punktem odniesienia.
PSO uzyskało słabszy przebieg best-so-far. Może to wynikać z faktu, że klasyczna
reprezentacja PSO operuje na pozycjach ciągłych, podczas gdy badana przestrzeń
zawiera wiele zmiennych dyskretnych i kategorycznych, które następnie muszą być
naprawiane do najbliższych dozwolonych wartości.

== Czas do osiągnięcia najlepszego wyniku

Maksymalna wartość `val_accuracy` nie opisuje w pełni zachowania metody. Istotne jest również to, jak szybko dana metoda znajduje swoją najlepszą konfigurację.

@fig:ttb-all pokazuje czas potrzebny do osiągnięcia najlepszej wartości `val_accuracy` przez każdą metodę na trzech analizowanych zbiorach danych.

#figure(
  image("figures/time_to_best_all_datasets.png"),
  caption: [Czas do osiągnięcia najlepszej `val_accuracy` dla każdej metody i zbioru danych.],
) <fig:ttb-all>

Warto interpretować tę metrykę razem z maksymalną osiągniętą dokładnością walidacyjną. Krótki czas do najlepszego wyniku nie musi oznaczać, że dana metoda znalazła najlepszą konfigurację globalnie — może jedynie oznaczać, że szybko osiągnęła swój własny najlepszy wynik w ramach danego uruchomienia. Na CIFAR-10 manual search osiąga najlepszy wynik po 4 ewaluacjach, random search po 7, a GA dopiero po 18. Na CIFAR-100 manual search także znajduje najlepszą konfigurację bardzo wcześnie, podczas gdy GA potrzebuje znacznie większej części budżetu. Na FashionMNIST najszybsze są manual search i PSO, ale najlepszy wynik nadal należy do manual search. Dla tej metryki wynik random jest zupełnie nieprzewidywalny, ponieważ nie ma mechanizmu uczenia się z wcześniejszych ewaluacji.

== Analiza zależności hiperparametrów i metryk

Dla każdej pary (zbiór danych, metoda) obliczono macierze korelacji Pearsona między hiperparametrami a metrykami uzyskanymi w fazie przeszukiwania: `val_accuracy`, `test_accuracy`, `time_sec` oraz `num_params`. Analiza ta ma charakter eksploracyjny i służy wskazaniu potencjalnych zależności w wynikach eksperymentu. Nie należy jej interpretować jako formalnego dowodu wpływu pojedynczych hiperparametrów na jakość modelu, ponieważ liczba ocenionych konfiguracji jest ograniczona, a próbki generowane przez metaheurystyki nie są niezależnym losowaniem z całej przestrzeni.

Ze względu na objętość raportu w głównej części przedstawiono jedną reprezentatywną mapę korelacji — dla metody GA na zbiorze CIFAR-10. Pełny zestaw map korelacji znajduje się w katalogu `notebooks/results/figures/hyperparam_correlations_*` repozytorium projektu @cnn_metaheuristics_repo.

#figure(
  image("figures/hyperparam_correlations_CIFAR10/hyperparam_metric_correlation_ga.png", width: 80%),
  caption: [Korelacje hiperparametrów z metrykami dla GA na zbiorze CIFAR-10.],
) <fig:corr-ga-c10>

W przedstawionym przykładzie zależności między hiperparametrami a jakością modelu
są umiarkowane. Najsilniejszy dodatni związek z `val_accuracy` ma liczba filtrów
w pierwszym bloku (`filters_1`, około $0.22$), natomiast liczba bloków i rozmiar
batcha wykazują umiarkowanie ujemną korelację z jakością. Pozostałe korelacje z
dokładnością walidacyjną są słabsze, co sugeruje, że w tej próbie jakość modelu nie
była kontrolowana przez pojedynczy hiperparametr, lecz przez kombinację kilku ustawień.

== Jakość końcowa po dotrenowaniu

W fazie przeszukiwania każda konfiguracja była oceniana po krótkim, 10-epokowym
treningu. Taka procedura pozwala ograniczyć koszt eksperymentu, ale nie daje jeszcze
pełnej informacji o jakości konfiguracji po dłuższym uczeniu. Dlatego dla każdej pary
(zbiór danych, metoda) wybrano konfigurację o najwyższej wartości `val_accuracy`
w fazie przeszukiwania, a następnie dotrenowano ją przez 20 epok. Wynik testowy
uzyskany po tym etapie traktujemy jako końcową ocenę danej metody.

Tabela @tab:final oraz rysunek @fig:final przedstawiają końcową wartość
`test_accuracy` po 20-epokowym dotrenowaniu najlepszej konfiguracji każdej metody.

#figure(
  caption: [Końcowa `test_accuracy` po 20-epokowym dotrenowaniu najlepszej konfiguracji.],
  table(
    columns: (auto, auto, auto, auto),
    align: (left, center, center, center),
    stroke: 0.5pt,
    table.header([*Metoda*], [*CIFAR-10*], [*CIFAR-100*], [*FashionMNIST*]),
    [ACO], [0.8109], [0.4308], [0.9202],
    [GA], [0.8406], [*0.5189*], [0.9331],
    [Harmony Search], [0.8161], [0.4680], [0.9305],
    [Manual search], [*0.8482*], [0.4666], [*0.9383*],
    [PSO], [0.7373], [0.3720], [0.9180],
    [Random search], [0.8449], [0.4680], [0.9313],
  )
) <tab:final>

#figure(
  image("figures/final_test_accuracy_pivot.png", width: 95%),
  caption: [Końcowa `test_accuracy` po retreningu — metoda $times$ zbiór.],
) <fig:final>

W przeprowadzonym eksperymencie manual search uzyskał najwyższą końcową
`test_accuracy` na CIFAR-10 (0.8482) i FashionMNIST (0.9383), natomiast GA
zwyciężył na CIFAR-100 (0.5189). Na CIFAR-10 random search był bardzo blisko GA
(0.8449 wobec 0.8406), a manual search wyprzedził pozostałe metody o około
0.3--0.8 p.p. Na CIFAR-100 GA uzyskał wynik o około 5.2 p.p. wyższy od manual
search, około 8.8 p.p. wyższy od ACO oraz około 5.1 p.p. wyższy od random search.

Manual search pozostaje jednak silną linią bazową. Na CIFAR-10 osiąga drugi wynik
po random search i przed ACO, natomiast na CIFAR-100 jest drugą najlepszą metodą.
Wskazuje to, że przy małym budżecie ewaluacji ręcznie dobrane konfiguracje mogą być
konkurencyjne wobec metod automatycznych, zwłaszcza jeśli zostały dobrane na podstawie
intuicji dotyczącej architektury CNN.

Krzywe dotrenowania na rysunku @fig:retrain pozwalają ocenić, czy konfiguracje wybrane
po krótkim treningu zachowują przewagę również w dłuższym uczeniu. W badanym ustawieniu
ranking metod po dotrenowaniu jest tylko częściowo zgodny z wynikami fazy przeszukiwania.
GA utrzymuje przewagę na CIFAR-100, natomiast na CIFAR-10 i FashionMNIST najlepsze
końcowe wyniki osiąga manual search. Oznacza to, że 10-epokowa ewaluacja była
użytecznym, choć przybliżonym, kryterium wyboru konfiguracji do dalszego treningu.

#figure(
  image("figures/final_retraining_val_curves.png", width: 100%),
  caption: [Krzywe `val_accuracy` w trakcie 20-epokowego dotrenowania.],
) <fig:retrain>

== Zmienność jakości ocenianych konfiguracji

Dotychczasowe zestawienia koncentrowały się głównie na najlepszych konfiguracjach
znalezionych przez poszczególne metody. Dodatkowo przeanalizowano rozkład jakości
wszystkich konfiguracji ocenionych w fazie przeszukiwania. Tabela @tab:summary-all
pokazuje średnią oraz odchylenie standardowe `val_accuracy` i `test_accuracy`
dla każdej metody na trzech zbiorach danych.

Wartości odchylenia standardowego nie opisują tutaj powtarzalności metod między
niezależnymi uruchomieniami. Pokazują jedynie, jak bardzo różniły się wyniki
konfiguracji ocenionych w ramach jednego przebiegu eksperymentu.

#figure(
  caption: [Średnia i odchylenie standardowe jakości konfiguracji ocenionych w fazie przeszukiwania. Skróty: C10 — CIFAR-10, C100 — CIFAR-100, FMNIST — FashionMNIST.],
  {
    set text(size: 10pt)

    table(
      columns: (auto, auto, auto, auto, auto, auto, auto),
      align: (left, center, center, center, center, center, center),
      stroke: 0.5pt,
      inset: (x: 4pt, y: 4pt),

      table.header(
        [*Metoda*],
        [*C10 val*], [*C10 test*],
        [*C100 val*], [*C100 test*],
        [*FMNIST val*], [*FMNIST test*],
      ),

      [ACO],
      [$0.612 plus.minus 0.171$], [$0.611 plus.minus 0.169$],
      [$0.234 plus.minus 0.115$], [$0.233 plus.minus 0.117$],
      [$0.859 plus.minus 0.180$], [$0.853 plus.minus 0.180$],

      [GA],
      [$0.711 plus.minus 0.107$], [$0.711 plus.minus 0.105$],
      [$0.320 plus.minus 0.129$], [$0.321 plus.minus 0.131$],
      [$0.923 plus.minus 0.012$], [$0.918 plus.minus 0.012$],

      [Harmony S.],
      [$0.537 plus.minus 0.207$], [$0.536 plus.minus 0.208$],
      [$0.200 plus.minus 0.106$], [$0.197 plus.minus 0.107$],
      [$0.867 plus.minus 0.182$], [$0.862 plus.minus 0.181$],

      [Manual],
      [$0.761 plus.minus 0.045$], [$0.758 plus.minus 0.043$],
      [$0.325 plus.minus 0.096$], [$0.323 plus.minus 0.102$],
      [$0.932 plus.minus 0.003$], [$0.925 plus.minus 0.002$],

      [PSO],
      [$0.375 plus.minus 0.276$], [$0.373 plus.minus 0.278$],
      [$0.065 plus.minus 0.084$], [$0.064 plus.minus 0.083$],
      [$0.851 plus.minus 0.190$], [$0.846 plus.minus 0.189$],

      [Random],
      [$0.604 plus.minus 0.199$], [$0.603 plus.minus 0.199$],
      [$0.220 plus.minus 0.123$], [$0.221 plus.minus 0.125$],
      [$0.907 plus.minus 0.029$], [$0.902 plus.minus 0.029$],
    )
  }
) <tab:summary-all>

Na CIFAR-10 najwyższą średnią jakość ocenianych konfiguracji uzyskał manual search,
przy jednocześnie bardzo małym rozrzucie wyników. Podobny obraz widać na CIFAR-100,
gdzie manual search ma najwyższe średnie `val_accuracy` i `test_accuracy`, natomiast
PSO osiąga wyraźnie niższą średnią i duży względny rozrzut. Sugeruje to, że w tym
eksperymencie ręcznie dobrane konfiguracje były szczególnie stabilną linią bazową,
natomiast PSO często odwiedzało słabe konfiguracje.

Na FashionMNIST różnice między metodami są mniejsze, ponieważ większość konfiguracji
osiąga stosunkowo wysoką jakość. Najniższy rozrzut ma manual search, co wynika
z faktu, że składa się z kilku ręcznie dobranych, sensownych konfiguracji bazowych.
Nie należy jednak interpretować tego jako dowodu ogólnej stabilności manual search,
ponieważ metoda ta nie eksploruje przestrzeni w taki sam sposób jak algorytmy
automatyczne.

== Najlepsze konfiguracje

Tabela @tab:best-all zestawia najlepsze konfiguracje znalezione przez poszczególne
metody na trzech zbiorach danych. Za najlepszą konfigurację uznano tę, która w fazie
przeszukiwania uzyskała najwyższą wartość `val_accuracy` dla danej pary
(zbiór danych, metoda). W tabeli pokazano najważniejsze hiperparametry wpływające
na architekturę i trening; pełne konfiguracje, obejmujące wszystkie parametry
z przestrzeni przeszukiwań, udostępniono w repozytorium projektu jako pliki
`best_configs_*.csv` @cnn_metaheuristics_repo.

#figure(
  caption: [Najlepsze konfiguracje znalezione przez metody na trzech zbiorach danych.],
  {
    set text(size: 8pt)

    table(
      columns: (auto, auto, auto, auto, auto, auto, auto, auto, auto),
      align: (left, left, center, center, center, center, center, center, center),
      stroke: 0.5pt,
      inset: (x: 3pt, y: 3pt),

      table.header(
        [*Zbiór*], [*Metoda*], [*val*], [*test*],
        [*lr*], [*bs*], [*blocks*], [*dropout*], [*dense*],
      ),

      [CIFAR-10], [ACO], [$0.7922$], [$0.7899$], [$1.0 times 10^(-3)$], [64], [3], [$0.10$], [64],
      [CIFAR-10], [GA], [$0.8178$], [$0.8095$], [$1.56 times 10^(-4)$], [64], [3], [$0.27$], [128],
      [CIFAR-10], [Manual], [$0.7980$], [$0.7947$], [$2.0 times 10^(-4)$], [128], [3], [$0.40$], [256],
      [CIFAR-10], [PSO], [$0.7984$], [$0.7834$], [$1.0 times 10^(-2)$], [128], [2], [$0.34$], [256],
      [CIFAR-10], [HS], [$0.7740$], [$0.7775$], [$1.56 times 10^(-4)$], [128], [3], [$0.27$], [128],
      [CIFAR-10], [Random], [$0.8166$], [$0.8079$], [$8.40 times 10^(-4)$], [128], [3], [$0.42$], [64],

      [CIFAR-100], [GA], [$0.4432$], [$0.4501$], [$1.56 times 10^(-4)$], [64], [3], [$0.27$], [128],
      [CIFAR-100], [Manual], [$0.4288$], [$0.4328$], [$5.0 times 10^(-4)$], [128], [2], [$0.30$], [256],
      [CIFAR-100], [Random], [$0.4094$], [$0.4147$], [$1.56 times 10^(-4)$], [128], [3], [$0.27$], [128],
      [CIFAR-100], [HS], [$0.4034$], [$0.4123$], [$1.56 times 10^(-4)$], [128], [3], [$0.27$], [128],
      [CIFAR-100], [ACO], [$0.3652$], [$0.3732$], [$1.0 times 10^(-4)$], [64], [3], [$0.10$], [64],
      [CIFAR-100], [PSO], [$0.3138$], [$0.3075$], [$1.40 times 10^(-3)$], [32], [2], [$0.09$], [128],

      [FashionMNIST], [Manual], [$0.9365$], [$0.9251$], [$2.0 times 10^(-4)$], [128], [3], [$0.40$], [256],
      [FashionMNIST], [HS], [$0.9342$], [$0.9280$], [$1.43 times 10^(-3)$], [32], [3], [$0.03$], [256],
      [FashionMNIST], [Random], [$0.9328$], [$0.9303$], [$6.34 times 10^(-4)$], [32], [1], [$0.33$], [128],
      [FashionMNIST], [GA], [$0.9322$], [$0.9276$], [$1.52 times 10^(-3)$], [256], [3], [$0.28$], [256],
      [FashionMNIST], [ACO], [$0.9262$], [$0.9183$], [$1.0 times 10^(-2)$], [128], [3], [$0.25$], [256],
      [FashionMNIST], [PSO], [$0.9197$], [$0.9152$], [$1.0 times 10^(-2)$], [256], [1], [$0.47$], [256],
    )
  }
) <tab:best-all>

Warto zauważyć, że najlepsze konfiguracje nie tworzą jednego uniwersalnego wzorca
dla wszystkich zbiorów. Dla CIFAR-10 i CIFAR-100 korzystne okazały się konfiguracje
z umiarkowaną liczbą bloków i niskim albo średnim dropoutem, natomiast na
FashionMNIST manual search i Harmony Search wykorzystały bardziej pojemne układy z
większą liczbą filtrów i wyższym dropoutem. Oznacza to, że skuteczność konfiguracji
jest zależna zarówno od zbioru danych, jak i od sposobu eksploracji przestrzeni
przez daną metodę.

// ============ DYSKUSJA ============

= Dyskusja i wnioski

W przeprowadzonym eksperymencie liderzy różnili się między zbiorami danych: GA
uzyskał najwyższą końcową `test_accuracy` na CIFAR-100, natomiast manual search
zwyciężył na CIFAR-10 i FashionMNIST. Wynik ten sugeruje, że w badanej przestrzeni
hiperparametrów mechanizmy selekcji, elityzmu i mutacji mogą dobrze kierować
populację, ale ich skuteczność zależy od charakterystyki zbioru i budżetu
ewaluacji. Nie należy traktować tego jako bezwarunkowej przewagi GA nad pozostałymi
metodami, ponieważ eksperyment wykonano przy ograniczonym budżecie ewaluacji
i dla pojedynczego ziarna losowego.

Istotnym wynikiem jest również bardzo dobra pozycja manual search. Ręcznie dobrane
konfiguracje okazały się szczególnie konkurencyjne na CIFAR-10 i FashionMNIST,
gdzie po dotrenowaniu uzyskały najlepsze wyniki. Na CIFAR-100 manual search pozostał
drugą najlepszą metodą po GA. Pokazuje to, że przy małym budżecie kilka sensownie
dobranych konfiguracji eksperckich może być trudną do pobicia linią bazową.

PSO uzyskało słabsze wyniki niż GA, ACO i często również random search. Możliwym
wyjaśnieniem jest niedopasowanie klasycznej, ciągłej reprezentacji PSO do mieszanej
przestrzeni hiperparametrów, w której wiele wymiarów ma charakter dyskretny lub
kategoryczny. Wartości cząstek musiały być naprawiane przez zaokrąglanie lub mapowanie
do najbliższych dopuszczalnych wartości, co mogło ograniczać użyteczność mechanizmu
prędkości.

Harmony Search osiągał wyniki pośrednie. Metoda była lepsza od PSO w części
eksperymentów, ale na trudniejszych zbiorach CIFAR ustępowała GA i ACO. Jedną
z możliwych przyczyn jest wysoka wartość parametru HMCR, która sprzyja wybieraniu
wartości z pamięci harmonii. Przy małej pamięci i ograniczonym budżecie może to
prowadzić do zbyt szybkiej eksploatacji wcześniej znalezionych konfiguracji kosztem
dalszej eksploracji.

Wyniki po 20-epokowym dotrenowaniu wskazują, że krótka, 10-epokowa ewaluacja była
użytecznym kryterium wyboru konfiguracji do dalszego treningu, ale nie przesądzała
o ostatecznym rankingu metod. W części przypadków lepszy wynik po retreningu miały
konfiguracje, które nie były absolutnie najlepsze w fazie przeszukiwania. Nie oznacza
to jednak, że `val_accuracy` po 10 epokach jest pełnym substytutem dłuższego
treningu. Jest to raczej kosztowo tańszy wskaźnik jakości konfiguracji.

Analiza korelacji hiperparametrów z metrykami miała charakter pomocniczy. Najłatwiej
interpretowalne były zależności związane z kosztem modelu, np. dodatnia korelacja
liczby bloków z czasem treningu. Zależności między pojedynczymi hiperparametrami
a jakością były słabsze i zależne od metody oraz zbioru danych.

== Ograniczenia i możliwe dalsze analizy

Najważniejszym ograniczeniem eksperymentu jest użycie pojedynczego ziarna losowego.
Wyniki metod stochastycznych mogą zależeć od inicjalizacji populacji, losowania
konfiguracji, podziału train/validation oraz inicjalizacji wag sieci. Silniejsze
wnioski wymagałyby powtórzenia eksperymentów dla kilku seedów i raportowania średnich
oraz odchyleń standardowych wyników końcowych.

Drugim ograniczeniem jest mały budżet przeszukiwania. Dwadzieścia ewaluacji to
realistyczny, ale bardzo ograniczony budżet dla 12-wymiarowej przestrzeni
hiperparametrów. W dalszych eksperymentach warto porównać metody także dla większych
budżetów, np. 40 lub 60 ewaluacji, oraz analizować krzywe best-so-far jako funkcję
liczby ewaluacji.
