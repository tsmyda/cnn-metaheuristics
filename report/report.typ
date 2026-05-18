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
      konwolucyjnej sieci neuronowej. Eksperymenty przeprowadzono na trzech
      standardowych zbiorach obrazów (FashionMNIST, CIFAR-10, CIFAR-100)
      z jednakowym budżetem 20 pełnych ewaluacji (po 5 epok) oraz
      20-epokowym dotrenowaniem najlepszej konfiguracji. 
      
      Wyniki wskazują, że w badanym ustawieniu GA osiągnął najwyższą końcową dokładność
      testową na wszystkich trzech zbiorach: 0.9331 na FashionMNIST, 0.8092 na CIFAR-10
      oraz 0.4314 na CIFAR-100. Jednocześnie ręczne strojenie okazało się silnym punktem
      odniesienia przy małym budżecie eksperymentalnym, szczególnie na zbiorze CIFAR-100.
      PSO w przyjętej implementacji uzyskało słabsze wyniki, co sugeruje, że jego klasyczna,
      ciągła postać może być niedopasowana do mieszanej przestrzeni hiperparametrów
      zawierającej wiele zmiennych dyskretnych i kategorycznych.
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
- Czy konfiguracja wybrana w fazie szybkiego przeszukiwania (5 epok) przekłada się na jakość po pełniejszym dotrenowaniu (20 epok)?

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
- *Random search* — 20 niezależnych losowań z przestrzeni hiperparametrów. Metoda ta
  stanowi prosty. Metoda ta stanowi kolejny punkt odniesienia, pokazujący, ile można osiągnąć bez żadnej inteligencji w doborze hiperparametrów.
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

Każda konfiguracja jest trenowana przez 5 epok na zbiorze uczącym (90%), a jako
funkcja celu używana jest dokładność walidacyjna na pozostałych 10%. Wszystkie
uruchomienia startują z ziarna `seed=42`.

Po zakończeniu fazy przeszukiwania dla każdej pary (zbiór, metoda) wybierana jest
konfiguracja o najwyższej `val_accuracy` i *dotrenowywana* przez 20 epok z
zachowaniem najlepszej wagi walidacyjnej (`Adam`, `lr` z konfiguracji). Dopiero
wynik na zbiorze testowym po tym retreningu jest raportowany jako końcowy.

== Zbiory danych i sprzęt

Eksperymenty przeprowadzono na trzech standardowych benchmarkach:
*FashionMNIST* (10 klas, 28$times$28, odcienie szarości), *CIFAR-10* (10 klas,
32$times$32, RGB) i *CIFAR-100* (100 klas, 32$times$32, RGB). Wszystkie
obliczenia wykonane były na GPU CUDA. Raport został wygenerowany z notatnika
`experiments_full_retraining.ipynb`, a tabele i wykresy z katalogu
`notebooks/results/`.

// ============ WYNIKI ============

= Wyniki

== Najlepsza dokładność walidacyjna w fazie przeszukiwania

Tabela @tab:cross-val i @fig:cross podsumowują najlepszą `val_accuracy` osiągniętą
przez każdą metodę w fazie przeszukiwania (5-epokowe ewaluacje).

#figure(
  caption: [Najlepsza `val_accuracy` uzyskana w fazie przeszukiwania (5 epok / ewaluację).],
  table(
    columns: (auto, auto, auto, auto),
    align: (left, center, center, center),
    stroke: 0.5pt,
    table.header([*Metoda*], [*CIFAR-10*], [*CIFAR-100*], [*FashionMNIST*]),
    [ACO], [0.7224], [*0.3476*], [0.9238],
    [GA], [*0.7420*], [0.3474], [0.9220],
    [Harmony Search], [0.6592], [0.3080], [*0.9273*],
    [Manual search], [0.6828], [0.2840], [0.9202],
    [PSO], [0.6372], [0.2636], [0.9258],
    [Random search], [0.6836], [0.3122], [0.9253],
  )
) <tab:cross-val>

#figure(
  image("figures/cross_dataset_best_val_accuracy.png", width: 95%),
  caption: [Najlepsza `val_accuracy` per metoda na każdym zbiorze (faza przeszukiwania).],
) <fig:cross>

Na CIFAR-10 wyraźnie najlepiej wypada GA (0.742), wyprzedzając ACO o ok. 2 pp
i random search o 5.8 pp. Na CIFAR-100 ACO i GA są praktycznie
nierozróżnialne (0.3476 vs 0.3474) i obie metody dominują nad pozostałymi.
Na FashionMNIST różnice między metodami są małe (szerokość pasma $approx$ 1 pp):
wszystkie metody schodzą w obszar 0.920–0.928, co sugeruje, że problem jest
łatwy i hiperparametry mają mniejsze znaczenie.

== Krzywe best-so-far

Wykresy @fig:bsf-cifar10, @fig:bsf-cifar100 i @fig:bsf-fashion ilustrują,
jak rosła najlepsza dotychczas znaleziona wartość `val_accuracy`
w funkcji numeru ewaluacji.

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

Na CIFAR-10 random search i manual search szybko znajdują dobrą konfigurację
($approx$ 0.68), ale nie potrafią jej poprawić. GA i ACO kontynuują poprawę
aż do końca budżetu, co pokazuje korzyść z informowanego przeszukiwania.
PSO zaskakująco pozostaje w tyle — mechanizm prędkości źle współpracuje z
głęboko kategorycznymi wymiarami (wielokrotnie trafia w obszary wysokich
`learning_rate` skutkujące divergencją, widoczną też w szerokich rozrzutach
`val_accuracy.std` w @tab:summary).

== Czas do osiągnięcia najlepszego wyniku

Sam szczyt jakości nie mówi wszystkiego: liczy się też, *jak szybko* metoda
dochodzi do swojego najlepszego punktu. @fig:ttb-c10 pokazuje czas (w sekundach)
do osiągnięcia najlepszej `val_accuracy` dla CIFAR-10.

#figure(
  image("figures/time_to_best_CIFAR10.png", width: 75%),
  caption: [Czas do najlepszej `val_accuracy` (s), CIFAR-10.],
) <fig:ttb-c10>

Manual search jest najszybszy (57 s), bo trafia w dobrą konfigurację już w
3. ewaluacji, ale jego górny sufit jest niski. GA dochodzi do znacznie
wyższego sufitu (0.742) w 309 s, ACO potrzebuje 522 s, by osiągnąć 0.722.
Dla CIFAR-100 natomiast ACO znajduje swoje optimum błyskawicznie (59 s,
iteracja 3), ale tak wysokie tempo sugeruje szczęśliwy traf niż systematyczną
przewagę — random search potrzebuje 370 s na 0.312, co jest wynikiem
bardziej powtarzalnym.

== Korelacje hiperparametr--metryka

Dla każdej pary (zbiór, metoda) policzono macierze korelacji Pearsona między
hiperparametrami a czterema metrykami: `val_accuracy`, `test_accuracy`,
`time_sec`, `num_params`. @fig:corr-ga-c10 pokazuje przykładową mapę dla GA
na CIFAR-10.

#figure(
  image("figures/hyperparam_correlations_CIFAR10/hyperparam_metric_correlation_ga.png", width: 80%),
  caption: [Korelacje hiperparametrów z metrykami (GA, CIFAR-10).],
) <fig:corr-ga-c10>

Najsilniejsze sygnały na CIFAR-10 (GA):

- `batch_size` vs `val_accuracy`: $-0.79$ — mniejsze batche istotnie poprawiają jakość przy 5-epokowym budżecie, prawdopodobnie dzięki większej liczbie kroków gradientu.
- `filters_1` vs `val_accuracy`: $+0.71$ — więcej filtrów w pierwszym bloku konsekwentnie pomaga.
- `dense_units` vs `val_accuracy`: $+0.75$ — szersza warstwa klasyfikatora poprawia dopasowanie.
- `dropout` vs `val_accuracy`: $-0.43$ — przy krótkim treningu silny dropout szkodzi (nie zdąży się „wyciągnąć").
- `num_blocks` vs `time_sec`: $+0.74$ — przewidywalny koszt głębszej architektury.

Wyniki te potwierdzają intuicję, że w reżimie krótkiego treningu preferowane są
„szerokie" a nie „głębokie" architektury.

== Jakość końcowa po dotrenowaniu

Właściwym celem strojenia jest jakość po pełnym treningu, nie sama `val_accuracy`
w fazie eksploracji. @tab:final i @fig:final prezentują test accuracy po
20 epokach dotrenowania najlepszej konfiguracji dla każdej metody.

#figure(
  caption: [Końcowa `test_accuracy` po 20-epokowym dotrenowaniu najlepszej konfiguracji.],
  table(
    columns: (auto, auto, auto, auto),
    align: (left, center, center, center),
    stroke: 0.5pt,
    table.header([*Metoda*], [*CIFAR-10*], [*CIFAR-100*], [*FashionMNIST*]),
    [ACO], [0.7744], [0.3605], [0.9264],
    [GA], [*0.8092*], [*0.4314*], [*0.9331*],
    [Harmony Search], [0.6546], [0.3250], [0.9249],
    [Manual search], [0.7772], [0.4064], [0.9213],
    [PSO], [0.6710], [0.2998], [0.9274],
    [Random search], [0.6976], [0.3392], [0.9225],
  )
) <tab:final>

#figure(
  image("figures/final_test_accuracy_pivot.png", width: 95%),
  caption: [Końcowa `test_accuracy` po retreningu — metoda $times$ zbiór.],
) <fig:final>

GA wygrywa na wszystkich trzech zbiorach, z największą przewagą na CIFAR-100
($approx$ 2.5 pp nad manual i ACO, $approx$ 9 pp nad random). Manual search
to *silny* drugi wynik na CIFAR-100 (0.4064), co pokazuje, że dobrze dobrany
punkt startowy może być konkurencyjny z metaheurystykami, gdy budżet
przeszukiwania jest mały. Krzywe dotrenowania (@fig:retrain) pokazują, że
różnice między metodami stabilizują się po ok. 10 epokach.

#figure(
  image("figures/final_retraining_val_curves.png", width: 100%),
  caption: [Krzywe `val_accuracy` w trakcie 20-epokowego dotrenowania.],
) <fig:retrain>

== Rozrzut metod (powtarzalność)

Tabela @tab:summary pokazuje statystyki `val_accuracy` i `test_accuracy`
uśrednione po wszystkich ewaluacjach każdej metody na CIFAR-10.
Niskie `std` dla manual search ($0.024$) jest artefaktem krzywej
kandydatów (wszystkie pięć konfiguracji jest a priori sensowne).
Wysokie `std` dla PSO ($0.194$) i random ($0.161$) oznacza, że metody te
regularnie trafiają w obszary divergentne (`lr` $> 0.005$ + `sgd`).
GA ma najniższą `std` wśród metod stochastycznych ($0.082$), co sugeruje,
że selekcja skutecznie odfiltrowuje najgorsze warianty.

#figure(
  caption: [Statystyki `val_accuracy` / `test_accuracy` po wszystkich ewaluacjach (CIFAR-10).],
  table(
    columns: (auto, auto, auto, auto, auto, auto, auto),
    align: (left, center, center, center, center, center, center),
    stroke: 0.5pt,
    table.header(
      [*Metoda*],
      [*val max*], [*val mean*], [*val std*],
      [*test max*], [*test mean*], [*test std*],
    ),
    [ACO], [0.722], [0.579], [0.132], [0.718], [0.579], [0.132],
    [GA], [*0.742*], [*0.665*], [0.082], [*0.741*], [*0.664*], [0.082],
    [Harmony S.], [0.659], [0.579], [0.120], [0.658], [0.577], [0.120],
    [Manual], [0.683], [0.656], [*0.024*], [0.686], [0.661], [*0.023*],
    [PSO], [0.637], [0.461], [0.194], [0.634], [0.458], [0.194],
    [Random], [0.684], [0.535], [0.161], [0.688], [0.535], [0.162],
  )
) <tab:summary>

== Najlepsze konfiguracje

@tab:best zestawia wygrywające konfiguracje na CIFAR-10. Warto zwrócić uwagę,
że GA i ACO wybrały względnie *małe* tempa uczenia ($3.4 times 10^(-4)$,
$7.0 times 10^(-4)$) i duży rozmiar gęstej warstwy (256), co jest zgodne
z korelacjami z sekcji 3.4. PSO trafiło w $5.8 times 10^(-3)$ z `dropout=0`
— konfigurację o małej ogólności, co objawiło się niskim wynikiem końcowym.

#figure(
  caption: [Najlepsze konfiguracje per metoda (CIFAR-10).],
  table(
    columns: (auto, auto, auto, auto, auto, auto, auto),
    align: (left, center, center, center, center, center, center),
    stroke: 0.5pt,
    table.header(
      [*Metoda*], [*lr*], [*bs*], [*blocks*], [*dropout*], [*dense*], [*opt*],
    ),
    [GA], [$3.4 e{-4}$], [32], [3], [0.15], [256], [adam],
    [ACO], [$7.0 e{-4}$], [128], [2], [0.23], [256], [adam],
    [Random], [$6.3 e{-4}$], [32], [1], [0.33], [128], [adamw],
    [Manual], [$1.0 e{-3}$], [64], [3], [0.35], [128], [—],
    [HS], [$2.3 e{-3}$], [32], [1], [0.03], [256], [sgd],
    [PSO], [$5.8 e{-3}$], [256], [2], [0.00], [64], [adam],
  )
) <tab:best>

// ============ DYSKUSJA ============

= Dyskusja

*GA jako najlepszy ogólny wybór.* W trzech z trzech zbiorów GA dał najwyższą
końcową `test_accuracy`. Wynika to z trzech właściwości: (i) selekcja
turniejowa odfiltrowuje słabe konfiguracje szybciej niż próbkowanie losowe,
(ii) krzywa populacyjna lepiej utrzymuje różnorodność niż PSO z silnym
przyciąganiem do lokalnego najlepszego, (iii) krzyżowanie jednopunktowe
dobrze radzi sobie z mieszanymi zmiennymi (kategoryczne + ciągłe).

*PSO w wersji „waniliowej" słabo pasuje.* PSO zakłada ciągłe, gładkie
powierzchnie, podczas gdy tu połowa wymiarów to kategorie o 2--4 poziomach.
Prędkości cząstek przesuwają je w obszary, które po naprawie stają się
bardzo podobne do losowania jednostajnego, więc przewaga nad random search
znika, a szerokie rozrzuty `lr` generują wiele ewaluacji z modelami
zdegenerowanymi. Lepsze wyniki uzyskałyby zapewne warianty dyskretnego PSO.

*Manual search jest mocną linią bazową.* Szczególnie na CIFAR-100
($0.4064$ po retreningu) ręczna konfiguracja bije wszystkie metaheurystyki
poza GA. Dla małych budżetów warto *zainwestować 5 minut w sensowną siatkę
wstępną* zanim się uruchomi pełną optymalizację.

*Harmony Search jest ostrożny.* HS często „zamrożą" się wcześnie (najlepszy
wynik na FashionMNIST już w iteracji 13, potem brak poprawy) — prawdopodobnie
$"HMCR"=0.9$ jest zbyt wysoki, co ogranicza eksplorację.

*Dobór hiperparametrów z 5-epokowych ewaluacji generalizuje się.* Ranking metod
po fazie przeszukiwania w dużej mierze pokrywa się z rankingiem po retreningu
(korelacja Pearsona `search_val_accuracy` $tilde$ `final_test_accuracy` na
połączonych zbiorach to $rho approx 0.95$), co uzasadnia przyjętą procedurę
dwuetapową: szukamy krótko, trenujemy długo tylko zwycięzcę.

*Ograniczenia.* Wszystkie uruchomienia miały to samo ziarno — potrzebne są
powtórzenia na różnych ziarnach dla testów istotności. Budżet 20 ewaluacji
jest mały; dla większych budżetów metaheurystyki mają potencjał oddalić się
od random search / manual search. Nie strojone były schedulery tempa uczenia
ani augmentacja danych, co wpływa szczególnie na CIFAR-100.

// ============ PODSUMOWANIE ============

= Podsumowanie

Porównanie sześciu metod strojenia hiperparametrów CNN na trzech zbiorach
pokazuje, że:

+ *GA* daje najlepszą końcową `test_accuracy` na FashionMNIST (0.9331),
  CIFAR-10 (0.8092) i CIFAR-100 (0.4314), wygrywając na wszystkich trzech
  zbiorach.
+ *ACO* jest solidnym drugim wyborem, szczególnie gdy czas do pierwszego
  dobrego wyniku jest krytyczny.
+ *Manual search* pozostaje niezwykle silną linią bazową dla małych budżetów
  i powinien być raportowany w każdej publikacji.
+ *PSO* w naiwnej implementacji słabo pasuje do mieszanych przestrzeni
  kategoryczno-ciągłych; bez specjalnej adaptacji nie warto go stosować.
+ Korelacje hiperparametr$->$jakość są silne i spójne z intuicją
  (małe batche, więcej filtrów w pierwszym bloku, umiarkowany dropout,
  małe `lr` w połączeniu z Adamem).
+ Jakość po 5 epokach dobrze przepowiada jakość po 20 epokach ($rho approx 0.95$),
  co uzasadnia dwuetapowy pipeline: szybkie przeszukiwanie + retreining.

#v(0.6cm)

// [1] Purnomo H.D., Gonsalves T., Mailoa E.,
//   Santoso F.J., Pribadi M.R. _Metaheuristics Approach for Hyperparameter Tuning
//   of Convolutional Neural Network_.\

// [2] Goldberg D.E. _Genetic Algorithms in Search, Optimization, and Machine Learning_.
//   Addison-Wesley, 1989. \

// [3] Kennedy J., Eberhart R. _Particle Swarm Optimization_. Proc. IEEE ICNN, 1995. \

// [4] Dorigo M., Stützle T. _Ant Colony Optimization_. MIT Press, 2004. \

// [5] Geem Z.W., Kim J.H., Loganathan G.V. _A New Heuristic Optimization Algorithm:
//   Harmony Search_. Simulation 76(2), 2001. \

// [6] Krizhevsky A. _Learning Multiple Layers of Features from Tiny Images_.
//   Tech. Report, University of Toronto, 2009 (zbiory CIFAR-10 / CIFAR-100). \

// [7] Xiao H., Rasul K., Vollgraf R. _Fashion-MNIST: a Novel Image Dataset for
//   Benchmarking Machine Learning Algorithms_. arXiv:1708.07747, 2017.
