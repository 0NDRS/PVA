# Skripta: Machine Learning vs. Deep Learning 

## Machine Learning (ML)

### 3 Základní Typy Machine Learning (ML)
#### 1. Supervised Learning (SL)
Definice: Supervised Learning (dozorované učení) je typ ML, kde algoritmy jsou trénovány na základě zadání dat spolu s odpovídajícími správnými výsledky.

Princip:

Data obsahují vstupy a odpovídající výstupy.
Algoritmus se snaží naučit vzory a vztahy mezi vstupy a výstupy.
Na základě naučených znalostí může poskytovat predikce pro nová data.
Příklad:

Klasifikace e-mailů jako spam nebo ne-spam na základě trénovacích dat.
#### 2. Unsupervised Learning (UL)
Definice: Unsupervised Learning (nedozorované učení) nevyužívá správné odpovědi k tréninku. Algoritmy se učí ze struktury dat a snaží se najít vzory nebo vztahy bez cílového výstupu.

Podkategorie UL:

Clustering
Princip: Algoritmy třídí vstupy na základě shodnosti.
Příklad: Hudební platformy třídící uživatele do skupin podle hudebních preferencí.
Association
Princip: Algoritmy sledují předchozí aktivity a odvozují vztahy.
Příklad: Doporučování produktů online obchodů na základě historie nákupů.
Dimensional Reduction
Princip: Redukce dimenzí dat pro další zpracování.
Příklad: Použití autoencoderu pro odstranění hluku z obrázků.
#### 3. Reinforcement Learning (RL)
Definice: Reinforcement Learning je inspirováno způsobem, jakým lidé a zvířata učí nové dovednosti interakcí s okolním světem.

Princip:

Agent interaguje s prostředím a rozhoduje o akcích.
Prostředí reaguje na akce agenta a poskytuje odměnu nebo trest.
Cílem agenta je maximalizovat kumulativní odměnu v čase.
Příklad:

Řízení autonomních vozidel nebo optimalizace procesů ve strojovém řízení.
Funkce Machine Learning
Omezení a Možnosti ML:

Závislost na Kvalitě Dat: Výkon ML je ovlivněn kvalitou a reprezentativností trénovacích dat.
Volba Algoritmu: Úspěch ML modelu závisí na volbě vhodného algoritmu pro konkrétní úkol.
Příklad:

Při klasifikaci e-mailů jako spamu nebo ne-spamu, kvalitní trénovací data obsahující rozmanitost e-mailů jsou klíčová pro úspěch modelu.


## Deep Learning (DL)
Jak to funguje
Definice: Deep Learning (Hluboké učení) je odvětvím strojového učení, které využívá hlubokých neuronových sítí pro analýzu a interpretaci dat.

Principy:

Hluboké neuronové sítě obsahují mnoho vrstev, které umožňují modelům učit se reprezentace dat hierarchicky.
Neuronové sítě jsou schopny automaticky odhalovat složité vzory v datech.
### 3 Typy Deep Learning Modelů
#### 1. Konvoluční neuronové sítě (CNN)
Princip:
Specializace: CNN jsou navrženy zejména pro zpracování a analýzu vizuálních dat, jako jsou obrázky a videa.
Konvoluce: Využívají konvoluční vrstvy k detekci různých vzorů v obrazu.
Pooling: Používají se pooling vrstvy k redukci rozměrů a zachování významných informací.
Příklad: Rozpoznávání objektů v obrázcích, klasifikace obrazů, detekce tváří.
#### 2. Rekurentní neuronové sítě (RNN)
Princip:
Práce s Sekvencemi: RNN jsou navrženy pro práci se sekvencemi dat, jako jsou časové řady nebo jazyková data.
Vnitřní Stav: Udržují vnitřní stav, který umožňuje zachytit kontext v čase.
Zpětné Propagace: Využívají zpětnou propagaci chyby k učení z předchozích stavů.
Příklad: Předpověď textu, generování hudby, strojový překlad.
#### 3. Hluboké autoenkódery
Princip:
Snížení Dimenzionality Dat: Hluboké autoenkódery slouží k redukci dimenzionality dat a odstranění hluku.
Encoder a Decoder: Obsahují encoder pro zakódování dat a decoder pro jejich dekódování.
Příklad: Zlepšení kvality obrázků odstraněním nežádoucích artefaktů, komprese dat.
Funkce Deep Learning
Síla Hlubokého Učení:
Automatické Extrahování Vlastností: Schopnost extrahovat relevantní vlastnosti ze vstupních dat.
Hierarchická Reprzentace: Schopnost učení hierarchických reprezentací dat.
Omezení a Možnosti DL
Potřeba Velkého Množství Dat: Pro efektivní učení je často vyžadováno velké množství dat.

Výpočetní Náročnost: Trénink hlubokých modelů může vyžadovat výpočetní zdroje.

#### Příklad:

Pro rozpoznávání objektů ve fotografiích jsou potřeba tisíce označených obrázků pro efektivní trénink modelu.
Tím jsme si přiblížili klíčové principy a typy Deep Learningu, a to včetně detailnějšího pohledu na jednotlivé modely. Dále porovnáme tuto technologii s Machine Learningem.

## Jak fungují vsrtvy v Deep Learingu (DL)

V Deep Learning (Hlubokém učení) jsou vrstvy základními stavebními bloky neuronových sítí. Každá vrstva má svou specifickou roli v extrakci a transformaci dat. Podívejme se na několik klíčových vrstev v Deep Learning:

### 1. Vstupní Vrstva
Funkce: Přijímá vstupní data, například obrázky, text nebo zvuk, a předává je do sítě.
Charakteristika: Počet neuronů v této vrstvě odpovídá rozměrům vstupních dat.
### 2. Skrytá (Hidden) Vrstva
Funkce: Tato vrstva je prostřední fází mezi vstupní a výstupní vrstvou, kde probíhá většina učení sítě.
Charakteristika: Může obsahovat různý počet neuronů a může být složena z několika vrstev, čímž vytváří hloubku sítě.
### 3. Výstupní Vrstva
Funkce: Generuje konečné výstupy nebo predikce na základě informací z předchozích vrstev.
Charakteristika: Počet neuronů odpovídá počtu tříd (pro klasifikaci) nebo formátu výstupu (např. regrese).
## Specifické Typy Vrstev
### 1. Konvoluční Vrstva (Convolutional Layer)
Použití: Zejména pro zpracování obrazových dat.
Funkce: Detekuje různé vzory a lokální struktury pomocí konvolučních filtrů.
Charakteristika: Obsahuje váhy (filtry), které se aplikují na malé části vstupních dat.
### 2. Rekurentní Vrstva (Recurrent Layer)
Použití: Při práci se sekvencemi dat, jako jsou časové řady nebo texty.
Funkce: Udržuje vnitřní stav pro zachycení kontextu v čase.
Charakteristika: Přenáší informace z předchozích časových kroků do současného kroku.
### 3. Normalizační Vrstva (Normalization Layer)
Použití: Normalizace výstupů vrstev, aby se stabilizoval a urychlil proces učení.
Funkce: Normalizuje aktivační hodnoty na nulový průměr a jednotkový rozptyl.
Charakteristika: Pomáhá předejít problémům, jako je přetrénování a zrychluje konvergenci modelu.
### 4. Dropout Vrstva
##### Funkce: Náhodně "vypíná" (ignoruje) některé neurony během tréninku, což zabraňuje přetrénování.
Charakteristika: Parametr určuje pravděpodobnost "vypnutí" každého neuronu.
Využití různých typů vrstev vytváří flexibilitu pro modelování různých typů dat a úkolů v rámci hlubokého učení. Hierarchie a vzájemné propojení těchto vrstev umožňují neuronovým sítím extrahovat a učit složité vzory v datech.

## Porovnání Machine Learning (ML) a Deep Learning (DL)
## Společné Principy
### 1. Učení z Dat:

##### ML: Trénink na základě dat a zkušeností, kde jsou algoritmy schopny provádět úkoly, pro které byly natrénovány.
##### DL: Hluboké učení také vychází z učení na datech, ale s využitím hlubokých neuronových sítí pro analýzu a interpretaci dat.
### 2. Výpočetní Modely:

##### ML: Využívá širokou škálu algoritmů, jako jsou rozhodovací stromy, lineární regrese, k-means a další.
##### DL: Zaměřuje se zejména na hluboké neuronové sítě, jako jsou konvoluční sítě, rekurentní sítě a autoenkódery.
## Struktura Modelů
### 1. Počet Vrstev:

##### ML: Obvykle se skládá z menšího počtu vrstev, obvykle jedné nebo dvou.
##### DL: Je charakteristické vícevrstvovými strukturami s možností obsahovat desítky až tisíce vrstev.
### 2. Schopnost Reprzentace Dat:

##### ML: Vytváří jednodušší reprezentace dat pomocí menších modelů.
##### DL: Schopno vytvářet složité hierarchické reprezentace dat díky hlubokým strukturám.
## Trénování Modelů
### 1. Množství Dat:

##### ML: Často vyžaduje méně dat pro efektivní trénování.
##### DL: Vyžaduje větší množství dat, aby dosáhlo optimálních výsledků.
### 2. Výpočetní Náročnost:

##### ML: Často vyžaduje nižší výpočetní náročnost.
##### DL: Pro trénování hlubokých sítí jsou často potřeba silné výpočetní zdroje.
## Aplikace
### 1. Druhy Úkolů:

##### ML: Často se využívá pro klasifikaci, regresi a shlukování.
##### DL: Ideální pro úkoly, jako je rozpoznávání obrazu, zpracování přirozeného jazyka a hlasové rozpoznávání.
### 2. Domény Použití:

##### ML: Využívá se ve vědeckém průzkumu, podnikovém rozhodování, finanční analýze.
##### DL: Často se uplatňuje v oblasti počítačového vidění, zpracování přirozeného jazyka, robotiky a autonomních systémů.
## Závěr
Obě technologie, Machine Learning a Deep Learning, mají své vlastní výhody a omezení. Zatímco Machine Learning se často uplatňuje v případech s menším objemem dat a vyžaduje méně výpočetních zdrojů, Deep Learning vyniká v komplexnějších úkolech, zejména v oblasti zpracování vizuálních a jazykových dat. Výběr mezi nimi závisí na konkrétním úkolu, dostupných datech a výpočetních zdrojích.
