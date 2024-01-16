# Skripta: Machine Learning vs. Deep Learning 

## Machine Learning (ML)

## 3 Základní Typy Machine Learning (ML)

### 1. Supervised Learning (SL)

##### Definice:
Supervised Learning (dozorované učení) představuje typ Machine Learning, kde algoritmy jsou trénovány na základě předložených dat spolu s odpovídajícími známými výsledky. V tomto přístupu máme dozorčího "učitele," který poskytuje správné odpovědi, a algoritmus se snaží naučit mapování mezi vstupními daty a odpovídajícími výstupy.

###### Princip:
###### 1. Trénovací Data:

Máme k dispozici trénovací data, která obsahují páry vstupů a odpovídajících výstupů. Například, pokud trénujeme algoritmus na klasifikaci e-mailů, každý e-mail v trénovací sadě bude obsahovat označení "spam" nebo "ne-spam."
###### 2. Naučení Vzorů:

Algoritmus se snaží identifikovat vzory a vztahy mezi vstupy a výstupy v trénovacích datech. To zahrnuje identifikaci klíčových rysů nebo atributů, které ovlivňují výsledky.
###### 3. Generování Predikcí pro Nová Data:

Jakmile je algoritmus úspěšně natrénován, může být použit k generování predikcí pro nová, dosud neviděná data. Například, pokud jsme natrénovali klasifikátor e-mailů, může být použit k určení, zda nový e-mail je spam či ne.
##### Příklad:
Klasifikace e-mailů jako spam nebo ne-spam na základě trénovacích dat.

###### Trénovací Data:

E-maily označené jako "spam" nebo "ne-spam."
Naučené Vzory:

Algoritmus se naučí rozpoznávat klíčové slova nebo vzory spojené s e-maily, které jsou pravděpodobně spamem.
Generování Predikce:

Když dostane nový e-mail, algoritmus analyzuje jeho obsah a rozhodne, zda je pravděpodobně spam nebo ne-spam na základě naučených vzorů.
Tímto způsobem dozorované učení umožňuje modelům generalizovat z trénovacích dat a aplikovat své znalosti na nová, dosud neviděná data. Je to široce využívaný přístup pro řešení různých problémů, jako je klasifikace, regrese a predikce.

### Ukázka

![image](https://github.com/0NDRS/PVA/assets/145441873/937c50a5-285f-4baa-af85-8c00a0af77ec)

Tento kód demonstruje vytvoření a trénování jednoduchého lineárního regresního modelu pomocí knihovny TensorFlow v jazyce Python:

Generování trénovacích dat:

X_train je vytvořeno jako 100 rovnoměrně rozložených hodnot od 0 do 10.
y_train je vytvořeno jako lineární vztah 2*X + 1 s přidaným náhodným šumem.
Vytvoření modelu v TensorFlow:

Vytvoření jednoduchého lineárního regresního modelu s jednou vrstvou (Dense) a jedním vstupem.
Kompilace modelu:

Kompilace modelu s optimizerem 'sgd' (Stochastic Gradient Descent) a loss funkcí 'mean_squared_error'.
Trénink modelu:

Trénink modelu na trénovacích datech po dobu 100 epoch.
Vytvoření nových dat pro predikci:

Vytvoření nových hodnot pro predikci modelu.
Predikce s modelem:

Použití natrénovaného modelu k predikci výstupních hodnot pro nová data.
Vizualizace výsledků:

Vykreslení skutečných dat a predikcí modelu pro vizualizaci výsledků.

Výsledek

Ve výsledku můžeme vidět modrá body jakožto vstupní data a AI vytvoří přímku tak, aby byla nejblíže všem bodům

![image](https://github.com/0NDRS/PVA/assets/145441873/f3d6be13-eb28-491e-92db-5619c5b91db5)

### 2. Unsupervised Learning (UL)

##### Definice:
Unsupervised Learning (nedozorované učení) je typ Machine Learning, který nevyžaduje správné odpovědi k tréninku. Algoritmy se učí ze struktury dat a snaží se najít vzory, skupiny nebo vztahy bez explicitního cílového výstupu. Je vhodný pro situace, kde nemáme předem označená data.

#### Podkategorie Unsupervised Learning (UL):
#### Clustering:
Princip: Algoritmy třídí vstupy na základě shodnosti.

Clusteringové algoritmy se snaží identifikovat skupiny nebo shluky podobných prvků v datech. Prvky uvnitř jednoho shluku by měly být si podobné, zatímco prvky v různých shlucích by měly být odlišné.
Příklad: Hudební platformy, které třídí uživatele do skupin podle podobných hudebních preferencí.

Algoritmy mohou analyzovat poslechové historie uživatelů a identifikovat skupiny s podobnými vkusy. To může vést k lepším doporučením a personalizovaným hudebním zážitkům.
##### Association:
Princip: Algoritmy sledují předchozí aktivity a odvozují vztahy.

Algoritmy pro asociační pravidla zkoumají data a hledají vzory spojené s častými kombinacemi atributů. Tyto vzory mohou odhalit vztahy mezi různými prvky v datech.
Příklad: Systémy doporučování produktů online obchodů na základě historie nákupů.

Analýza historie nákupů může odhalit, které produkty jsou často kupovány společně. Na základě těchto vzorů může systém navrhovat další produkty, které by mohly zajímat zákazníka.
##### Dimensional Reduction:
Princip: Redukce dimenzí dat pro další zpracování.

Cílem je snížit počet atributů nebo dimenzí dat, což usnadňuje analýzu, vizualizaci a zpracování dat.
Příklad: Použití autoencoderu pro odstranění hluku z obrázků, čímž se snižuje jejich rozměr a zvyšuje kvalita.

Autoencoder je model neuronové sítě, který se učí reprezentaci dat a následně může být použit k odstranění nepodstatelných informací nebo šumu z obrázků, což zjednodušuje jejich zpracování.

#### Principy Jednotlivých Typů Unsupervised Learning:
##### Clustering:
Identifikace Podobných Vzorů nebo Skupin v Datech:

Algoritmy pro shlukování analyzují vzory v datech a snaží se identifikovat skupiny, ve kterých jsou data podobná.
Příklady Algoritmů:

##### K-means:
Rozděluje data do K shluků na základě průměrných hodnot atributů.
Hierarchické Shlukování:
Vytváří hierarchickou strukturu shluků, což umožňuje podrobnější nebo hrubší rozdělení.
Association:
Hledání Vzorů v Datech spojených s Častými Kombinacemi:

Algoritmy pro asociační pravidla identifikují vztahy mezi různými atributy nebo položkami, které se v datech často vyskytují společně.
Použití v E-commerce pro Doporučování Produktů:

Algoritmy mohou odhalit, které produkty jsou často zakoupeny společně a poskytnout personalizované doporučení zákazníkům.
Dimensional Reduction:
Cíl: Snížení Počtu Atributů nebo Funkcí v Datech:

Princip spočívá ve snížení dimenzionality dat, což znamená zachování co nejvíce informací při snížení počtu atributů.
Efektivní Zpracování Dat a Eliminace Nepodstatelných Informací:

Redukce dimenzionality usnadňuje analýzu a zpracování dat a současně pomáhá odstranit redundantní nebo nepodstatelné informace.
Příklady:
##### Clustering:

Například, K-means algoritmus může být použit pro klasifikaci zákazníků do různých segmentů podle jejich nákupních vzorů.
Association:

V e-commerce může algoritmus odhalit, že zákazníci, kteří kupují chléb, mají často zájem o máslo, což může vést k doporučení těchto položek společně.
Dimensional Reduction:

PCA (Principal Component Analysis) může být použit pro snížení dimenzionality obrázků, což umožní efektivnější zpracování a analyzování obrázkových dat.

### Ukázka

![image](https://github.com/0NDRS/PVA/assets/145441873/2713408f-ac81-403a-82e0-469c544b2e54)

Zde vidíme ukázku Unsupervised Learningu, kdy zde máme neuronovou síť s autoenkoderem, který usměrňuje modré body:

Generování náhodných dat:

Vytvoření 100 náhodných bodů ve 2D pomocí NumPy.
Vytvoření K-means modelu:

V Tensorflow používáme tf.compat.v1.estimator.experimental.KMeans pro vytvoření K-means modelu s 3 shluky.
Definice vstupní funkce:

Vytvoření vstupní funkce input_fn, která převede data na tensor TensorFlow.
Trénování modelu K-means:

Trénink modelu K-means pomocí metody train s využitím vstupní funkce.
Přiřazení k shlukům:

Získání přiřazení k shlukům pro každý bod v datech.
Vykreslení dat:

Vykreslení dat s ohledem na přiřazení k shlukům. Každý shluk je zobrazen jinou barvou pomocí c=assignments.

Výsledek

Zde vidíme výsledek programu, kdy můžeme vidět modré body jakožto nějaký šum a AI šum odstraní viz, červené body

![image](https://github.com/0NDRS/PVA/assets/145441873/60679802-d742-46e7-9aae-69160a53e98b)


### 3. Reinforcement Learning (RL)
#### Definice:
Reinforcement Learning (RL) je paradigma strojového učení, které je inspirováno způsobem, jakým lidé a zvířata učí nové dovednosti interakcí s okolním světem.

#### Princip:
Agent interaguje s prostředím a rozhoduje o akcích:

Agent, reprezentující entitu schopnou učení, provádí akce v daném prostředí. Tyto akce mohou mít různé důsledky.
Prostředí reaguje na akce agenta a poskytuje odměnu nebo trest:

Po provedení akce prostředí reaguje udělením odměny nebo trestu. Odměna slouží jako zpětná vazba pro agenta a udává, jak dobře nebo špatně se agent rozhodl.
Cílem agenta je maximalizovat kumulativní odměnu v čase:

Agent se snaží naučit optimální strategii tak, aby v průběhu času maximalizoval kumulativní odměnu. To znamená, že se snaží vybrat akce, které vedou k co nejlepším výsledkům.
##### Příklad:
Řízení autonomních vozidel:

Agentem může být systém autonomního řízení, který provádí různé akce, jako jsou změny rychlosti nebo směru, a prostředí reaguje na tyto akce, například pohybujícími se vozidly nebo překážkami. Odměna nebo trest může být přidělena na základě bezpečné a efektivní navigace.
Optimalizace procesů ve strojovém řízení:

Agent může být program, který optimalizuje procesy v továrně nebo výrobním zařízení. Prostředí reaguje na provedené akce agenta a poskytuje zpětnou vazbu v podobě efektivity procesů a minimalizace chyb.
Funkce Machine Learning:
Reinforcement Learning často využívá pokročilé techniky strojového učení, zejména hlubokého učení (Deep Learning), k naučení složitých vzorů a strategií v interakci s prostředím.

#### Omezení a Možnosti ML:
Závislost na Kvalitě Dat

Kvalita prostředí a přesnost odměn je klíčová pro úspěšné učení agenta.
Volba Algoritmu:

Volba algoritmu pro učení agenta je klíčovým faktorem pro dosažení optimálních výsledků.
Příklad:
Při klasifikaci e-mailů jako spamu nebo ne-spamu, kvalitní trénovací data obsahující rozmanitost e-mailů jsou klíčová pro úspěch modelu.

### Ukázka

![image](https://github.com/0NDRS/PVA/assets/145441873/4fe9ff84-ed8e-4522-99ae-c883401bca93)

Tento kód implementuje jednoduchý algoritmus Q-learningu pro učení s posilovaním. Q-learning je algoritmus, který umožňuje agentovi naučit se optimální strategii rozhodování ve specifickém prostředí. V tomto konkrétním případě je prostředí zjednodušeno na pevný počet stavů a akcí.

Stručně:

Agent se trénuje v prostředí s pevným počtem stavů a akcí pomocí Q-learningu.
Q-Values jsou aktualizovány na základě odměn a odhadovaných budoucích odměn v každém stavu.
Po trénování jsou vypsány naučené Q-Values a provedeno testování agenta ve zvoleném testovacím stavu.

Import knihoven:

Importuje knihovnu NumPy, což je populární knihovna pro práci s numerickými daty v Pythonu. V tomto kódu bude použita pro práci s maticemi Q-Values.
Definice prostředí:

num_states určuje počet stavů v prostředí.
num_actions určuje počet akcí, které může agent podniknout.
Q_values je matice Q-Values, která slouží k odhadu hodnot akcí ve stavu. Při inicializaci jsou všechny hodnoty nastaveny na nulu.
Nastavení parametrů Q-Learningu:

learning_rate určuje, jak rychle agent aktualizuje své odhady Q-Values na základě nových informací.
discount_factor ovlivňuje, jak moc agent bere v úvahu budoucí odměny při aktualizaci Q-Values.
num_episodes určuje počet epizod (her), které agent použije k trénování.
Trénování Q-Learningem:

Cyklus for episode in range(num_episodes): iteruje přes všechny trénovací epizody.
state je náhodně inicializovaný počáteční stav pro každou epizodu.
Vnitřní cyklus while not done: reprezentuje jednu epizodu hry.
Volba akce je řízena epsilon-greedy strategií, která umožňuje náhodný výběr akce s pravděpodobností epsilon.
Simulovaný přechod stavů a získání odměny.
Aktualizace Q-Values pomocí Q-Learning aktualizace.
Výpis naučených Q-Values:

Po dokončení trénování se vypíší naučené Q-Values pro každý stav a akci. Tato matice poskytuje odhad budoucích odměn pro každou možnou akci ve všech stavech.
Testování naučeného agenta:

Náhodně se vybere testovací stav.
Agent vybere nejlepší akci podle naučených Q-Values pro daný testovací stav.
Výsledky testování se vypíší, což může obsahovat informace o testovacím stavu, vybrané akci a dalších relevantních informacích.


Výsledek

![image](https://github.com/0NDRS/PVA/assets/145441873/40c827b3-1db1-4237-96ba-6ce833d5de54)



## Předzpracování Dat v Machine Learning

### 1. Úvod:
Dobrý den všem. Dnes se zaměříme na klíčový krok v procesu Machine Learning, a to je předzpracování dat. Správné předzpracování dat je klíčové pro dosažení úspěšných výsledků ve strojovém učení.

### 2. Co je Předzpracování Dat:
Předzpracování dat je soubor kroků a technik, které se aplikují na vstupní data, aby se zlepšila kvalita a vhodnost pro modelování. Toto zahrnuje různé činnosti, jako jsou:

Čištění Dat: Odstranění chybějících nebo nesprávných hodnot.
Normalizace: Převedení hodnot na stejný měřítko.
Transformace: Úprava dat pro lepší přizpůsobení modelům.
Redukce Rozměrů: Snížení počtu atributů pro zjednodušení modelů.
### 3. Kroky Předzpracování Dat:

#### a. Čištění Dat:
Identifikace a odstranění chybějících hodnot:

Procházíme dataset a identifikujeme hodnoty, které chybí.
Volba strategie: Odstranění řádků nebo doplnění chybějících hodnot (např., průměrem nebo mediánem).
Kontrola a korekce chybných hodnot:

Zjištění anomálií nebo nesprávných hodnot.
Oprava chybných hodnot na základě logiky domény nebo strategie odstranění.

#### b. Normalizace:
Zajištění, aby všechny atributy měly podobné měřítko:

Výpočet statistik (průměr, rozptyl) pro každý atribut.
Použití metody normalizace (standardizace nebo normalizace) k převedení hodnot na podobné měřítko.
Použití standardizace nebo normalizace podle potřeby:

Standardizace (z-score normalizace): Převedení hodnot na standardní normální rozdělení.
Normalizace do rozsahu: Převedení hodnot na konkrétní rozsah, často (0, 1).

#### c. Transformace:
Zakódování kategoriálních atributů do číselných hodnot:

Použití metody jako Label Encoding nebo One-Hot Encoding k převodu kategoriálních dat na čísla.
Label Encoding: Přiřazuje unikátní číslo každé kategorii.
One-Hot Encoding: Vytváří binární sloupce pro každou kategorii.
Vytvoření nových atributů pro lepší reprezentaci dat:

Extrahování nových vlastností z existujících atributů.
Přidání interakčních termínů nebo agregovaných statistik.

#### d. Redukce Rozměrů:
Použití technik jako Principal Component Analysis (PCA) pro snížení dimenzionality:
Aplikace lineární transformace k datům pro redukci dimenzionality.
Výběr počtu hlavních komponent tak, aby zachoval dostatek informací.
Snížení počtu atributů, což zjednodušuje modely a snižuje výpočetní náročnost.
Tímto způsobem, provedením těchto kroků předzpracování dat, můžeme zajistit, že data jsou vhodná pro úspěšné vytváření a trénování modelů v oblasti strojového učení.

### Ukázka

![image](https://github.com/0NDRS/PVA/assets/145441873/3d25cf2e-bd09-48a2-9e67-23ee76a818c2)

#### Tento kód představuje proces předzpracování dat pomocí knihoven zejména z balíčku scikit-learn v jazyce Python. Zde je popis jednotlivých částí:

Vytvoření umělých dat:
Generují se náhodná data (X1, X2, y) s použitím NumPy. Data jsou následně uložena do DataFrame (data), který je poté exportován do CSV souboru.
## Evaluace Modelu
Evaluace modelu je klíčovým krokem v procesu strojového učení, který nám poskytuje informace o výkonu modelu na nezávislých datech. Různé metriky a postupy jsou využívány k hodnocení schopnosti modelu generalizovat na nová data. Zde se zaměříme na principy evaluace modelu.

Normalizace dat v TensorFlow:
Funkce normalize_data načte data ze souboru, provede standardizaci dat pomocí StandardScaler a následně vytvoří DataFrame s normalizovanými daty.
Původní a normalizovaná data jsou vypsána na výstup.

Tento kód generuje, ukládá a normalizuje umělá data a poskytuje přehled o původních a normalizovaných datech v podobě výstupu.

Výsledek

Zde vidíme normalizovaná data, kde hodnoty se hodně lišíy, ale zaměřte se především na relativní vzájemné postavení hodnot v rámci každého atributu a na to, jak jsou standardizované hodnoty blízké střední hodnotě 0 a jak mají standardní odchylku 1.

![image](https://github.com/0NDRS/PVA/assets/145441873/65b2c933-17df-4142-96cf-d43e33dc2788)

### 1. Rozdělení Dat:
#### Princip:
Data jsou rozdělena na trénovací a testovací sady.
#### Vysvětlení:
Trénovací data jsou použita pro samotné učení modelu, zatímco testovací data slouží k ověření, jak dobře se model naučený na trénovacích datech chová na nových, neviděných datech.
### 2. Metriky Evaluace:
#### Princip:
Používání metrik pro kvantifikaci výkonu modelu.
#### Vysvětlení:
Metriky zahrnují přesnost (accuracy), přesnost (precision), návratnost (recall), F1 skóre a další. Každá metrika poskytuje jiný úhel pohledu na výkon modelu a je vhodná pro různé typy úloh.
### 3. Cross-Validation:
#### Princip:
Opakované rozdělení dat do trénovacích a testovacích sad.
#### Vysvětlení:
Křížová validace umožňuje lepší využití dostupných dat a poskytuje robustnější odhad výkonu modelu. Typickým příkladem je K-fold cross-validation.
### 4. ROC Křivka a AUC:
#### Princip:
Grafická reprezentace citlivosti a specifičnosti modelu.
#### Vysvětlení:
Receiver Operating Characteristic (ROC) křivka a plocha pod ní (AUC) jsou často používány pro modely klasifikace. Poskytují komplexní pohled na citlivost a specifičnost při různých prahových hodnotách.
### 5. Matice Záměn:
#### Princip:
Vytvoření tabulky s počty správných a nesprávných klasifikací.
#### Vysvětlení:
Matice záměn poskytuje detailní informace o tom, jak model klasifikuje jednotlivé třídy. Obsahuje hodnoty jako true positive, true negative, false positive a false negative.
### 6. Regrese a Metriky Regrese:
#### Princip:
Pro úlohy regrese jsou využívány specifické metriky.
#### Vysvětlení:
Mean Absolute Error (MAE), Mean Squared Error (MSE) a R-squared jsou běžné metriky pro hodnocení regresních modelů.
### 7. Overfitting a Underfitting:
#### Princip:
Monitorování jevů, kdy model buď příliš dobře zapamatovává trénovací data (overfitting) nebo je nedostatečně komplexní (underfitting).
#### Vysvětlení:
Overfitting a underfitting mohou vést k nepřesné generalizaci na nová data. Pravidelná validace modelu na testovacích datech pomáhá vyhodnocovat tyto jevy.
### 8. Interpretace Výsledků:
#### Princip:
Analýza výsledků s ohledem na konkrétní požadavky a kontext úlohy.
#### Vysvětlení:
Interpretace výsledků je klíčová pro zajištění, že výkon modelu odpovídá očekáváním a potřebám konkrétní úlohy.
### 9. Přizpůsobení Metodologii:
Princip:
Přizpůsobení metody evaluace podle konkrétních charakteristik úlohy.
Vysvětlení:
Různé úlohy vyžadují různé metriky a přístupy k evaluaci. Přizpůsobení metodologie zajišťuje relevantní a informativní výsledky.
Příklad:
Při klasifikaci e-mailů jako spamu nebo ne-spamu můžeme použít metriky jako přesnost, návratnost a F1 skóre. Matice záměn nám poskytne detailní informace o tom, kolik e-mailů bylo správně či nesprávně klasifikováno jako spam nebo ne-spam. Tyto informace jsou klíčové pro další vylepšování modelu a zvyšování jeho účinnosti.

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
