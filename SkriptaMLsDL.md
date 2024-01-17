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
## 3 Typy Deep Learning Modelů

### 1. Konvoluční neuronové sítě (CNN)

#### Konvoluční Vrstvy:

Hlavním prvkem CNN jsou konvoluční vrstvy. Tyto vrstvy aplikují filtry (konvoluce) na vstupní data, což umožňuje extrahovat různé úrovně příznaků z obrazu nebo jiných vizuálních dat.
Filtry jsou malé matice vah, které se skládají zahodnocených hodnot. Během konvoluce je tento filtr posouván po vstupních datech a vytváří mapy příznaků.
#### Pooling (Slévání):

Po každé konvoluci může následovat pooling, což je operace slévání. Typicky se používá max pooling, kde se z každé skupiny hodnot vybere maximální hodnota. Pooling snižuje prostorovou dimenzi dat a zároveň zachovává důležité příznaky.
#### Aktivační Funkce:

Aktivační funkce, obvykle ReLU (Rectified Linear Unit), se aplikuje na výsledné hodnoty po každé konvoluci nebo poolingu. Aktivační funkce přidává nelinearitu do modelu.
#### Plně Propojené Vrstvy:

Konvoluční a pooling vrstvy jsou často následovány plně propojenými vrstvami, které se starají o konečné rozhodování a klasifikaci. Tyto vrstvy mohou mít také aktivační funkce.
#### Využití Vah a Lokálních Vzorů:

CNN využívají sdílení vah a tím se snižuje počet parametrů v modelu. Každý filtr se používá na celý obrázek, což umožňuje modelu zachytit lokální vzory nezávisle na jejich umístění.

### Použití Konvolučních Neuronových Sítí:

##### Rozpoznávání Obrazu:

CNN jsou široce využívány pro rozpoznávání objektů, tváří, znaků a dalších prvků v obrazech.
##### Klasifikace Obrazu:

Jsou efektivní při klasifikaci obrazů do různých tříd nebo kategorií.
##### Segmentace Obrazu:

Pomáhají při segmentaci obrazu na jednotlivé části nebo objekty.
##### Předzpracování Vstupů do Hlubokých Sítí:

Často se používají k předzpracování vstupních dat do hlubokých sítí pro jiné úlohy.
##### Analytika Videa:

Jsou využívány při analýze a pochopení videa, detekci pohybu a dalších úlohách v oblasti videa.
##### Zpracování Přirozeného Jazyka:

CNN mohou být adaptovány pro práci s textem a zpracování přirozeného jazyka.

Ukázka:

![image](https://github.com/0NDRS/PVA/assets/145441873/e822a29d-1a40-45fc-9b5e-58d470dd11eb)

Tento kód demonstruje jednoduchý konvoluční neuronový model pro klasifikaci číslic ze známého datasetu MNIST:

#### Načtení a Předzpracování Dat:

Načítá dataset MNIST, který obsahuje černobílé obrázky číslic od 0 do 9.
Normalizuje hodnoty pixelů na rozsah 0-1, což pomáhá při rychlejším učení modelu.
Přidává dimenzi pro kanál, protože černobílé obrázky mají pouze jednu složku.
Kóduje cílové třídy pomocí one-hot encoding, což je běžná praxe při práci s kategoriálními daty.
Rozděluje data na trénovací a validační sady pro evaluaci modelu.
#### Interaktivní Část pro Výběr Čísla:

Umožňuje uživateli zadat číslo od 0 do 9.
Vybere odpovídající obrázky a popisky z validační sady pro zadané číslo.
Vykreslí první obrázek z vybraných dat s odpovídajícím popiskem pro kontrolu.
#### Definice a Trénování Modelu:

Definuje konvoluční neuronový model, který se skládá z konvolučních vrstev, poolovacích vrstev a plně propojených vrstev.
Kompiluje model s optimizátorem Adam, který efektivně aktualizuje váhy během učení, kategoriální cross-entropy jako loss funkcí (měření chyby) a accuracy jako metrikou pro sledování úspěšnosti modelu.
Trénuje model na trénovacích datech a sleduje vývoj úspěšnosti a chyby na validační sadě během několika epoch.
#### Vizualizace Konvolučních Filtrů:

Získává váhy první konvoluční vrstvy modelu, což jsou filtry, které model naučil detekovat vzory v datech.
Vykresluje tyto filtry pro vizuální kontrolu toho, co model "vidí" ve vstupních datech.
#### Vizualizace Výstupů Konvolučních Vrstev:

Načítá vybraný obrázek a získává výstupy prvních šesti vrstev modelu pro tento obrázek.
Vykresluje tyto výstupy pro lepší pochopení, jak se data transformují v průběhu konvolučních vrstev.
Celkově má kód za účel demonstrovat proces vytváření, trénování a vizualizace konvolučního neuronového modelu pro rozpoznávání číslic z datasetu MNIST.

Výsledek

![image](https://github.com/0NDRS/PVA/assets/145441873/d02df71a-c69c-416e-8b66-9009fd548228)


#### 2. Rekurentní neuronové sítě (RNN)

#### Rekurentní Jednotka:

Nejzákladnější stavební jednotkou RNN je rekurentní jednotka. Tato jednotka umožňuje přenos informací z jednoho kroku sekvence na další.
V každém kroku RNN vezme vstupní hodnotu a předchozí stav a vytvoří nový stav a výstup.
#### Zpětná Propagace v Čase (BPTT):

BPTT je algoritmus používaný pro trénování RNN. Principiálně je to podobné zpětné propagaci chyby v běžných neuronových sítích, ale s tím rozdílem, že se prochází časem a aktualizují se váhy podle chyb v časových krocích.
#### Problém Explodujících a Mizejících Gradientů:

Při trénování RNN může docházet k problémům s gradientem, zejména když se gradienty buď exponenciálně zvyšují (explodující gradienty) nebo exponenciálně klesají na nulu (mizející gradienty).
Pro řešení těchto problémů byly vyvinuty varianty RNN, jako jsou LSTM (Long Short-Term Memory) a GRU (Gated Recurrent Unit), které mají lepší schopnost zacházet s dlouhodobými závislostmi v datech.
#### LSTM (Long Short-Term Memory):

LSTM je speciální typ rekurentní neuronové sítě, který byl navržen k řešení problému mizejících gradientů a umožňuje modelu lépe zachytit dlouhodobé závislosti v datech.
Má komplexní strukturu s pomocnými branami, které rozhodují, které informace mají být uchovány a které mohou být zapomenuty.
Použití Rekurentních Neuronových Sítí:
#### NLP (Natural Language Processing):

RNN jsou široce využívány v oblasti zpracování přirozeného jazyka pro překlad, generování textu a další aplikace.
#### Časové Řady:

Používají se k modelování časových řad, například při předpovídání budoucích hodnot finančních ukazatelů.
#### Řečové Rozpoznávání:

RNN jsou účinné při práci s řečovými daty, zejména při převodu řeči na text nebo naopak.
#### Generování Seznamů:

Jsou schopny generovat nové sekvence podle vzorů naučených ze vstupních dat.
#### Robotika a řízení:

Využívají se k řízení a navigaci robotů na základě vizuálních dat nebo senzorů.
Rekurentní neuronové sítě jsou mocným nástrojem pro práci se sekvenciálními daty a jsou stále důležitou součástí oblasti hlubokého učení. Jejich schopnost zachytit kontext a závislosti mezi daty je klíčová pro mnoho aplikací v oblasti umělé inteligence.

### Ukázka

![image](https://github.com/0NDRS/PVA/assets/145441873/91908349-7da6-474d-ab8a-d0d8534951d5)

Tento model je navržen pro generování textu na základě trénovacích vět. Konkrétněji, je trénován na posloupnostech slov v texte a poté schopný generovat nové textové sekvence na základě zadaného počátečního seed textu:

Tokenizace textu:

Kód začíná tokenizací trénovacích vět. Tokenizace je proces převedení textu na sekvenci čísel, kde každé slovo je přiřazeno unikátnímu číslu.
V tomto případě se používá třída Tokenizer z Kerasu.
Příprava trénovacích sekvencí:

Následuje příprava trénovacích sekvencí. Jedná se o vytváření n-gramů (posloupností slov o délce n) ze vstupních vět.
Tato sekvence se bude používat jako vstupní a výstupní data pro trénování modelu.
Padding:

Pro účely trénování sekvence musí mít stejnou délku. Proto se provádí padding na začátku každé sekvence.
Rozdělení na X a y:

Sekvence se rozdělí na vstupní část (X: všechny sloupce kromě posledního) a výstupní část (y: poslední sloupec).
Definice modelu RNN:

Vytváří se model rekurentní neuronové sítě (RNN) pomocí Kerasu. Model obsahuje vložení (embedding) pro převod čísel na vektory, LSTM vrstvu pro učení dlouhodobých závislostí a plně propojenou vrstvu s aktivací softmax pro predikci dalšího slova.
Kompilace modelu:

Model se kompiluje s použitím optimizátoru Adam a kategoriální křížovou entropií jako ztrátovou funkcí.
Trénování modelu:

Model se trénuje na připravených trénovacích datech s 100 epochami.
Generování textu:

Funkce generate_text přijímá seed text, počet generovaných slov, model, délku vstupní sekvence a tokenizer.
Během generování se postupně přidávají nová slova k původnímu textu na základě pravděpodobností predikovaných modelem.
Teplota ovlivňuje míru "kreativity" generovaného textu. Vyšší teplota dává větší váhu nejistým předpovědím.
Generování a výpis výsledků:

Závěrečná část kódu generuje text pro různé seed texty a vypisuje výsledky.

Výsledek

Vidíme, že model vygeneroval zcela nové věty na základě poskytlých dat. Věty nedávají smysl jelikož se jedná o zjednodušený model.

![image](https://github.com/0NDRS/PVA/assets/145441873/68d8331d-5df6-4194-9221-2e23f5501c63)


### 3. Hluboké autoenkódery

#### 1. Co jsou Hluboké Autoenkódery:

Hluboké autoenkódery jsou specifický typ umělé neuronové sítě, který se skládá z enkódovací a dekódovací části.
Tyto sítě mají schopnost učit se efektivní reprezentaci vstupních dat a jsou často používány pro redukci dimenze, odstranění šumu a extrakci významných rysů v datech.
#### 2. Struktura Hlubokých Autoenkóderů:

Enkódovací část snižuje dimenzi vstupních dat na kompaktní reprezentaci, zvanou latentní prostor.
Latentní prostor obsahuje důležité rysy a vzory z původních dat.
Dekódovací část obnovuje data z latentního prostoru do původní dimenze.
#### 3. Vícevrstvý Autoenkóder:

Hluboký autoenkóder má více vrstev než tradiční autoenkóder.
Tato vrstvená struktura umožňuje modelu učit se hierarchie abstrakcí a reprezentací v datech.
#### 4. Trénink Hlubokých Autoenkóderů:

Trénink hlubokých autoenkóderů se provádí pomocí metody zpětného šíření chyby (backpropagation) a optimalizačních algoritmů, např. stochastického gradientového sestupu (SGD).
#### 5. Redukce Dimenze:

Hluboké autoenkódery jsou často používány k redukci dimenze dat, což umožňuje zachování důležitých informací a odstranění redundantních nebo nepodstatných prvků.
#### 6. Odstranění Šumu:

Autoenkódery mají schopnost odstranit šum z dat, což je užitečné v oblastech, kde jsou data zkreslena nebo obsahují chyby.
#### 7. Aplikace Hlubokých Autoenkóderů:

Hluboké autoenkódery jsou používány v různých oblastech, včetně počítačového vidění, zpracování přirozeného jazyka, analýzy dat a generativního modelování.
#### 8. Výzvy:

Trénink hlubokých autoenkóderů může být náročný a vyžaduje velké množství dat.
Správná volba architektury modelu a hyperparametrů je klíčová pro úspěšné učení.
#### 9. Generativní Schopnosti:

Hluboké autoenkódery mají generativní schopnosti, což znamená, že mohou generovat nová data podobná těm, na kterých byly trénovány.
#### 10. Sdílení Vah (Weight Sharing):

Váhy enkódovací a dekódovací části jsou obvykle sdíleny, což zvyšuje kapacitu modelu a umožňuje efektivní reprezentaci dat.

#### Příklad:

Pro rozpoznávání objektů ve fotografiích jsou potřeba tisíce označených obrázků pro efektivní trénink modelu.
Tím jsme si přiblížili klíčové principy a typy Deep Learningu, a to včetně detailnějšího pohledu na jednotlivé modely. Dále porovnáme tuto technologii s Machine Learningem.

Ukázka:

![image](https://github.com/0NDRS/PVA/assets/145441873/2001c9e7-6856-4ba7-8237-afd092d0c551)

Zde vidíme hluboký autoenkóder, kdy neuronoé sítě se učí z dat a následně mnimalizují šum:

Import knihoven:

Importujeme TensorFlow a další potřebné knihovny.
Importujeme vrstvy a nástroje z Keras.
#### Generování umělých dat (pro demonstrační účely):

Vytváříme umělá data pro trénink autoenkóderu. Každý řádek reprezentuje jeden vzor.
Definice hlubokého autoenkóderu:

Vytváříme funkci deep_autoencoder, která vytváří model hlubokého autoenkóderu pomocí Keras.
Model má enkóder a dekóder, s postupně se zmenšujícím a zvětšujícím počtem neuronů ve skrytých vrstvách.
Příprava dat a normalizace:

Normalizujeme vstupní data do rozsahu [0, 1], což je častá praxe pro neuronové sítě.
Vytvoření a trénování modelu:

Vytváříme instanci autoenkóderu pomocí definované funkce.
Kompilujeme model s optimizerem 'adam' a loss funkcí 'mean_squared_error'.
Trénujeme model na umělých datech.
Vizualizace výsledků:

Vytváříme vizualizaci vstupních a rekonstruovaných obrázků.
Zobrazujeme několik příkladů vedle sebe pro porovnání kvality rekonstrukce.

Výsledek

Horní řada je vygenerovaný obraz a spodní je upravená verze se sníženým šumem

![image](https://github.com/0NDRS/PVA/assets/145441873/9411b30c-8890-49c7-b813-5e09b8d2b0cb)



## Jak fungují vsrtvy v Deep Learingu (DL)

V Deep Learning (Hlubokém učení) jsou vrstvy základními stavebními bloky neuronových sítí. Každá vrstva má svou specifickou roli v extrakci a transformaci dat. Podívejme se na několik klíčových vrstev v Deep Learning:

### 1. Nejzákladnější stavební bloky:

Vstupní vrstva (Input Layer): Přijímá vstupní data a předává je dalším vrstvám. Každý neuron v této vrstvě reprezentuje jednu vstupní vlastnost.
Výstupní vrstva (Output Layer): Produkuje výstupní predikce nebo klasifikace. Počet neuronů v této vrstvě závisí na typu úlohy (regrese, klasifikace, generace atd.).
### 2. Skryté vrstvy (Hidden Layers):

Propojení vrstev: Neurony v jedné vrstvě jsou propojeny s neurony v následující vrstvě. Každá spojení má přidělenou váhu, která se mění během tréninku.
Aktivační funkce: Každý neuron v skryté vrstvě používá aktivační funkci k tomu, aby zpracoval vstup a generoval výstup. Populární aktivační funkce zahrnují ReLU (Rectified Linear Unit), sigmoid a tanh.
### 3. Funkce skrytých vrstev:

Extrahování hierarchií: Skryté vrstvy umožňují modelu učit se hierarchie abstrakcí a složitostí v datech. Každá vrstva může zachytit různé úrovně reprezentací.
### 4. Konvoluční vrstvy v konvolučních neuronových sítích (CNN):

Lokální vzory: Konvoluční vrstvy jsou efektivní při zachycování lokálních vzorů ve vstupních datech, například v obrazech. Používají konvoluci a pooling operace.
### 5. Rekurentní vrstvy v rekurentních neuronových sítích (RNN):

Zpětná vazba v čase: Rekurentní vrstvy umožňují modelům pracovat s posloupnostmi dat a mají schopnost udržovat vnitřní stav časové závislosti.
### 6. Normalizační vrstvy (Batch Normalization):

Normalizace dat: Normalizační vrstvy přispívají k stabilizaci a rychlejší konvergenci modelu tím, že normalizují výstupy vrstev.
### 7. Funkce dropout:

Prevence přetrénování: Dropout je technika, kde náhodně vynecháváme některé neurony během tréninku, což pomáhá zabránit přetrénování modelu.
### 8. Loss (Ztrátová) funkce:

Hodnota chyby: Loss funkce hodnotí rozdíl mezi predikcemi modelu a skutečnými hodnotami. Cílem je minimalizovat tuto hodnotu během tréninku.
### 9. Optimalizace:

Optimalizační algoritmy: Algoritmy, jako je SGD (Stochastic Gradient Descent), Adam nebo RMSprop, slouží k aktualizaci vah modelu tak, aby minimalizovaly hodnotu loss funkce.
### 10. Zpětná propagace chyby:

Učení pomocí chyby: Zpětná propagace chyby (backpropagation) je algoritmus, který se používá k aktualizaci vah v modelu na základě chyby predikce a skutečné hodnoty.
### 11. Regularizace:

Prevence přetrénování: Techniky jako L1 nebo L2 regularizace slouží k omezení velikosti vah a prevenci přetrénování modelu.
### 12. Overfitting (Přetrénování):

Problém přetrénování: Přetrénování může nastat, když model příliš dobře zapamatuje trénovací data, ale nemá schopnost generalizovat na nová data.
### 13. Vstupní normalizace:

Normalizace vstupu: Normalizace vstupních dat pomáhá stabilizovat proces trénování a urychluje konvergenci modelu.
### 14. Výstupní aktivační funkce:

### Podoba výstupu: Aktivační funkce výstupní vrstvy se volí v závislosti na typu úlohy (např. sigmoid pro binární klasifikaci, softmax pro multiklasovou klasifikaci).
Deep Learning využívá těchto stavebních bloků pro vytváření složitých modelů, které mají schopnost reprezentovat, generalizovat a interpretovat různé druhy dat. Hierarchická struktura a adaptabilita těchto vrstev jsou klíčové pro úspěch v oblastech jako počítačové vidění, zpracování přirozeného jazyka, hlasové rozpoznávání a mnoho dalších.

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
