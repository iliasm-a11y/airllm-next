# Cadre théorique — exécution exacte d'un MoE géant hors-cœur

## T0. Modèle et notations

- Modèle cible : $L$ couches, dont $L_m$ couches MoE. Chaque couche MoE $\ell$ possède $N$ experts routés de taille $s$ octets (format stocké) et sélectionne $k$ experts par token. GLM-5.3 : $L=78$, $L_m=75$, $N=256$, $k=8$, $s=37{,}75$ Mo en FP8. Qwen3.8 : $L=L_m=92$, $N=512$, $k=10$, $s=50{,}3$ M paramètres par expert.
- $A$ : octets **toujours actifs** par passe (attention, indexeur DSA ou projections Gated DeltaNet, experts partagés, couches denses, routeurs, normes, tête de sortie).
- Une **passe** $p$ est un passage avant qui traite un ensemble de positions $S_p$, de taille $n_p=|S_p|$. Sans spéculation $n_p=1$ ; spéculation linéaire de profondeur $K$ : $n_p=K+1$ ; arbre de brouillon : $n_p$ nœuds ; prefill : $n_p=P$ ; multi-flux : $n_p=B$.
- Routage : $E_\ell(t)\subset\{1..N\}$ avec $|E_\ell(t)|=k$. Union d'une passe : $U_\ell(S)=\bigcup_{t\in S}E_\ell(t)$.
- Mémoire : budget $M$ pour les poids. Ensemble résident $R=R_A\cup C$ (poids toujours actifs résidents, et cache d'experts $C_\ell$).
- Stockage : périphériques $d$ de débit $\beta_d$ (octets/s) et de capacité $\mathrm{Cap}_d$. Taux de compression sans perte $\rho\in(0,1]$. Débit de décodage $\delta$ (octets décompressés/s). Débit de calcul $\Phi$.

**Octets lus en stockage pendant une passe :**
$$B_p=\rho\Big[(A-|R_A|)+s\sum_{\ell}\big|U_\ell(S_p)\setminus C_\ell(p)\big|\Big]+B^{KV}_p .$$

**Temps d'une passe** (recouvrement idéal des E/S, du calcul et du décodage) :
$$T_p\;\ge\;\max\Big\{\tfrac{B_p}{\beta_{\text{eff}}},\;\tfrac{F_p}{\Phi},\;\tfrac{B_p}{\rho\,\delta}\Big\}.$$

**Débit (théorème de renouvellement-récompense).** Sous stationnarité, les passes forment un processus régénératif, et chaque passe émet $\tau_p\ge1$ tokens. Le temps moyen par token émis vaut $\mathbb E[T_p]/\mathbb E[\tau_p]$. Les **octets par token accepté** (OTA) valent
$$\mathrm{OTA}=\frac{\mathbb E[B_p]}{\mathbb E[\tau_p]}.$$
L'OTA est la grandeur à minimiser. Trois quantités sont liées sans être identiques : la réduction de RAM agit sur $|R|$, la réduction des octets transférés agit sur l'OTA, et l'accélération effective est le rapport des temps. Ce dernier n'égale le rapport des OTA que si le régime reste limité par la bande passante, c'est-à-dire si $B_p/\beta_{\text{eff}}$ domine.

---

## T1. Théorème 1 — Taxe d'exactitude (borne inférieure du trafic)

**Hypothèse G (généricité).** Pour chaque expert $e$, et pour chaque bit de $W_e$, il existe une entrée $x$ telle que la valeur calculée de $\mathrm{FFN}_e(x)$, en arithmétique de référence, change quand ce bit change. Pour des poids entraînés, c'est vrai sauf sur un ensemble de mesure nulle. C'est l'hypothèse « pire cas sur les entrées » : l'algorithme doit rester exact pour **toute** invite.

**Énoncé.** Soit un algorithme qui, pour chaque token émis $t$ et chaque couche $\ell$, calcule exactement $\mathrm{FFN}_e(x_\ell(t))$ pour tout $e\in E_\ell(t)$. Alors :
1. Dans chaque passe $p$, toute l'information de chaque expert $e\in U_\ell(S_p)$ est soit résidente au début de $p$, soit lue pendant $p$.
2. Le trafic de la passe $p$ à la couche $\ell$ vérifie $\;\mathrm{trafic}_\ell(p)\ \ge\ \rho_{\min}\,s\,\big|U_\ell(S_p)\setminus C_\ell(p)\big|$, où $\rho_{\min}s$ est l'entropie, en octets, d'un expert sous le meilleur modèle sans perte.
3. Sur une fenêtre de passes $\mathcal W$ avec une capacité RAM de $c_\ell$ experts à la couche $\ell$ :
$$\mathrm{trafic}_\ell(\mathcal W)\ \ge\ \rho_{\min}\,s\Big(\Big|\bigcup_{p\in\mathcal W}U_\ell(S_p)\Big|-c_\ell\Big)^+ .$$

**Preuve.**
1. Supposons qu'un bit $b$ de $W_e$ ne soit ni résident ni lu. La transcription complète de l'exécution est alors indépendante de $b$. Par G, il existe une entrée pour laquelle basculer $b$ change la sortie exacte. L'algorithme produirait la même sortie dans les deux cas, ce qui contredit l'exactitude.
2. Par le théorème du codage de source, transférer une variable aléatoire sans perte demande au moins son entropie. Les experts sont mutuellement quasi indépendants : l'information mutuelle inter-experts mesurée est quasi nulle (Annexe Entropie). On a donc $H(W_{U\setminus C})=\sum_{e}H(W_e)=|U\setminus C|\,\rho_{\min}s$.
3. Au début de $\mathcal W$, la RAM contient au plus $c_\ell$ experts distincts de la couche $\ell$. Chaque autre expert distinct utilisé dans $\mathcal W$ doit être lu au moins une fois. $\square$

**Lecture.** Cette borne vaut pour **tout** algorithme exact, spéculatif ou non. C'est la borne de pagination de Belady appliquée à des requêtes regroupées par passes. Il en découle que seuls **quatre leviers exacts** existent :
- (i) diminuer les octets par paramètre, au mieux jusqu'à l'entropie ($\rho\to\rho_{\min}$) ;
- (ii) servir des requêtes depuis la RAM (résidence et cache) ;
- (iii) faire partager une même lecture à plusieurs tokens (passes multi-positions) ;
- (iv) augmenter $\beta_{\text{eff}}$ (matériel).

Toute méthode qui prétend dépasser cette borne modifie le calcul : elle n'est pas exacte.

> Les symétries de permutation des neurones intermédiaires réduisent l'information d'un expert de $\log_2(2048!)\approx 2{,}4$ Ko seulement, contre 37,75 Mo par expert. La borne n'en est pas affectée.

---

## T2. Théorème 2 — Borne de la médiante : spéculation et experts routés

Dans une passe spéculative linéaire de profondeur $K$, on traite les $K+1$ positions $t_0$ (dernier token émis) et $t_1..t_K$ (brouillons). On note :
- $\pi_j=\Pr(\text{les brouillons }1..j\text{ sont tous acceptés})$, avec $\pi_0=1$. Les tokens émis par passe valent $\tau(K)=\sum_{j=0}^{K}\pi_j$ (tokens acceptés, plus la correction ou le bonus).
- $\nu_j=\mathbb E\,|E_\ell(t_j)\setminus\bigcup_{i<j}E_\ell(t_i)|/k$, la **nouveauté marginale** de la position $j$ dans la passe, avec $\nu_0=1$.

Par linéarité, $\mathbb E|U_\ell(S_p)|=k\sum_{j=0}^{K}\nu_j$. Toutes les positions sont chargées, qu'elles soient acceptées ou non. Hors cache, l'amortissement du trafic routé par rapport au décodage simple vaut
$$G_r(K)=\frac{\tau(K)}{\sum_{j=0}^{K}\nu_j}=\frac{\sum_{j}\pi_j}{\sum_j\nu_j}.$$

**Théorème.** $\displaystyle \min_{0\le j\le K}\frac{\pi_j}{\nu_j}\ \le\ G_r(K)\ \le\ \max_{0\le j\le K}\frac{\pi_j}{\nu_j}\ \le\ \Gamma:=\sup_{j\ge0}\frac{\pi_j}{\nu_j}.$

**Preuve.** Inégalité de la médiante : $\sum_j\pi_j=\sum_j(\pi_j/\nu_j)\,\nu_j\le\max_j(\pi_j/\nu_j)\sum_j\nu_j$. L'autre inégalité est symétrique. $\square$

**Corollaires.**
1. *Critère d'utilité.* Puisque $\pi_0/\nu_0=1$, la spéculation ne réduit le trafic routé ($G_r>1$) que s'il existe $j\ge1$ avec $\pi_j>\nu_j$. Il faut que la probabilité d'accepter le $j$-ième brouillon dépasse sa nouveauté en experts. Si $\nu_j\ge\pi_j$ pour tout $j$, aucune profondeur ne réduit le trafic routé, quelle que soit la qualité du brouillon.
2. *Plafond.* Avec une acceptance géométrique $\pi_j=\alpha^j$ et une nouveauté décroissante, le supremum est en général atteint en $j=1$ : $G_r\le\max(1,\alpha/\nu_1,\alpha^2/\nu_2,\dots)$. Par exemple, $\alpha=0{,}8$ et $\nu_1=0{,}65$ donnent $G_r\le1{,}23$.
3. *Arbres.* Pour un arbre de brouillon, chaque nœud $n$ a une probabilité d'atteinte $\pi_n$ et une nouveauté $\nu_n$, d'où $G_r\le\max_n\pi_n/\nu_n$. Les branches secondaires ont un $\pi_n$ faible et un $\nu_n$ voisin de $\nu_1$ : elles **dégradent** le rapport. Pour des MoE à routage fin en régime de passes étroites, les chaînes dominent les arbres.
4. *Gain total.* Soit $A_{nr}$ les octets toujours actifs non résidents, $R$ les octets routés par token en décodage simple et $d$ le coût du brouillon par token proposé. Alors
$$\mathrm{OTA}(K)=\frac{A_{nr}+R\sum_{j\le K}\nu_j+K d}{\tau(K)},\qquad G(K)=\frac{A_{nr}+R}{\mathrm{OTA}(K)} .$$
La partie toujours active est amortie d'un facteur $\tau$, la partie routée d'un facteur $G_r$ au plus. La spéculation est donc **beaucoup plus rentable pour Qwen3.8**, où $A$ représente environ la moitié des octets actifs, que pour GLM-5.3.

---

## T3. Théorème 3 — Saturation de l'union (prefill et multi-flux)

Soit $n$ tokens de routages indépendants, par exemple issus de flux différents, avec des probabilités marginales d'inclusion $p_e$ telles que $\sum_ep_e=k$. Alors
$$\mathbb E|U_\ell|=\sum_{e}\big[1-(1-p_e)^n\big]\ \le\ N\big[1-(1-k/N)^n\big].$$

**Preuve.** L'égalité vient de l'indépendance. L'inégalité vient de la convexité de $x\mapsto(1-x)^n$ (Jensen) : $\frac1N\sum_e(1-p_e)^n\ge(1-k/N)^n$. $\square$

Le modèle uniforme indépendant est donc le **pire cas** pour l'union. Une popularité inégale ou une corrélation temporelle positive la réduisent. On mesure les courbes empiriques. Dans le modèle uniforme, la fraction d'experts touchés atteint $f$ pour $n_f\approx (N/k)\ln\frac1{1-f}$ :

| | $N/k$ | $n_{0,9}$ | $n_{0,99}$ |
|---|---:|---:|---:|
| GLM-5.3 | 32 | 74 | 147 |
| Qwen3.8 | 51,2 | 118 | 236 |

**Conséquence physique.** Un prefill de quelques centaines de tokens coûte à peu près une lecture **complète** du modèle. Pour GLM-5.3 en FP8, cela représente $\approx 755$ Go, soit environ 250 s à 3 Go/s. Par le Théorème 1, aucun algorithme exact ne fait mieux sans cache, à $\rho$ et $\beta$ donnés. Chaque tour de conversation qui ajoute plus d'une centaine de tokens (message, sortie d'outil) paie ce prix. La persistance du cache KV entre les tours est donc indispensable, de même que le regroupement des entrées en attente dans un même balayage.

---

## T4. Théorème 4 — Optimalité « tout ou presque rien » de la taille de passe

On note :
- $c(n)$ les octets attendus d'une passe à $n$ positions spéculatives, croissante et bornée par $C_{\max}=A_{nr}+s\,L_mN$ ;
- $\tau(n)$ l'espérance de tokens émis du meilleur arbre à $n$ nœuds, croissante ;
- $n\le n_{\max}$ la contrainte de calcul (temps de calcul de la passe au plus égal à son temps d'E/S).

**Théorème.** Soit $\varepsilon\in(0,1)$ et $n_\varepsilon$ tel que $c(n)\ge(1-\varepsilon)C_{\max}$ pour $n\ge n_\varepsilon$. L'efficacité $\eta(n)=\tau(n)/c(n)$ vérifie, pour tout $n\in[n_\varepsilon,n_{\max}]$,
$$\eta(n)\le\frac{\eta(n_{\max})}{1-\varepsilon}.$$
L'optimum global est donc soit une passe **étroite** ($n<n_\varepsilon$, en pratique $K\le2$), soit le **balayage maximal** $n_{\max}$, à un facteur $1/(1-\varepsilon)$ près.

**Preuve.** Pour $n\ge n_\varepsilon$, on a $\eta(n)=\tau(n)/c(n)\le\tau(n_{\max})/((1-\varepsilon)C_{\max})$. Comme $c(n_{\max})\le C_{\max}$, il vient $\tau(n_{\max})/C_{\max}\le\eta(n_{\max})$. $\square$

**Règle de décision.** Balayer si et seulement si $\tau(n_{\max})\,c_{\text{étroit}}>C_{\max}\,\tau_{\text{étroit}}$. Pour GLM-5.3 en FP8 ($C_{\max}\approx 755$ Go, $c_{\text{étroit}}\approx 30$ Go pour $\tau_{\text{étroit}}\approx1{,}2$), le balayage n'est rentable que si un arbre de $n_{\max}$ nœuds fait accepter environ **25 tokens ou plus** par passe. C'est plausible pour des sorties très prévisibles (édition de code, recopie, JSON répétitif), improbable pour de la prose libre. Le balayage est une **passe de prefill dont la fin est un arbre de brouillon** : il réutilise exactement le chemin couche par couche d'AirLLM. Corollaire immédiat : tout prefill étant déjà un balayage saturé, il faut **y greffer gratuitement** un arbre de réponse spéculatif, jusqu'à $n_{\max}$.

---

## T5. Théorème 5 — Résidence : inversion de priorité

Garder résident un bloc de poids $b$, de taille $|b|$ et de probabilité d'usage par passe $q_b$, économise $\rho\,q_b|b|$ octets par passe. Maximiser l'économie sous $\sum|b|\le M$ est un sac à dos dont la densité de valeur vaut $q_b$. L'optimum fractionnaire trie les blocs par $q_b$ décroissant. Comme $|b|\ll M$, l'écart à l'optimum entier est d'au plus un bloc.

**Énoncé.** Les poids toujours actifs ($q=1$) précèdent tout expert, car $q_e=\Pr(e\in U_\ell(S_{p+1})\mid\text{historique})<1$. Un expert ne devient prioritaire que si sa probabilité de réutilisation à la passe suivante approche 1.

**Conséquence.** Pour GLM-5.3 en FP8 ($A\approx19$ Go $>M\approx9$–$10$ Go) comme pour Qwen3.8 ($A\approx47$ Go en FP8), la politique optimale ne garde **aucun expert résident** entre les passes. Toute la RAM disponible va aux poids toujours actifs, stockés au format compact (FP8, voire compressé sans perte et décodé à la volée). C'est l'inverse des caches d'experts conçus pour les GPU de 24 à 80 Go. Le critère secondaire, à densité égale, consiste à rendre résidents en priorité les routeurs et les entrées d'attention : on supprime ainsi la latence de découverte des routes.

---

## T6. Théorème 6 — Agrégation de périphériques

On a $T^{IO}_p=\max_dB_{p,d}/\beta_d\ \ge\ \sum_dB_{p,d}/\sum_d\beta_d$, avec égalité si et seulement si $B_{p,d}\propto\beta_d$. L'accélération par rapport au seul périphérique 1 est au plus $\sum_d\beta_d/\beta_1$. Le placement optimal en espérance est un programme linéaire. On note $x_{b,d}\in[0,1]$ la fraction du bloc $b$ placée sur $d$ et $f_b$ la fréquence d'accès du bloc. On minimise $T$ sous les contraintes $\sum_bx_{b,d}f_b|b|\le T\beta_d$, $\sum_bx_{b,d}|b|\le\mathrm{Cap}_d$ et $\sum_dx_{b,d}=1$. L'équilibre doit être tenu **par couche**, faute de quoi un périphérique attend l'autre à chaque couche. La réplication des blocs les plus chauds permet un équilibrage dynamique à l'exécution.

---

## T7. Théorème 7 — Débit multi-flux

Avec $B$ flux indépendants regroupés par passe, $c(B)=A_{nr}+s\sum_\ell\mathbb E|U_\ell(B)|\le A_{nr}+sL_mN[1-(1-k/N)^B]$. On obtient le débit agrégé $X(B)=B/T(c(B))$ et la latence par token d'un flux $T(c(B))$. $X$ croît avec $B$, jusqu'aux limites de calcul et de KV. La latence croît vers la durée d'un balayage complet. C'est un gain **exact** en débit, pas en latence.

---

## T8. Proposition 8 — Codage sans perte

Si le décodage s'exécute en parallèle des E/S avec $\delta\ge\beta_{\text{eff}}/\rho$, l'accélération du régime limité par les E/S vaut exactement $1/\rho$. On a par ailleurs $\rho\ge\rho_{\min}=H/(8s)$. Le codage sans perte agit aussi sur la **capacité** : un checkpoint tient sur un disque de capacité $\mathrm{Cap}$ si et seulement si $\rho\cdot\text{taille}\le\mathrm{Cap}$.

**Impossibilité (Qwen3.8 BF16 sur 2 To).** En BF16, le signe et les 7 bits de mantisse sont quasi équiprobables, et l'exposant a une entropie d'environ 2,5 bits au moins. Il faut donc au moins $\approx10{,}5$ bits par paramètre, soit $\ge 3{,}2$ To pour $2{,}446\cdot10^{12}$ paramètres. Le checkpoint BF16 original ne peut tenir sur 2 To par **aucun** codage exact. La valeur mesurée est donnée en annexe.

---

## T9. Proposition 9 — Retour arrière exact des états récurrents par rejeu

Pour une couche Gated DeltaNet, $S_t=f(S_{t-1},q_t,k_t,v_t,g_t,\beta_t)$. Pendant la passe de vérification, on conserve $S_{t_0}$ (un seul instantané) et, pour chaque position, le tuple des entrées projetées $(k_t,v_t,g_t,\beta_t)$. Après la décision d'acceptation au rang $a$, on recalcule $S_{t_0+a}$ en réappliquant la même récurrence, avec le même noyau séquentiel et dans le même ordre d'opérations. Le résultat est identique bit à bit à l'état qu'aurait produit le décodage séquentiel. Coût mémoire : un instantané (≈ 579 Mo pour Qwen3.8 en FP32) plus ≈ 5 Mo par position, au lieu d'un instantané par position. Coût de calcul : $O(a\,H\,d_kd_v)$ par couche, négligeable. Condition : les projections de la passe de vérification doivent être invariantes au batch (T10).

---

## T10. Proposition 10 — Invariance au batch, donc égalité bit à bit

Supposons que chaque noyau calcule chaque ligne de sortie avec un ordre de réduction indépendant du nombre de lignes traitées ensemble, et que la combinaison MoE somme les experts dans un ordre canonique. Alors les logits de chaque position d'une passe de vérification sont **identiques bit à bit** à ceux du décodage séquentiel. La spéculation gloutonne émet exactement la même séquence, ex aequo compris. La version stochastique (échantillonnage spéculatif par rejet) échantillonne exactement la même loi. Sans invariance au batch, les écarts restent bornés par l'erreur de réordonnancement flottant $\gamma_n\sum_i|x_iw_i|$, avec $\gamma_n\approx nu$. Ils peuvent alors inverser des quasi-égalités d'argmax ou de top-k du routeur : c'est une différence numérique **caractérisée**, et non une dégradation du modèle.

---

## T11. Proposition 11 — Plafond de vitesse en flux unique exact

En combinant T1 à T8, en flux unique exact et en régime de passes étroites :
$$T_{\text{token}}\ \ge\ \frac{\rho_{\min}}{\beta_{\text{eff}}}\Big[\frac{A-M_A}{\tau_{\max}}+\frac{R\,(1-h^{\mathrm{OPT}})}{\Gamma}\Big].$$
Ici $h^{\mathrm{OPT}}$ est le taux de succès de Belady pour la capacité restante (quasi nulle par T5), $\Gamma$ le plafond de T2 et $\tau_{\max}\le1/(1-\alpha)$. Les valeurs numériques, issues des mesures, figurent en section Résultats. Ce plafond explique pourquoi un facteur 10 n'est pas accessible avec un seul SSD externe, et pourquoi un facteur 100 exigerait $\beta_{\text{eff}}$ de l'ordre de la bande passante DRAM.
