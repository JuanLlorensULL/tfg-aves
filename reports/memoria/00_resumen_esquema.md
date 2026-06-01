# Esquema del Resumen / Abstract (cap. 0)

> Fuente de verdad del esquema del resumen. Redactado tras cerrar caps. 1–9.
> La plantilla deja `\chapter*{Resumen}` + `\chapter*{Abstract}` (máx. 2 págs c/u),
> sin subsecciones. El resumen es prosa corrida (4 párrafos) que sintetiza toda la
> memoria; el abstract lo refleja en inglés.

## Decisiones del autor (2026-06-01)
1. **Sin "Palabras clave" / "Keywords"** (se ciñe a la plantilla original).
2. **Mínimo de cifras**: resumen puramente cualitativo, sin números; el detalle
   cuantitativo vive en los capítulos.
3. **~1 página densa** por idioma (la plantilla permite hasta 2).

## Restricciones
- Una página densa por idioma. Sin relleno.
- Cero rayas. Sin jerga "O1/O2…". Tono investigador, no apologético.
- Sin cifras concretas (decisión del autor).
- No lleva `\cite` (es un resumen); las fuentes se citan en la intro y preliminares.

## Estructura del Resumen (ES) — 4 párrafos

**P1 — Contexto y problema (3–4 líneas).**
La geolocalización GPS ha desplazado el cuello de botella del estudio del movimiento
animal de registrar a predecir. Pregunta del trabajo: dada la posición de hoy, ¿dónde
estará mañana? Caso de estudio: la gaviota sombría (*Larus fuscus*), migrante de larga
distancia con fenología nítida, sobre datos públicos de seguimiento.

**P2 — Enfoque híbrido y método (5–7 líneas).**
Se diseña y evalúa un enfoque híbrido que combina modelos probabilísticos y aprendizaje
supervisado. A partir de los registros GPS en bruto se construye una secuencia diaria de
posiciones por individuo. Sobre ella: (a) una cadena de Markov visible fija la baseline
interpretable y el criterio de evaluación común (log-loss) frente a la persistencia;
(b) un modelo oculto de Markov infiere para cada día un estado de comportamiento
(estacionario o en migración), coherente con la fenología de la especie; (c) ese estado
alimenta a modelos de aprendizaje supervisado (Random Forest, XGBoost, LightGBM), que
predicen el desplazamiento usando solo información disponible antes del instante predicho.
Se añade una herramienta de cartografía interactiva para visualizar y comparar las
predicciones sobre el mapa.

**P3 — Hallazgo central y resultados (5–7 líneas).**
Un hallazgo recorre todo el trabajo: la regla trivial de persistencia ("mañana donde hoy")
es muy difícil de superar, porque la especie pasa la mayoría de los días sin desplazarse.
El valor no está en un salto de exactitud, sino en calibrar mejor la incertidumbre
(banda de predicción con cobertura próxima a la nominal), en aislar el régimen de
migración (donde la predicción sí aporta) y en caracterizar el techo estructural de
predecir a un día con información local. La formulación de regresión de cuantiles iguala
a la persistencia en agregado y entrega una banda de incertidumbre que la herramienta
lleva al mapa. (Sin cifras explícitas, por decisión del autor.)

**P4 — Aportación y cierre (2–3 líneas).**
La contribución es triple: la caracterización rigurosa de un límite estructural, una
herramienta interactiva de comparación, y un criterio metodológico (la coherencia
biológica como validación). Trabajo futuro: modelos de secuencia, viento y predicción
multi-paso.

## Estructura del Abstract (EN)
Espejo fiel del resumen ES (mismos 4 párrafos, misma terminología técnica en inglés:
visible Markov chain, hidden Markov model, supervised learning, quantile regression,
persistence baseline, log-loss). Sin keywords (decisión del autor).
