# ⚖️ Políticas de Governança, Ética e Reprodutibilidade em IA

Este documento define os requisitos de conformidade ética, transparência e reprodutibilidade do projeto.

---

## 1. Reprodutibilidade e Determinismo
* Todas as inicializações de sementes aleatórias devem utilizar `semente=42` (`random_state=42` / `tf.random.set_seed(42)`).
* Todos os pacotes devem estar versionados em `requirements.txt`.
* A divisão de dados deve ser sempre estratificada (`stratify=y`) para garantir equilíbrio entre as 10 classes.

---

## 2. Prevenção de Data Leakage e Incerteza
* Os normalizadores e scalers devem ser ajustados exclusivamente com os dados de treino (`fit` no treino, `transform` no teste).
* Os experimentos de generalização extrema (OOD) devem analisar a calibração de confiança da função Softmax para mitigar o risco de **Falsa Certeza (*Overconfidence*)**.

---

## 3. Rastreabilidade e Auditoria
* Todas as execuções de treinamento devem registrar metadados (timestamp, hiperparâmetros, tempos de CPU/GPU e métricas) no banco de dados estruturado e em arquivos CSV.
