# Agent Memory 项目

## 对齐清单

记录 Agent Memory 实验中需要统一固定的公共设置。后续方法迭代时，默认只改变 memory 方法本身；下面这些设置应保持一致。

### 1. 模型协议

- Answer 模型配置

```yaml
answer:
  name: gpt-4o-mini
  base_url: https://api.openai.com/v1
  api_key_env: OPENAI_API_KEY
  temperature: 0.0
  max_tokens: 8192
  thinking: default
```

- Embedding 模型配置

```yaml
embedding:
  name: text-embedding-3-small
  base_url: https://api.openai.com/v1
  api_key_env: OPENAI_API_KEY
  dims: 1536
  max_input_bytes: 8192
  normalize: true
```

- Judge 模型配置

```yaml
judge:
  name: gpt-4o-mini
  base_url: https://api.openai.com/v1
  api_key_env: OPENAI_API_KEY
  temperature: 0.0
  max_tokens: 8192
  timeout_seconds: 120
  max_retries: 2
  thinking: default
```

### 2. Prompt Template 协议

- Answer template

普通问题的 answer prompt：
```text
Answer the user's question based on the provided context.

User Question: {query}

Relevant Context: {context_str}

Requirements:
1. First, think through the reasoning process
2. Then provide a very CONCISE answer (short phrase about core information)
3. Answer must be based ONLY on the provided context
4. All dates in the response must be formatted as 'DD Month YYYY' but you can output more or less details if needed
5. Return your response in JSON format

Output Format:
{
  "reasoning": "Brief explanation of your thought process",
  "answer": "Concise answer in a short phrase"
}

Now answer the question. Return ONLY the JSON, no other text.
```

LoCoMo Category 5 的 answer prompt：
```text
Based on the context below, answer the following question.

Context:{context_str}

Question: {question}

Select the correct answer from the following two options. If the given answer is wrong or not answerable based on the context, you should choose "Not mentioned in the conversation".

Option A: {option_a}
Option B: {option_b}

Requirements:
1. Choose the option that best matches the context
2. If neither answer is supported by the context, or if the provided specific answer is incorrect, choose "Not mentioned in the conversation"
3. Return your response in JSON format

Output Format:
{
  "reasoning": "Brief explanation of your choice",
  "answer": "Your selected answer"
}

Return ONLY the JSON, no other text.
```

- Judge template

LoCoMo:
```text
Your task is to label an answer to a question as ’CORRECT’ or ’WRONG’. You will be given the following data:
    (1) a question (posed by one user to another user), 
    (2) a ’gold’ (ground truth) answer, 
    (3) a generated answer
which you will score as CORRECT/WRONG.

The point of the question is to ask about something one user should know about the other user based on their prior conversations.
The gold answer will usually be a concise and short answer that includes the referenced topic, for example:
Question: Do you remember what I got the last time I went to Hawaii?
Gold answer: A shell necklace
The generated answer might be much longer, but you should be generous with your grading - as long as it touches on the same topic as the gold answer, it should be counted as CORRECT. 

For time related questions, the gold answer will be a specific date, month, year, etc. The generated answer might be much longer or use relative time references (like "last Tuesday" or "next month"), but you should be generous with your grading - as long as it refers to the same date or time period as the gold answer, it should be counted as CORRECT. Even if the format differs (e.g., "May 7th" vs "7 May"), consider it CORRECT if it's the same date.

Now it's time for the real question:
Question: {question}
Gold answer: {gold_answer}
Generated answer: {generated_answer}

Return ONLY a valid JSON object in the following format:
{
  "label": "CORRECT"
}

The value of "label" must be exactly "CORRECT" or "WRONG". Do not include any explanation or extra text.
```

LongMemEval:

`single-session-user` / `single-session-assistant` / `multi-session`:
```text
I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no.

Question: {question}

Correct Answer: {answer}

Model Response: {response}

Is the model response correct? Answer yes or no only.
```

`temporal-reasoning`:
```text
I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. In addition, do not penalize off-by-one errors for the number of days. If the question asks for the number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., predicting 19 days when the answer is 18), the model's response is still correct.

Question: {question}

Correct Answer: {answer}

Model Response: {response}

Is the model response correct? Answer yes or no only.
```

`knowledge-update`:
```text
I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response contains some previous information along with an updated answer, the response should be considered as correct as long as the updated answer is the required answer.

Question: {question}

Correct Answer: {answer}

Model Response: {response}

Is the model response correct? Answer yes or no only.
```

`single-session-preference`:
```text
I will give you a question, a rubric for desired personalized response, and a response from a model. Please answer yes if the response satisfies the desired response. Otherwise, answer no. The model does not need to reflect all the points in the rubric. The response is correct as long as it recalls and utilizes the user's personal information correctly.

Question: {question}

Rubric: {answer}

Model Response: {response}

Is the model response correct? Answer yes or no only.
```

`abstention`:
```text
I will give you an unanswerable question, an explanation, and a response from a model. Please answer yes if the model correctly identifies the question as unanswerable. The model could say that the information is incomplete, or some other information is given but the asked information is not.

Question: {question}

Explanation: {answer}

Model Response: {response}

Does the model correctly identify the question as unanswerable? Answer yes or no only.
```

### 3. 指标协议

```yaml
retrieval:
  top_k: must_report

metrics:
  accuracy:
    source: llm_judge
    locomo_positive_label: CORRECT
    longmemeval_positive_label: yes
    unknown: wrong

  f1:
    type: token_level_macro_f1
    normalization: lowercase_then_regex_word_tokens

  bleu:
    type: corpus_bleu_4
    normalization: lowercase_then_regex_word_tokens
 
  by_type:
    group_by: question_type_or_category
    fields: [accuracy, f1, bleu]

  token_cost:
    fields: [build_tokens, query_tokens]

  time_cost:
    fields: [build_time_seconds, query_time_seconds]
```

## 4. 基建 Baseline

当前实现一个最小 naive RAG baseline，作为后续方法迭代的基础代码。

### 4.1 执行流程

```text
原始数据
  -> 统一读取为 Example
  -> 将历史对话切成 chunk
  -> 对 embedding 输入按 max_input_bytes 截断后编码 chunk
  -> 保存 chunk 和 embedding
  -> 对 question 编码；如果样本包含 benchmark question_date，则一起作为检索 query
  -> cosine similarity 检索 top-k chunk
  -> 使用统一 answer prompt 生成答案
  -> 输出 prediction jsonl
  -> 使用统一 judge prompt 评测
  -> 统计 accuracy / F1 / BLEU / token cost / time cost
```

### 4.2 代码结构

```text
agent-memory/
  configs/
    base.yaml              # 公共配置：answer、embedding、judge、retrieval、输出路径

  data/
    locomo10.json               # LoCoMo benchmark 数据
    longmemeval_s_cleaned.json  # LongMemEval benchmark 数据

  core/
    config.py              # 读取和覆盖 yaml 配置
    io.py                  # env、json、jsonl 读写
    schema.py              # Example、Turn、Chunk、RetrievedChunk
    embedding.py           # 本地 vLLM embedding 客户端
    llm.py                 # OpenAI-compatible chat 客户端

  datasets/
    load.py                # 根据 dataset 参数分发到具体 loader
    longmemeval.py         # LongMemEval -> Example
    locomo.py              # LoCoMo -> Example

  prompts/
    answer.py              # answer prompt、LoCoMo category 5 prompt、答案解析
    judge.py               # LoCoMo / LongMemEval judge prompt

  baseline/
    chunking.py            # turn / pair 两种 chunk 方式
    store.py               # 保存和读取 chunk + embedding
    retrieve.py            # cosine top-k 检索
    pipeline.py            # build_memory 和 answer 主逻辑

  evaluation/
    judge.py               # 调 judge 模型生成 judge_label
    metrics.py             # 统计 F1、BLEU、token/time cost

  run_baseline.py          # baseline 命令行入口，按 memory_id 分组并行
```

### 4.3 配置入口

默认配置文件：

```text
configs/base.yaml
```

命令行中可以临时覆盖部分配置。下面这个例子不会修改 `base.yaml`，只会让本次运行使用 `top_k=20`、`chunk_unit=turn`，并把输出写到指定路径：

```bash
python -m run_baseline \
  --config configs/base.yaml \
  --dataset longmemeval \
  --data data/longmemeval_s_cleaned.json \
  --top-k 20 \
  --chunk-unit turn \
  --out outputs/baseline/predictions.jsonl \
  --store-root outputs/baseline/stores \
  --log-file outputs/logs/baseline.log \
  --mode full \
  --workers 4 \
  --overwrite
```

### 4.4 运行 Baseline

LongMemEval：

```bash
python -m run_baseline \
  --dataset longmemeval \
  --data data/longmemeval_s_cleaned.json \
  --out outputs/baseline/longmemeval_predictions.jsonl \
  --store-root outputs/baseline/longmemeval_stores \
  --log-file outputs/logs/longmemeval_run.log \
  --mode full \
  --top-k 20 \
  --workers 4 \
  --overwrite
```

LoCoMo：

```bash
python -m run_baseline \
  --dataset locomo \
  --data data/locomo10.json \
  --out outputs/baseline/locomo_predictions.jsonl \
  --store-root outputs/baseline/locomo_stores \
  --log-file outputs/logs/locomo_run.log \
  --mode full \
  --top-k 20 \
  --workers 2 \
  --overwrite
```

默认并行数按数据集设置：LongMemEval 为 4，LoCoMo 为 2。也可以通过 `--workers` 手动覆盖。

只构建 memory：

```bash
python -m run_baseline \
  --dataset longmemeval \
  --data data/longmemeval_s_cleaned.json \
  --store-root outputs/baseline/longmemeval_stores \
  --log-file outputs/logs/longmemeval_build.log \
  --mode build \
  --workers 4 \
  --overwrite
```

只生成答案：

```bash
python -m run_baseline \
  --dataset longmemeval \
  --data data/longmemeval_s_cleaned.json \
  --out outputs/baseline/longmemeval_predictions.jsonl \
  --store-root outputs/baseline/longmemeval_stores \
  --log-file outputs/logs/longmemeval_query.log \
  --mode query \
  --workers 4
```

### 4.5 评测

先用 judge 模型打标签：

```bash
python -m evaluation.judge \
  --pred outputs/baseline/longmemeval_predictions.jsonl \
  --out outputs/baseline/longmemeval_predictions.judge.jsonl \
  --log-file outputs/logs/longmemeval_judge.log \
  --overwrite
```

再统计指标：

```bash
python -m evaluation.metrics \
  --pred outputs/baseline/longmemeval_predictions.judge.jsonl \
  --out outputs/baseline/longmemeval_metrics.md \
  --format markdown
```

### 4.6 输出字段

prediction jsonl 中每一行对应一个 QA。正常成功样本包含：

```text
sample_id                 # QA id
memory_id                 # 共享 memory 的 id；LoCoMo 中同一 conversation 共享同一个 memory_id
dataset                   # longmemeval 或 locomo
question                  # 当前问题
question_date             # benchmark 给定的问题日期，不使用系统当前日期
question_type             # 题目类型
answer                    # reference answer
hypothesis                # answer 模型生成的最终答案
raw_response              # answer 模型原始输出
method                    # naive_rag_baseline
model                     # answer 模型
embedding_model           # embedding 模型
chunk_unit                # turn / pair
top_k                     # 检索返回数量
num_chunks                # 当前 memory 中的 chunk 数
build_tokens              # memory 构建阶段 token
query_tokens              # query + answer 阶段 token
build_time_seconds        # memory 构建耗时
query_time_seconds        # 当前 QA 检索和回答耗时
error                     # 成功时为 null
```

judge 后的 jsonl 会在原 prediction 字段基础上额外增加：

```text
judge_model               # judge 模型
judge_response            # judge 模型原始输出
judge_label               # true / false / null
judge_tokens              # judge 阶段 token
```

metrics 文件默认建议保存为 Markdown 表格，包含整体指标和按题目类型分组的指标：

```text
outputs/baseline/longmemeval_metrics.md  # LongMemEval 指标表
outputs/baseline/locomo_metrics.md       # LoCoMo 指标表
```
