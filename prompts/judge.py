from __future__ import annotations

from src.memory.core.schema import Example


LOCOMO_JUDGE_TEMPLATE = """Your task is to label an answer to a question as 'CORRECT' or 'WRONG'. You will be given the following data:
    (1) a question (posed by one user to another user),
    (2) a 'gold' (ground truth) answer,
    (3) a generated answer
which you will score as CORRECT/WRONG.

The point of the question is to ask about something one user should know about the other user based on their prior conversations.
The gold answer will usually be a concise and short answer that includes the referenced topic. The generated answer might be much longer, but you should be generous with your grading - as long as it touches on the same topic as the gold answer, it should be counted as CORRECT.

For time related questions, the gold answer will be a specific date, month, year, etc. The generated answer might be much longer or use relative time references, but you should be generous with your grading - as long as it refers to the same date or time period as the gold answer, it should be counted as CORRECT.

Now it's time for the real question:
Question: {question}
Gold answer: {gold_answer}
Generated answer: {generated_answer}

Return ONLY a valid JSON object in the following format:
{{
  "label": "CORRECT"
}}

The value of "label" must be exactly "CORRECT" or "WRONG". Do not include any explanation or extra text.
"""


LONGMEMEVAL_DEFAULT_TEMPLATE = """I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no.

Question: {question}

Correct Answer: {answer}

Model Response: {response}

Is the model response correct? Answer yes or no only.
"""


LONGMEMEVAL_TEMPORAL_TEMPLATE = """I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. In addition, do not penalize off-by-one errors for the number of days. If the question asks for the number of days/weeks/months, etc., and the model makes off-by-one errors, the model's response is still correct.

Question: {question}

Correct Answer: {answer}

Model Response: {response}

Is the model response correct? Answer yes or no only.
"""


LONGMEMEVAL_UPDATE_TEMPLATE = """I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response contains some previous information along with an updated answer, the response should be considered as correct as long as the updated answer is the required answer.

Question: {question}

Correct Answer: {answer}

Model Response: {response}

Is the model response correct? Answer yes or no only.
"""


LONGMEMEVAL_PREFERENCE_TEMPLATE = """I will give you a question, a rubric for desired personalized response, and a response from a model. Please answer yes if the response satisfies the desired response. Otherwise, answer no. The model does not need to reflect all the points in the rubric. The response is correct as long as it recalls and utilizes the user's personal information correctly.

Question: {question}

Rubric: {answer}

Model Response: {response}

Is the model response correct? Answer yes or no only.
"""


LONGMEMEVAL_ABSTENTION_TEMPLATE = """I will give you an unanswerable question, an explanation, and a response from a model. Please answer yes if the model correctly identifies the question as unanswerable. The model could say that the information is incomplete, or some other information is given but the asked information is not.

Question: {question}

Explanation: {answer}

Model Response: {response}

Does the model correctly identify the question as unanswerable? Answer yes or no only.
"""


def judge_messages(example: Example, generated_answer: str) -> list[dict[str, str]]:
    if example.dataset == "locomo":
        prompt = LOCOMO_JUDGE_TEMPLATE.format(
            question=example.question,
            gold_answer=example.answer,
            generated_answer=generated_answer,
        )
    else:
        prompt = _longmemeval_template(example.question_type).format(
            question=example.question,
            answer=example.answer,
            response=generated_answer,
        )
    return [{"role": "user", "content": prompt}]


def _longmemeval_template(question_type: str) -> str:
    if question_type == "temporal-reasoning":
        return LONGMEMEVAL_TEMPORAL_TEMPLATE
    if question_type == "knowledge-update":
        return LONGMEMEVAL_UPDATE_TEMPLATE
    if question_type == "single-session-preference":
        return LONGMEMEVAL_PREFERENCE_TEMPLATE
    if question_type == "abstention":
        return LONGMEMEVAL_ABSTENTION_TEMPLATE
    return LONGMEMEVAL_DEFAULT_TEMPLATE

