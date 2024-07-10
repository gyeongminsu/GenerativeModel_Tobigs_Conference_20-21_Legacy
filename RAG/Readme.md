### Prompt 생성 방법론
1. 생성 AI 모델에 넣기 전에 Prompt를 더 잘 만들 수 있을지를 고민(좋은 Prompt가 좋은 생성 모델의 결과를 만들기 때문)
2. 처음에는 기존에 존재하는 생성 모델로 만든 이미지 결과를 보고 잘 만들어진 이미지의 Prompt를 추려 데이터화한 후 RAG기법을 이용해 Prompt를 강화하려 했음
3. 하지만 실제로 DiffusionDB같이 생성 모델로 만들어진 이미지-Prompt 쌍 데이터를 휴리스틱하게 확인해본 결과 초창기 생성 모델로 만들어진 결과이기에 전체적인 성능이 좋지 않고, 생성된 이미지에 대한 평가가 주관적이기 때문에 “잘 만들어진 Prompt-이미지 쌍 데이터”를 만들기가 어려움(실제로 팀원 7명이 나누어 정성평가를 진행)
4. 따라서 기존 RAG 방식의 Retrieval을 제하고 Generation 방법론을 강화하여 좋은 Prompt를 만들고자 함
5. 그렇게 사용하게 된 것이 Instruction Tuning
6. Instruction Tuning은 LLM이 특정 작업이나 명령을 더 잘 수행하도록 하기 위해 사용하는 방법으로 모델이 주어진 Instruction을 이해하고 그에 따라 행동할 수 있도록 학습시키는 것을 말함
7. 대표적인 LLM 모델들은 이런식의 Instruction Tuning으로 학습한 Instruct 모델을 배포하고 있는데 우리는 llam3-8b-instruct 모델을 사용(gpt-3.5-instruct도 사용해 보았지만 성능이 좋지 않음)
8. 또한 여기에 CoF(Chain-of-Thought) 프롬프팅을 사용하여 모델이 더 풍부한 Prompt를 만들 수 있도록 만듦
9. 그렇게 나온 Prompt 예시 :
```
Query :
"A profile sitting next to a tree with a Christmas mood."
Answer : 
"Create an AI pet profile with a picture of my dog, situated next to a Christmas tree with festive decorations, wearing a festive collar, and surrounded by snowflakes, conveying a joyful and playful atmosphere."
```
