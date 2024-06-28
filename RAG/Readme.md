### Rag 방법론
1. DiffusionDB 데이터 정제
- "Dog" 단어가 있는 Prompt만 뽑아 이미지가 잘 만들어졌는지를 정성적으로 평가  
- 해당 이미지만 사용  

2. Retriever
- Retriever 모델을 사용하여 사용자가 원하는 스타일과 가장 비슷한 Prompt를 찾음

3. Generator
- OpenAI의 GPT API를 사용  
- 사용자가 원하는 스타일과 Retriever 모델의 결과로 나온 두 자연어를 모두 활용하여 생성모델에 입력으로 들어갈 Prompt를 생성
