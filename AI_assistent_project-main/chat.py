from src.rag.pipeline import RAGPipeline

print("Загрузка ассистента...")
rag = RAGPipeline()

print("\n==============================================")
print("Ассистент готов! Задавайте вопросы.")
print("Чтобы выйти, напишите 'exit' или 'выход'.")
print("==============================================\n")

while True:
    question = input("\nВаш вопрос: ")
    
    if question.lower() in ['exit', 'quit', 'выход']:
        print("До свидания!")
        break
        
    if not question.strip():
        continue
        
    print("Ищу ответ в документах...")
    
    try:
        result = rag.ask(question, top_k=4)
        
        print("\n--- ОТВЕТ ИИ ---")
        print(result["answer"])
        
        # Проверяем, не отказалась ли нейросеть отвечать
        answer_lower = result["answer"].lower()
        if "не знаю" in answer_lower or "нет информации" in answer_lower or "не могу" in answer_lower:
            print("\n(Система не нашла подходящих источников в базе)")
        else:
            print("\n--- ИСТОЧНИКИ ---")
            for i, chunk in enumerate(result["chunks"], 1):
                title = chunk['metadata'].get('document_title', 'Неизвестно')
                doc_id = chunk['metadata'].get('document_id', '')
                print(f"{i}. {title} ({doc_id})")
            
        print("\n" + "-"*50)
        
    except Exception as e:
        print(f"\n[!] Произошла ошибка: {e}")
