from search_pipeline.rag_chain import build_rag_chain

print("RAG Chatbot Ready!")
print("Type 'exit' to quit.\n")
ragchain = build_rag_chain()
while True:
    query = input("\nAsk your question: ")

    if query.lower() == "exit":
        break

    
    ans=ragchain(query)

    print("\n Final Answer:\n")
    print(ans["answer"])

    print("-" * 50)