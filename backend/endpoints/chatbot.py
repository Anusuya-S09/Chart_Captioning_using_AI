# from transformers import pipeline

# class TransformersChatBot:
#     def __init__(self, csv_data, text_info):
#         self.nlp = pipeline('question-answering')
#         self.csv_data = csv_data
#         self.text_info = text_info

#     def respond(self, question):
#         if 'chart' in question.lower():
#             return self.text_info
#         elif 'data' in question.lower():
#             return self.handle_data_query(question)
#         else:
#             return "I'm sorry, I don't understand the question."

#     def handle_data_query(self, question):
#         # Example: Use a model to extract information related to the question
#         for row in self.csv_data:
#             # Placeholder for a more advanced query handling
#             if any(keyword.lower() in question.lower() for keyword in row.values()):
#                 return str(row)
#         return "No data found related to your question."

# # Example usage
# csv_data = load_data_from_csv('extracted_data_example.csv')
# text_info = load_info_from_text('chart_info_example.txt')
# preprocessed_data = preprocess_data(csv_data)

# chatbot = TransformersChatBot(preprocessed_data, text_info)

# # Simulating a chat
# print(chatbot.respond("Tell me about the chart type."))
# print(chatbot.respond("What data is available about the PieChart?"))
