from tflite_support.task.text import BertQuestionAnswerer

# Create the BertQuestionAnswerer object from a TensorFlow lite model
question_answerer = BertQuestionAnswerer.create_from_file("./path_to_file")

# Create context for question answering
context = "Alexander the Great was a king of the ancient Greek kingdom of Macedon.[a] He succeeded his father Philip II to the throne in 336 BC at the age of 20, and spent most of his ruling years conducting a lengthy military campaign throughout Western Asia and Egypt. By the age of thirty, he had created one of the largest empires in history, stretching from Greece to northwestern India.[2] He was undefeated in battle and is widely considered to be one of history's greatest and most successful military commanders. Until the age of 16, Alexander was tutored by Aristotle. In 335 BC, shortly after his assumption of kingship over Macedon, he campaigned in the Balkans and reasserted control over Thrace and Illyria before marching on the city of Thebes, which was subsequently destroyed in battle. Alexander then led the League of Corinth, and used his authority to launch the pan-Hellenic project envisaged by his father, assuming leadership over all Greeks in their conquest of Persia."

# Create questions
questions = {
     "Who is Alexander the Great": None,
     "Until the age of 16, who tutored Alexander": None,
}

for question, _ in questions.items():
    answer = question_answerer.answer(context, question)
    questions[question] = answer

print(questions)
