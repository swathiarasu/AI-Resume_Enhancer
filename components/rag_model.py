
import google.generativeai as genai
import os

class RAGModel:
    def __init__(self, model_name="models/gemini-2.5-flash-preview-04-17"):
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        self.model = genai.GenerativeModel(model_name)

    def generate_answer(self, question, context):
        prompt = f"""You are an AI-powered Resume Enhancer. You are provided with user resume details and other relevant context whenever a question is asked. This context can include details like Skills, Experience, Education, Certifications, Job Role, Salary Expectation, Projects Count, and an AI Score.

        Use this context to provide personalized suggestions for enhancing the user's resume. These suggestions should focus on improving skills, highlighting relevant experiences, and making the resume more appealing to recruiters.

        Context:
        {context}

        Given the above context, answer the following question with actionable advice:
        {question}
        """
        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            return f" Gemini API error: {e}"
