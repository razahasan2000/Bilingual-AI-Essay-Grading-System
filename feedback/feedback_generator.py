import re

class FeedbackGenerator:
    def __init__(self):
        pass

    def detect_language(self, text):
        # Basic check for Arabic characters
        if re.search(r'[\u0600-\u06FF]', text):
            return "arabic"
        return "english"

    def generate_feedback(self, text, score):
        lang = self.detect_language(text)
        
        # Simulated analysis metrics
        word_count = len(text.split())
        sent_count = len(re.split(r'[.!?\u06D4]', text))
        unique_words = len(set(text.split()))
        lexical_diversity = unique_words / (word_count + 1)

        feedback = {
            "score": round(score, 1),
            "feedback": {
                "grammar": self._get_grammar_feedback(lang, score),
                "content": self._get_content_feedback(lang, word_count, score),
                "structure": self._get_structure_feedback(lang, sent_count, score),
                "vocabulary": self._get_vocab_feedback(lang, lexical_diversity, score),
                "suggestions": self._get_suggestions(lang, score)
            }
        }
        return feedback

    def _get_grammar_feedback(self, lang, score):
        if lang == "arabic":
            if score > 8: return "القواعد النحوية ممتازة وتركيب الجمل سليم تماماً."
            if score > 5: return "القواعد جيدة بشكل عام، مع وجود بعض أخطاء التشكيل أو التوافق بين الفعل والفاعل."
            return "هناك حاجة ماسة لمراجعة قواعد النحو الأساسية والتركيز على بنية الجملة."
        else:
            if score > 8: return "Excellent grammar and syntax. Your sentences are complex and well-formed."
            if score > 5: return "Good grammar overall, but there are some minor subject-verb agreement issues."
            return "Needs significant improvement in basic grammar and sentence construction."

    def _get_content_feedback(self, lang, words, score):
        if lang == "arabic":
            if words > 200: return "المحتوى ثري ويغطي الموضوع بشكل شامل."
            return "المحتوى جيد ولكن يمكن التوسع في طرح الأفكار وتقديم المزيد من الأدلة."
        else:
            if words > 200: return "Rich content that addresses the prompt comprehensively."
            return "Good content, but you could expand on your ideas with more evidence."

    def _get_structure_feedback(self, lang, sents, score):
        if lang == "arabic":
            if sents > 10: return "التنظيم الهيكلي للمقال ممتاز مع انتقال سلس بين الفقرات."
            return "الهيكل مقبول، لكن حاول ربط الجمل ببعضها البعض باستخدام أدوات الربط المناسبة."
        else:
            if sents > 10: return "Well-structured with a clear introduction, body, and conclusion."
            return "Acceptable structure, but try to use more transition words to link your ideas."

    def _get_vocab_feedback(self, lang, diversity, score):
        if lang == "arabic":
            if diversity > 0.6: return "استخدام مفردات لغوية راقية ومتنوعة."
            return "المفردات جيدة، حاول استخدام كلمات أكثر تخصصاً لإثراء النص."
        else:
            if diversity > 0.6: return "Impressive vocabulary with sophisticated word choices."
            return "Good vocabulary, but try to use more academic terms to enhance the essay."

    def _get_suggestions(self, lang, score):
        if lang == "arabic":
            if score > 8: return "استمر في القراءة لزيادة ثرائك اللغوي وتجربة أساليب بلاغية جديدة."
            return "ركز على مراجعة أدوات الربط والتدقيق الإملائي وتوسيع قاعدة مفرداتك."
        else:
            if score > 8: return "Keep reading broadly to incorporate even more varied rhetorical devices."
            return "Focus on proofreading for minor errors and expanding your academic vocabulary."
