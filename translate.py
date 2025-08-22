from googletrans import Translator

translator = Translator()
result = translator.translate("吸血鬼… だね ", src="ja", dest="vi")
print(result.text)  # Kết quả: "Xin chào"
