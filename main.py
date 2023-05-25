import os
import random
import shutil
from io import BytesIO
import telebot
#import cv2
#from dotenv import load_dotenv
import detect
from telebot import types
import telebot
import time

import detectFormula
import detectSymb


bot = telebot.TeleBot('TOKEN');
def translate(word):
	u = word.split()
	x = {'mult': '*', 'div': '/', 'minus': '-', 'plus': '+', 'eqv': '=', 'LBracket': '(', 'RBracket': ')', 'integ': chr(8747), 'other_class': ""};

	a = ''
	values = x.keys()

	for i in range(len(u)):

		if u[i] in values:
			w = x[u[i]]
		else:
			w = u[i]

		a += w+" "

	return a



t0 = time.time()

@bot.message_handler(commands=['start'])
def start_message(message):
	bot.send_message(message.chat.id, 'Привет! Пришлите ваше фото!\nДокументы не принимаются!')

@bot.message_handler(content_types=['text'])
def send_welcome(message):
	bot.send_message(message.chat.id,'Привет! Пришлите ваше фото!\nДокументы не принимаются!');

@bot.message_handler(content_types=['photo'])

def handle_docs_photo(message):

	try:


		start = time.time()
		file_info = bot.get_file(message.photo[len(message.photo) - 1].file_id)
		downloaded_file = bot.download_file(file_info.file_path)

		dire1 = "photos"
		for f in os.listdir(dire1):
			os.remove(os.path.join(dire1, f))

		src =  file_info.file_path;
		with open(src, 'wb') as new_file:
			new_file.write(downloaded_file)
		bot.reply_to(message, "Фото добавлено")
		bot.send_message(message.chat.id, 'Начинаю поиск формулы');

		'''dire = "detect\\exp"
		for f in os.listdir(dire):
			os.remove(os.path.join(dire, f))'''

		path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'detectFormula\\exp')
		shutil.rmtree(path)
		detectFormula.run()

		#img_res = BytesIO(open("detect\\exp\\file_11.jpg", "rb").read())
		#bot.send_photo(message.chat.id, img_res)

		#photo = open('detect\\exp\\crops\\' + os.listdir('detect\\exp\\crops\\')[0], 'rb')
		#bot.send_photo(message.from_user.id, photo)
		if len(os.listdir('detectFormula\\exp\\')) == 1:
			bot.send_message(message.chat.id, 'Формул не обнаружено');
		else:
			path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'detectSymb\\expSymb')
			shutil.rmtree(path)

			detectSymb.runSymb()

			photo = open('detectSymb\\expSymb\\' + os.listdir('detectSymb\\expSymb\\')[0], 'rb')
			#bot.send_photo(message.from_user.id, photo)
			q = []
			with open("tut.txt", "r") as file1:
				# итерация по строкам
				for line in file1:
					s = line.strip()


					q.append(s)

			m = s.split(" ")
			a = ''
			d = translate(s)
			q = q[1:]

			for i in range(len(q)):
				photo = open('detectFormula\\exp\\crops\\' + os.listdir('detectFormula\\exp\\crops\\')[i], 'rb')
				bot.send_photo(message.from_user.id, photo)
				if translate(q[i]) == "":
					bot.send_message(message.chat.id, 'no detections');
				else:
					bot.send_message(message.chat.id, translate(q[i]));
			print(q)

			bot.send_message(message.chat.id, 'Готово!');
			end = time.time() - start  ## собственно время работы программы

			print(end)

	except Exception as e:

		#bot.reply_to(message, e)
		#print(e)
		bot.reply_to(message, "Произошла ошибка обнаружения. Попробуйте загрузить другую фотографию.")
		print(e)


@bot.message_handler(content_types=["audio", "document"])
def no_photo(message):
	bot.send_message(message.chat.id,'Я принимаю только фотографии, не документы\nЕсли Вы хотели отправить фото, а получился документ, пожалуйста, поставьте галочку в окошко "Сжать изображение"');

@bot.message_handler(func=lambda message: True)
def echo_all(message):
	bot.reply_to(message, message.text)

bot.infinity_polling()
