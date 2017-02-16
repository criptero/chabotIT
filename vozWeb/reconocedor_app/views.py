from django.shortcuts import render
from django.http import JsonResponse
import query as q
import sys
import commands


# Create your views here.
def index(request):
	return render(request, 'reconocedor/index.html')

def getResultsAjax(request):
	resp_data = {}
	transcription = request.GET.get("transcription").encode("utf-8")
	transcription = transcription.replace(" ", "$")
	result=commands.getoutput('/bin/bash /var/www/vozWeb/reconocedor_app/algoritmo.sh \''+transcription+'\'')
	print result
	start = result.find("<query>")
	start = int(start)+7
	end = result.find("</query>")
	result = result[start:int(end)]
	resp_data ["data"] = q.make_sql_sentence(result)
	#print resp_data
	return JsonResponse(resp_data) 

def textVoice(request):
	return render(request, 'reconocedor/templates/pages/textVoice.html')
	
def voiceText(request):
	return render(request, 'reconocedor/templates/pages/voiceText.html')	

def voiceCommand(request):
	return render(request, 'reconocedor/templates/pages/voiceCommand.html')
  
def buscadorofertas(request):
	return render(request, 'reconocedor/templates/pages/buscadorofertas.html')		
	
def buscadorPruebas(request):
	return render(request, 'reconocedor/templates/pages/buscadorPruebas.html')	

def buscadorPruebas2(request):
	return render(request, 'reconocedor/templates/pages/buscadorPruebas2.html')	
