
from  django.conf.urls import url

from . import views

urlpatterns=[
	url(r'^index$', views.index, name='index'),
	url(r'^getResultsAjax', views.getResultsAjax, name='getResultsAjax'),
	url(r'^textVoice$', views.textVoice, name='textVoice'),
	url(r'^voiceText$', views.voiceText, name='voiceText'),
	url(r'^buscadorofertas$', views.buscadorofertas, name='buscadorofertas'),
	url(r'^buscadorPruebas$', views.buscadorPruebas, name='buscadorPruebas'),
	url(r'^buscadorPruebas2$', views.buscadorPruebas2, name='buscadorPruebas2'),
	url(r'^voiceCommand$', views.voiceCommand, name='voiceCommand'),
	]