/*global webkitSpeechRecognition */
var tabla = undefined;
(function() {
	'use strict';

	// check for support (webkit only)
	if (! ('webkitSpeechRecognition' in window) ) return;

	var talkMsg = 'Diga su consulta';

	// seconds to wait for more input after last
	var patience = 5;

	function capitalize(str) {
		return str.length ? str[0].toUpperCase() + str.slice(1) : str;
	}

	var inputEls = document.getElementsByClassName('speech-input');

	[].forEach.call(inputEls, function(inputEl) {
		// create wrapper
		var wrapper = document.createElement('div');
		wrapper.classList.add('si-wrapper');

		// create mic button
		var micBtn = document.createElement('button');
		micBtn.classList.add('si-btn');
		micBtn.textContent = 'speech input';
		var micIcon = document.createElement('span');
		var holderIcon = document.createElement('span');
		micIcon.classList.add('si-mic');
		holderIcon.classList.add('si-holder');
		micBtn.appendChild(micIcon);
		micBtn.appendChild(holderIcon);

		// gather inputEl data
		var nextNode = inputEl.nextSibling;
		var parent = inputEl.parentNode;
		var inputHeight = inputEl.offsetHeight;
		var inputRightBorder = parseInt(getComputedStyle(inputEl).borderRightWidth, 10);
		var buttonSize = 0.8 * inputHeight;
		// Size bounds (useful for textareas).
		if (buttonSize > 50) buttonSize = 50;

		// append mic and input to wrapper
		wrapper.appendChild(parent.removeChild(inputEl));
		wrapper.appendChild(micBtn);


		
		// size and position mic and input
		micBtn.style.top = 0.1 * inputHeight + 'px';
		micBtn.style.height = micBtn.style.width = buttonSize + 'px';
		inputEl.style.paddingRight = buttonSize - inputRightBorder + 'px';

		// append wrapper where input was
		parent.insertBefore(wrapper, nextNode);
		
		// setup recognition
		var finalTranscript = '';
		var textoFijo = 'Has dicho... ';
		var recognizing = false;
		var timeout;
		var oldPlaceholder = null;
		var recognition = new webkitSpeechRecognition();
		recognition.continuous = true;
		recognition.lang="es-ES";

		function restartTimer() {
			timeout = setTimeout(function() {
				recognition.stop();
			}, patience * 1000);
		}

		recognition.onstart = function() {
			oldPlaceholder = inputEl.placeholder;
			inputEl.placeholder = talkMsg;
			recognizing = true;
			micBtn.classList.add('listening');
			restartTimer();
		};

		recognition.onend = function() {
			recognizing = false;
			clearTimeout(timeout);
			micBtn.classList.remove('listening');
			if (oldPlaceholder !== null) inputEl.placeholder = oldPlaceholder;
		};

		recognition.onresult = function(event) {
			var video = document.getElementById("Video1");
			var textoBocadillo = document.getElementById("bocadillo").textContent;
			var llamas = "llamas";
			var textoIntroductorio = "Me llamo everisa, y estoy aquí para ayudarte a encontrar ofertas de trabajo";
			var parametros = {
				//onstart: IniciaVideo,
				onend: FinalizarVideo
			}
			var parametros2 = {
				//onstart: IniciaVideo,
				onend: FinalizarVideoSinBuscar
			}
		
			clearTimeout(timeout);
			for (var i = event.resultIndex; i < event.results.length; ++i) {
				if (event.results[i].isFinal) {
					finalTranscript += event.results[i][0].transcript;
				}
			}
			finalTranscript = capitalize(finalTranscript);
			inputEl.value = finalTranscript;
			$("#bocadillo").text(textoFijo + finalTranscript)
			document.getElementById("bocadillo").style.display = 'block';

			video.play();
						
			if (finalTranscript.indexOf(llamas) != -1)
			{
				responsiveVoice.speak(textoIntroductorio, "Spanish Female", parametros2);
			}
			else
			{
				responsiveVoice.speak(textoFijo + finalTranscript, "Spanish Female", parametros);
			}
	
			//
			

			restartTimer();

		};
		
		function IniciaVideo(text) {
			var videoafinalizar = document.getElementById("Video1");
			videoafinalizar.play();
		
		};

		function FinalizarVideo(text, transcript) {
			var videoafinalizar = document.getElementById("Video1");
			videoafinalizar.pause();
			//$('#bocadillo').change();
			sendTranscription(finalTranscript);
		
		};


	
		micBtn.addEventListener('click', function(event) {
			//document.getElementById("bocadillo").style.display = 'none';
			event.preventDefault();
			if (recognizing) {
				recognition.stop();
				return;
			}
			inputEl.value = finalTranscript = '';
			
			recognition.start();
		}, false);
	});
})();

		function LeeTitulosOfertas(){
			
			var video = document.getElementById("Video1");
			var parametros2 = {
				onend: FinalizarVideo2
			}
			
			video.play();
			
			var tabla2 = document.getElementById("tablaBusquedaEntidadHabilitada");
			var tdsTabla2 = tabla2.getElementsByTagName("td");
			var texto = "";
			
			texto = "Se han encontrado 3 resultados. Los resultados encontrados son: ";
			var i =0;
			for (i=1; i<tdsTabla2.length; i=i+4){
				if (i+4 > tdsTabla2.length) {
					texto = texto + " y " + tdsTabla2[i].textContent;
				}
				else
				{
					texto = texto + tdsTabla2[i].textContent + ", ";
				}				
			}
			responsiveVoice.speak(texto, "Spanish Female", parametros2);
			
	
				};
		function LeeDetalleOferta(DetalleOferta){
			
			var video = document.getElementById("Video1");
			var parametros3 = {
				onend: FinalizarVideo3
			}
			
			video.play();			
			responsiveVoice.speak(DetalleOferta, "Spanish Female", parametros3);
			
	
				};
				
		function FinalizarVideo2(text) {
			var videoafinalizar = document.getElementById("Video1");
			videoafinalizar.pause();
		
		};
		
		function FinalizarVideo3(text) {
			var videoafinalizar = document.getElementById("Video1");
			videoafinalizar.pause();
		
		};

		function insertaRegistrosEnTabla () {
			var ContenidoPrevioTabla = document.getElementById('tablaBusquedaEntidadHabilitada').innerHTML;
			var InformacionAInsertar = ContenidoPrevioTabla + "<tr><td><img src=\"/static/images/icono_informacion.png\" id=\"mas_detalle\" width=\"40\" height=\"40\" onclick='LeeDetalleOferta(\"Descripción Oferta 1\")' style=\"cursor: pointer\"></td><td>Título Oferta 1</td><td>Descripción Oferta 1</td><td>Requerimientos Oferta 1</td></tr>";
			InformacionAInsertar = InformacionAInsertar + "<tr><td><img src=\"/static/images/icono_informacion.png\" id=\"mas_detalle\" width=\"40\" height=\"40\" onclick='LeeDetalleOferta(\"Descripción Oferta 2\")' style=\"cursor: pointer\"></td><td>Título Oferta 2</td><td>Descripción Oferta 2</td><td>Requerimientos Oferta 2</td></tr>";
			InformacionAInsertar = InformacionAInsertar + "<tr><td><img src=\"/static/	/icono_informacion.png\" id=\"mas_detalle\" width=\"40\" height=\"40\" onclick='LeeDetalleOferta(\"Descripción Oferta 3\")' style=\"cursor: pointer\"></td><td>Título Oferta 3</td><td>Descripción Oferta 3</td><td>Requerimientos Oferta 3</td></tr>";
			document.getElementById('tablaBusquedaEntidadHabilitada').innerHTML = InformacionAInsertar; 

				};



		function contarResultados (num) {
			var video = document.getElementById("Video1");
			video.play();
			switch (num) {
			  case 0:
				responsiveVoice.speak("Lo siento, no he encontrado ningún resultado", "Spanish Female", {onend: FinalizarVideoSinBuscar});
			    break;
			  case 1:
			 	responsiveVoice.speak("He encontrado un resultado", "Spanish Female", {onend: FinalizarVideoSinBuscar});
			  	break;
			  default:
			 	responsiveVoice.speak("He encontrado "+num+" resultados", "Spanish Female", {onend: FinalizarVideoSinBuscar});
			    break;
			}
			var texto = "Las primeras ofertas encontradas son: ";
			if($(".sorting_1").length!==0){
				for(var i=0; i<=2;i++){
					texto += $(".sorting_1")[i].children[0].innerHTML + ".,.    "
				}
				responsiveVoice.speak(texto, "Spanish Female", {onend: FinalizarVideoSinBuscar});
			}
		}



		
		function FinalizarVideoSinBuscar(text) {
			var videoafinalizar = document.getElementById("Video1");
			videoafinalizar.pause();
			//$('#bocadillo').change();
		
		};



