function mover_video(){
	var video = document.getElementById("Video1");
    video.play();
    button.textContent = "||";
}

function esperar_video(responsiveVoice){
	while(responsiveVoice.isPlaying()){
	}
}

function parar_video(){
	var video = document.getElementById("Video1");
    video.pause();
    button.textContent = ">";
   
	
}
